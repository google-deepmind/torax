# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""bootstrap current source profile."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import chex
import jax
from jax import numpy as jnp
from jax.scipy import integrate
from torax import constants
from torax import jax_utils
from torax import physics
from torax import state
from torax.config import runtime_params_slice
from torax.fvm import cell_variable
from torax.geometry import geometry
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source
from torax.sources import source_profiles


@dataclasses.dataclass(kw_only=True)
class RuntimeParams(runtime_params_lib.RuntimeParams):
  """Configuration parameters for the bootstrap current source."""

  # Multiplication factor for bootstrap current
  bootstrap_mult: float = 1.0
  mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED

  def make_provider(
      self,
      torax_mesh: geometry.Grid1D | None = None,
  ) -> RuntimeParamsProvider:
    return RuntimeParamsProvider(**self.get_provider_kwargs(torax_mesh))


@chex.dataclass
class RuntimeParamsProvider(runtime_params_lib.RuntimeParamsProvider):
  """Provides runtime parameters for a given time and geometry."""

  runtime_params_config: RuntimeParams

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(**self.get_dynamic_params_kwargs(t))


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  bootstrap_mult: float


def _default_output_shapes(geo) -> tuple[int, int, int, int]:
  return (
      source.ProfileType.CELL.get_profile_shape(geo)  # sigmaneo
      + source.ProfileType.CELL.get_profile_shape(geo)  # bootstrap
      + source.ProfileType.FACE.get_profile_shape(geo)  # bootstrap face
      + (1,)  # I_bootstrap
  )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class BootstrapCurrentSource(source.Source):
  """Bootstrap current density source profile.

  Unlike other sources within TORAX, the BootstrapCurrentSource provides more
  than just the bootstrap current. It uses a neoclassical physics model to
  also determine the neoclassical conductivity. Outputs are as follows:

  - sigmaneo: neoclassical conductivity
  - bootstrap current (on cell and face grids)
  - total integrated bootstrap current
  """

  SOURCE_NAME: ClassVar[str] = 'j_bootstrap'

  @property
  def source_name(self) -> str:
    return self.SOURCE_NAME

  @property
  def supported_modes(self) -> tuple[runtime_params_lib.Mode, ...]:
    return (
        runtime_params_lib.Mode.ZERO,
        runtime_params_lib.Mode.MODEL_BASED,
        runtime_params_lib.Mode.PRESCRIBED,
    )

  @property
  def output_shape_getter(self) -> source.SourceOutputShapeFunction:
    return _default_output_shapes

  @property
  def affected_core_profiles(self) -> tuple[source.AffectedCoreProfile, ...]:
    return (source.AffectedCoreProfile.PSI,)

  def get_value(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> source_profiles.BootstrapCurrentProfile:
    static_source_runtime_params = static_runtime_params_slice.sources[
        self.source_name
    ]
    dynamic_source_runtime_params = dynamic_runtime_params_slice.sources[
        self.source_name
    ]
    # Make sure the input mode requested is supported.
    self.check_mode(static_source_runtime_params.mode)
    # Make sure the input params are the correct type.
    if not isinstance(dynamic_source_runtime_params, DynamicRuntimeParams):
      raise TypeError(
          'Expected DynamicRuntimeParams, got '
          f'{type(dynamic_source_runtime_params)}.'
      )
    bootstrap_current = calc_neoclassical(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        temp_ion=core_profiles.temp_ion,
        temp_el=core_profiles.temp_el,
        ne=core_profiles.ne,
        ni=core_profiles.ni,
        psi=core_profiles.psi,
    )
    zero_profile = source_profiles.BootstrapCurrentProfile.zero_profile(geo)
    is_zero_mode = (
        static_source_runtime_params.mode == runtime_params_lib.Mode.ZERO.value
    )
    return source_profiles.BootstrapCurrentProfile(
        sigma=bootstrap_current.sigma,
        sigma_face=bootstrap_current.sigma_face,
        j_bootstrap=jax_utils.select(
            is_zero_mode,
            zero_profile.j_bootstrap,
            bootstrap_current.j_bootstrap,
        ),
        j_bootstrap_face=jax_utils.select(
            is_zero_mode,
            zero_profile.j_bootstrap_face,
            bootstrap_current.j_bootstrap_face,
        ),
        I_bootstrap=jax_utils.select(
            is_zero_mode,
            zero_profile.I_bootstrap,
            bootstrap_current.I_bootstrap,
        ),
    )

  def get_source_profile_for_affected_core_profile(
      self,
      profile: chex.ArrayTree,
      affected_core_profile: int,
      geo: geometry.Geometry,
  ) -> jax.Array:
    return jnp.where(
        affected_core_profile in self.affected_core_profiles_ints,
        profile[self.SOURCE_NAME],
        jnp.zeros_like(geo.rho),
    )


@jax_utils.jit
def calc_neoclassical(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    temp_ion: cell_variable.CellVariable,
    temp_el: cell_variable.CellVariable,
    ne: cell_variable.CellVariable,
    ni: cell_variable.CellVariable,
    psi: cell_variable.CellVariable,
) -> source_profiles.BootstrapCurrentProfile:
  """Calculates sigmaneo, j_bootstrap, and I_bootstrap.

  Args:
    dynamic_runtime_params_slice: General configuration parameters.
    geo: Torus geometry.
    temp_ion: Ion temperature. We don't pass in a full `core_profiles` here
      because this function is used to create the `Currents` in the initial
      `State`.
    temp_el: Ion temperature.
    ne: Electron density.
    ni: Main ion density.
    psi: Poloidal flux.

  Returns:
    A BootstrapCurrentProfile. See that class's docstring for more info.
  """
  dynamic_source_runtime_params = dynamic_runtime_params_slice.sources[
      BootstrapCurrentSource.SOURCE_NAME
  ]
  assert isinstance(dynamic_source_runtime_params, DynamicRuntimeParams)
  # Many variables throughout this function are capitalized based on physics
  # notational conventions rather than on Google Python style
  # pylint: disable=invalid-name

  # Formulas from Sauter PoP 1999. Future work can include Redl PoP 2021
  # corrections.

  true_ne_face = ne.face_value() * dynamic_runtime_params_slice.numerics.nref
  true_ni_face = ni.face_value() * dynamic_runtime_params_slice.numerics.nref
  Zeff_face = dynamic_runtime_params_slice.plasma_composition.Zeff_face

  # # local r/R0 on face grid
  epsilon = (geo.Rout_face - geo.Rin_face) / (geo.Rout_face + geo.Rin_face)
  epseff = (
      0.67 * (1.0 - 1.4 * jnp.abs(geo.delta_face) * geo.delta_face) * epsilon
  )
  aa = (1.0 - epsilon) / (1.0 + epsilon)
  ftrap = 1.0 - jnp.sqrt(aa) * (1.0 - epseff) / (1.0 + 2.0 * jnp.sqrt(epseff))

  # Spitzer conductivity
  NZ = 0.58 + 0.74 / (0.76 + Zeff_face)
  lnLame = (
      31.3 - 0.5 * jnp.log(true_ne_face) + jnp.log(temp_el.face_value() * 1e3)
  )
  lnLami = (
      30
      - 3 * jnp.log(dynamic_runtime_params_slice.plasma_composition.Zi)
      - 0.5 * jnp.log(true_ni_face)
      + 1.5 * jnp.log(temp_ion.face_value() * 1e3)
  )

  sigsptz = (
      1.9012e04 * (temp_el.face_value() * 1e3) ** 1.5 / Zeff_face / NZ / lnLame
  )

  # We don't store q_cell in the evolving core profiles, so we need to
  # recalculate it.
  q_face, _ = physics.calc_q_from_psi(
      geo=geo,
      psi=psi,
      q_correction_factor=dynamic_runtime_params_slice.numerics.q_correction_factor,
  )
  nuestar = (
      6.921e-18
      * q_face
      * geo.Rmaj
      * true_ne_face
      * Zeff_face
      * lnLame
      / (
          ((temp_el.face_value() * 1e3) ** 2)
          * (epsilon + constants.CONSTANTS.eps) ** 1.5
      )
  )
  nuistar = (
      4.9e-18
      * q_face
      * geo.Rmaj
      * true_ni_face
      * Zeff_face**4
      * lnLami
      / (
          ((temp_ion.face_value() * 1e3) ** 2)
          * (epsilon + constants.CONSTANTS.eps) ** 1.5
      )
  )

  # Neoclassical correction to spitzer conductivity
  ft33 = ftrap / (
      1.0
      + (0.55 - 0.1 * ftrap) * jnp.sqrt(nuestar)
      + 0.45 * (1.0 - ftrap) * nuestar / (Zeff_face**1.5)
  )
  signeo_face = 1.0 - ft33 * (
      1.0
      + 0.36 / Zeff_face
      - ft33 * (0.59 / Zeff_face - 0.23 / Zeff_face * ft33)
  )
  sigmaneo = sigsptz * signeo_face

  # Calculate terms needed for bootstrap current
  denom = (
      1.0
      + (1 - 0.1 * ftrap) * jnp.sqrt(nuestar)
      + 0.5 * (1.0 - ftrap) * nuestar / Zeff_face
  )
  ft31 = ftrap / denom
  ft32ee = ftrap / (
      1
      + 0.26 * (1 - ftrap) * jnp.sqrt(nuestar)
      + 0.18 * (1 - 0.37 * ftrap) * nuestar / jnp.sqrt(Zeff_face)
  )
  ft32ei = ftrap / (
      1
      + (1 + 0.6 * ftrap) * jnp.sqrt(nuestar)
      + 0.85 * (1 - 0.37 * ftrap) * nuestar * (1 + Zeff_face)
  )
  ft34 = ftrap / (
      1.0
      + (1 - 0.1 * ftrap) * jnp.sqrt(nuestar)
      + 0.5 * (1.0 - 0.5 * ftrap) * nuestar / Zeff_face
  )

  F32ee = (
      (0.05 + 0.62 * Zeff_face)
      / (Zeff_face * (1 + 0.44 * Zeff_face))
      * (ft32ee - ft32ee**4)
      + 1
      / (1 + 0.22 * Zeff_face)
      * (ft32ee**2 - ft32ee**4 - 1.2 * (ft32ee**3 - ft32ee**4))
      + 1.2 / (1 + 0.5 * Zeff_face) * ft32ee**4
  )

  F32ei = (
      -(0.56 + 1.93 * Zeff_face)
      / (Zeff_face * (1 + 0.44 * Zeff_face))
      * (ft32ei - ft32ei**4)
      + 4.95
      / (1 + 2.48 * Zeff_face)
      * (ft32ei**2 - ft32ei**4 - 0.55 * (ft32ei**3 - ft32ei**4))
      - 1.2 / (1 + 0.5 * Zeff_face) * ft32ei**4
  )

  term_0 = (1 + 1.4 / (Zeff_face + 1)) * ft31
  term_1 = -1.9 / (Zeff_face + 1) * ft31**2
  term_2 = 0.3 / (Zeff_face + 1) * ft31**3
  term_3 = 0.2 / (Zeff_face + 1) * ft31**4
  L31 = term_0 + term_1 + term_2 + term_3

  L32 = F32ee + F32ei

  L34 = (
      (1 + 1.4 / (Zeff_face + 1)) * ft34
      - 1.9 / (Zeff_face + 1) * ft34**2
      + 0.3 / (Zeff_face + 1) * ft34**3
      + 0.2 / (Zeff_face + 1) * ft34**4
  )

  alpha0 = -1.17 * (1 - ftrap) / (1 - 0.22 * ftrap - 0.19 * ftrap**2)
  alpha = (
      alpha0
      + 0.25
      * (1 - ftrap**2)
      * jnp.sqrt(nuistar)
      / (1 + 0.5 * jnp.sqrt(nuistar))
      + 0.315 * nuistar**2 * ftrap**6
  ) / (1 + 0.15 * nuistar**2 * ftrap**6)

  # calculate bootstrap current
  prefactor = (
      -geo.F_face
      * dynamic_source_runtime_params.bootstrap_mult
      * 2
      * jnp.pi
      / geo.B0
  )

  pe = true_ne_face * (temp_el.face_value()) * 1e3 * 1.6e-19
  pi = true_ni_face * (temp_ion.face_value()) * 1e3 * 1.6e-19

  dpsi_drnorm = psi.face_grad()
  dlnne_drnorm = ne.face_grad() / ne.face_value()
  dlnni_drnorm = ni.face_grad() / ni.face_value()
  dlnte_drnorm = temp_el.face_grad() / temp_el.face_value()
  dlnti_drnorm = temp_ion.face_grad() / temp_ion.face_value()

  global_coeff = prefactor[1:] / dpsi_drnorm[1:]
  global_coeff = jnp.concatenate([jnp.zeros(1), global_coeff])

  necoeff = L31 * pe
  nicoeff = L31 * pi
  tecoeff = (L31 + L32) * pe
  ticoeff = (L31 + alpha * L34) * pi

  j_bootstrap_face = global_coeff * (
      necoeff * dlnne_drnorm
      + nicoeff * dlnni_drnorm
      + tecoeff * dlnte_drnorm
      + ticoeff * dlnti_drnorm
  )

  #  j_bootstrap_face = jnp.concatenate([jnp.zeros(1), j_bootstrap_face])
  j_bootstrap_face = jnp.array(j_bootstrap_face)
  j_bootstrap = geometry.face_to_cell(j_bootstrap_face)
  sigmaneo_cell = geometry.face_to_cell(sigmaneo)

  I_bootstrap = integrate.trapezoid(
      j_bootstrap_face * geo.spr_face,
      geo.rho_face_norm,
  )

  return source_profiles.BootstrapCurrentProfile(
      sigma=sigmaneo_cell,
      sigma_face=sigmaneo,
      j_bootstrap=j_bootstrap,
      j_bootstrap_face=j_bootstrap_face,
      I_bootstrap=I_bootstrap,
  )

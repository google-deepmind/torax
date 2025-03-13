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
from typing import ClassVar, Literal

import chex
import jax
from jax import numpy as jnp
from torax import constants
from torax import jax_utils
from torax import math_utils
from torax import state
from torax.config import runtime_params_slice
from torax.fvm import cell_variable
from torax.geometry import geometry
from torax.sources import base
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source
from torax.sources import source_profiles


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  bootstrap_mult: float


class BootstrapCurrentSourceConfig(base.SourceModelBase):
  """Bootstrap current density source profile.

  Attributes:
    bootstrap_mult: Multiplication factor for bootstrap current.
  """

  source_name: Literal['j_bootstrap'] = 'j_bootstrap'
  bootstrap_mult: float = 1.0
  mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED

  def model_func(self):
    raise NotImplementedError(
        'Bootstrap current source is not meant to be used as a model.'
    )

  def build_source(self) -> BootstrapCurrentSource:
    return BootstrapCurrentSource()

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(
        prescribed_values=self.prescribed_values.get_value(t),
        bootstrap_mult=self.bootstrap_mult,
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
  DEFAULT_MODEL_FUNCTION_NAME: ClassVar[str] = 'calc_neoclassical'

  @property
  def source_name(self) -> str:
    return self.SOURCE_NAME

  @property
  def affected_core_profiles(self) -> tuple[source.AffectedCoreProfile, ...]:
    return (source.AffectedCoreProfile.PSI,)

  def get_bootstrap(
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
    # Make sure the input params are the correct type.
    if not isinstance(dynamic_source_runtime_params, DynamicRuntimeParams):
      raise TypeError(
          'Expected DynamicRuntimeParams, got '
          f'{type(dynamic_source_runtime_params)}.'
      )
    if (
        static_source_runtime_params.mode
        == runtime_params_lib.Mode.PRESCRIBED.value
    ):
      raise NotImplementedError(
          'Prescribed mode not supported for bootstrap current.'
      )

    bootstrap_current = calc_sauter_model(
        bootstrap_multiplier=dynamic_source_runtime_params.bootstrap_mult,
        nref=dynamic_runtime_params_slice.numerics.nref,
        Zeff_face=dynamic_runtime_params_slice.plasma_composition.Zeff_face,
        Zi_face=core_profiles.Zi_face,
        ne=core_profiles.ne,
        ni=core_profiles.ni,
        temp_el=core_profiles.temp_el,
        temp_ion=core_profiles.temp_ion,
        psi=core_profiles.psi,
        q_face=core_profiles.q_face,
        geo=geo,
    )
    if static_source_runtime_params.mode == runtime_params_lib.Mode.ZERO.value:
      bootstrap_current = source_profiles.BootstrapCurrentProfile(
          sigma=bootstrap_current.sigma,
          sigma_face=bootstrap_current.sigma_face,
          j_bootstrap=jnp.zeros_like(bootstrap_current.j_bootstrap),
          j_bootstrap_face=jnp.zeros_like(bootstrap_current.j_bootstrap_face),
          I_bootstrap=jnp.zeros_like(bootstrap_current.I_bootstrap),
      )
    return bootstrap_current

  def get_value(
      self,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      calculated_source_profiles: source_profiles.SourceProfiles | None,
  ) -> tuple[chex.Array, ...]:
    raise NotImplementedError('Call `get_bootstrap` instead.')

  def get_source_profile_for_affected_core_profile(
      self,
      profile: chex.ArrayTree,
      affected_core_profile: int,
      geo: geometry.Geometry,
  ) -> jax.Array:
    raise NotImplementedError('Call `get_bootstrap` instead.')


@jax_utils.jit
def calc_sauter_model(
    *,
    bootstrap_multiplier: float,
    nref: float,
    Zeff_face: chex.Array,
    Zi_face: chex.Array,
    ne: cell_variable.CellVariable,
    ni: cell_variable.CellVariable,
    temp_el: cell_variable.CellVariable,
    temp_ion: cell_variable.CellVariable,
    psi: cell_variable.CellVariable,
    q_face: chex.Array,
    geo: geometry.Geometry,
) -> source_profiles.BootstrapCurrentProfile:
  """Calculates sigmaneo, j_bootstrap, and I_bootstrap."""
  # pylint: disable=invalid-name

  # Formulas from Sauter PoP 1999. Future work can include Redl PoP 2021
  # corrections.

  true_ne_face = ne.face_value() * nref
  true_ni_face = ni.face_value() * nref

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
      - 3 * jnp.log(Zi_face)
      - 0.5 * jnp.log(true_ni_face)
      + 1.5 * jnp.log(temp_ion.face_value() * 1e3)
  )

  sigsptz = (
      1.9012e04 * (temp_el.face_value() * 1e3) ** 1.5 / Zeff_face / NZ / lnLame
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
  prefactor = -geo.F_face * bootstrap_multiplier * 2 * jnp.pi / geo.B0

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

  I_bootstrap = math_utils.area_integration(j_bootstrap, geo)

  return source_profiles.BootstrapCurrentProfile(
      sigma=sigmaneo_cell,
      sigma_face=sigmaneo,
      j_bootstrap=j_bootstrap,
      j_bootstrap_face=j_bootstrap_face,
      I_bootstrap=I_bootstrap,
  )

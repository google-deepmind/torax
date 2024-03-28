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

import chex
from jax import numpy as jnp
from jax.scipy import integrate
from torax import config_slice
from torax import constants
from torax import geometry
from torax import jax_utils
from torax import physics
from torax import state
from torax.fvm import cell_variable
from torax.sources import source
from torax.sources import source_config
from torax.sources import source_profiles


@jax_utils.jit
def calc_neoclassical(
    dynamic_config_slice: config_slice.DynamicConfigSlice,
    geo: geometry.Geometry,
    temp_ion: cell_variable.CellVariable,
    temp_el: cell_variable.CellVariable,
    ne: cell_variable.CellVariable,
    ni: cell_variable.CellVariable,
    jtot_face: jnp.ndarray,
    psi: cell_variable.CellVariable,
) -> source_profiles.BootstrapCurrentProfile:
  """Calculates sigmaneo, j_bootstrap, and I_bootstrap.

  Args:
    dynamic_config_slice: General configuration parameters.
    geo: Torus geometry.
    temp_ion: Ion temperature. We don't pass in a full `State` here because this
      function is used to create the `Currents` in the initial `State`.
    temp_el: Ion temperature.
    ne: Electron density.
    ni: Main ion denisty.
    jtot_face: Total current density on face grid.
    psi: Poloidal flux.

  Returns:
    A BootstrapCurrentProfile. See that class's docstring for more info.
  """
  # Many variables throughout this function are capitalized based on physics
  # notational conventions rather than on Google Python style
  # pylint: disable=invalid-name

  # should be compared later to RAPTOR with Zeff=1
  # Formulas from Sauter PoP 1999. Future work can include Redl PoP 2021
  # corrections.

  true_ne_face = ne.face_value() * dynamic_config_slice.nref
  true_ni_face = ni.face_value() * dynamic_config_slice.nref

  # # local r/R0 on face grid
  epsilon = (geo.Rout_face - geo.Rin_face) / (geo.Rout_face + geo.Rin_face)
  epseff = (
      0.67 * (1.0 - 1.4 * jnp.abs(geo.delta_face) * geo.delta_face) * epsilon
  )
  aa = (1.0 - epsilon) / (1.0 + epsilon)
  ftrap = 1.0 - jnp.sqrt(aa) * (1.0 - epseff) / (1.0 + 2.0 * jnp.sqrt(epseff))

  # Spitzer conductivity
  NZ = 0.58 + 0.74 / (0.76 + dynamic_config_slice.Zeff)
  # TODO(b/323504363): should we expand the log to get rid of the exponentiation,
  # sqrt, etc? ne.value * dynamic_config_slice.nref is large, it might be
  # advantages to avoid calculating it explicitly.
  #
  # note for later: ne
  lnLame = 31.3 - jnp.log(jnp.sqrt(true_ne_face) / (temp_el.face_value() * 1e3))
  # note for later: ni
  lnLami = 30 - jnp.log(
      dynamic_config_slice.Zi**3
      * jnp.sqrt(true_ne_face)
      / ((temp_ion.face_value() * 1e3) ** 1.5)
  )

  sigsptz = (
      1.9012e04
      * (temp_el.face_value() * 1e3) ** 1.5
      / dynamic_config_slice.Zeff
      / NZ
      / lnLame
  )

  # We don't store q_cell in the evolving core profiles, so we need to
  # recalculate it.
  q_face, _ = physics.calc_q_from_jtot_psi(
      geo,
      jtot_face,
      psi,
      dynamic_config_slice.Rmaj,
      dynamic_config_slice.q_correction_factor,
  )
  nuestar = (
      6.921e-18
      * q_face
      * dynamic_config_slice.Rmaj
      * true_ne_face
      * dynamic_config_slice.Zeff
      * lnLame
      / (
          ((temp_el.face_value() * 1e3) ** 2)
          * (epsilon + constants.CONSTANTS.eps) ** 1.5
      )
  )
  nuistar = (
      4.9e-18
      * q_face
      * dynamic_config_slice.Rmaj
      * true_ni_face
      * dynamic_config_slice.Zeff**4
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
      + 0.45 * (1.0 - ftrap) * nuestar / (dynamic_config_slice.Zeff**1.5)
  )
  signeo_face = 1.0 - ft33 * (
      1.0
      + 0.36 / dynamic_config_slice.Zeff
      - ft33
      * (
          0.59 / dynamic_config_slice.Zeff
          - 0.23 / dynamic_config_slice.Zeff * ft33
      )
  )
  sigmaneo = sigsptz * signeo_face

  # Calculate terms needed for bootstrap current
  denom = (
      1.0
      + (1 - 0.1 * ftrap) * jnp.sqrt(nuestar)
      + 0.5 * (1.0 - ftrap) * nuestar / dynamic_config_slice.Zeff
  )
  ft31 = ftrap / denom
  ft32ee = ftrap / (
      1
      + 0.26 * (1 - ftrap) * jnp.sqrt(nuestar)
      + 0.18
      * (1 - 0.37 * ftrap)
      * nuestar
      / jnp.sqrt(dynamic_config_slice.Zeff)
  )
  ft32ei = ftrap / (
      1
      + (1 + 0.6 * ftrap) * jnp.sqrt(nuestar)
      + 0.85 * (1 - 0.37 * ftrap) * nuestar * (1 + dynamic_config_slice.Zeff)
  )
  ft34 = ftrap / (
      1.0
      + (1 - 0.1 * ftrap) * jnp.sqrt(nuestar)
      + 0.5 * (1.0 - 0.5 * ftrap) * nuestar / dynamic_config_slice.Zeff
  )

  F32ee = (
      (0.05 + 0.62 * dynamic_config_slice.Zeff)
      / (dynamic_config_slice.Zeff * (1 + 0.44 * dynamic_config_slice.Zeff))
      * (ft32ee - ft32ee**4)
      + 1
      / (1 + 0.22 * dynamic_config_slice.Zeff)
      * (ft32ee**2 - ft32ee**4 - 1.2 * (ft32ee**3 - ft32ee**4))
      + 1.2 / (1 + 0.5 * dynamic_config_slice.Zeff) * ft32ee**4
  )

  F32ei = (
      -(0.56 + 1.93 * dynamic_config_slice.Zeff)
      / (dynamic_config_slice.Zeff * (1 + 0.44 * dynamic_config_slice.Zeff))
      * (ft32ei - ft32ei**4)
      + 4.95
      / (1 + 2.48 * dynamic_config_slice.Zeff)
      * (ft32ei**2 - ft32ei**4 - 0.55 * (ft32ei**3 - ft32ei**4))
      - 1.2 / (1 + 0.5 * dynamic_config_slice.Zeff) * ft32ei**4
  )

  term_0 = (1 + 1.4 / (dynamic_config_slice.Zeff + 1)) * ft31
  term_1 = -1.9 / (dynamic_config_slice.Zeff + 1) * ft31**2
  term_2 = 0.3 / (dynamic_config_slice.Zeff + 1) * ft31**3
  term_3 = 0.2 / (dynamic_config_slice.Zeff + 1) * ft31**4
  L31 = term_0 + term_1 + term_2 + term_3

  L32 = F32ee + F32ei

  L34 = (
      (1 + 1.4 / (dynamic_config_slice.Zeff + 1)) * ft34
      - 1.9 / (dynamic_config_slice.Zeff + 1) * ft34**2
      + 0.3 / (dynamic_config_slice.Zeff + 1) * ft34**3
      + 0.2 / (dynamic_config_slice.Zeff + 1) * ft34**4
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
      -geo.F_face * dynamic_config_slice.bootstrap_mult * 2 * jnp.pi / geo.B0
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
  sigmaneo = geometry.face_to_cell(sigmaneo)

  I_bootstrap = integrate.trapezoid(
      j_bootstrap_face * geo.spr_face,
      geo.r_face,
  )

  return source_profiles.BootstrapCurrentProfile(
      sigma=sigmaneo,
      j_bootstrap=j_bootstrap,
      j_bootstrap_face=j_bootstrap_face,
      I_bootstrap=I_bootstrap,
  )


# TODO( b/314308399): Remove this as a source and create a
# Neoclassical class which will compute sigmaneo, j_bootstrap, etc.


def _default_output_shapes(
    unused_config, geo, unused_core_profiles
) -> tuple[int, int, int, int]:
  # TODO( b/314308399): When refactoring neoclassical "sources",
  # revisit what we actually need to return here.
  return (
      source.ProfileType.CELL.get_profile_shape(geo)  # sigmaneo
      + source.ProfileType.CELL.get_profile_shape(geo)  # bootstrap
      + source.ProfileType.FACE.get_profile_shape(geo)  # bootstrap face
      + (1,)  # I_bootstrap
  )


@dataclasses.dataclass(frozen=True, kw_only=True)
class BootstrapCurrentSource(source.Source):
  """Bootstrap current density source profile.

  Unlike other sources within TORAX, the BootstrapCurrentSource provides more
  than just the bootstrap current. It uses neoclassical physics to determine
  the following:
    - sigmaneo
    - bootstrap current (on cell and face grids)
    - integrated bootstrap current
  """

  name: str = 'j_bootstrap'
  output_shape_getter: source.SourceOutputShapeFunction = _default_output_shapes
  supported_types: tuple[source_config.SourceType, ...] = (
      source_config.SourceType.MODEL_BASED,
  )

  # Don't include affected_core_profiles in the __init__ arguments.
  # Freeze this param.
  affected_core_profiles: tuple[source.AffectedCoreProfile, ...] = (
      dataclasses.field(
          init=False,
          default_factory=lambda: (source.AffectedCoreProfile.PSI,),
      )
  )

  def get_value(
      self,
      dynamic_config_slice: config_slice.DynamicConfigSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles | None = None,
      temp_ion: cell_variable.CellVariable | None = None,
      temp_el: cell_variable.CellVariable | None = None,
      ne: cell_variable.CellVariable | None = None,
      ni: cell_variable.CellVariable | None = None,
      jtot_face: jnp.ndarray | None = None,
      psi: cell_variable.CellVariable | None = None,
  ) -> source_profiles.BootstrapCurrentProfile:
    if not core_profiles and any([
        not temp_ion,
        not temp_el,
        ne is None,
        ni is None,
        jtot_face is None,
        not psi,
    ]):
      raise ValueError(
          'If you cannot provide "core_profiles", then provide all of '
          'temp_ion, temp_el, ne, ni, jtot_face, and psi.'
      )
    # pytype: disable=attribute-error
    temp_ion = temp_ion or core_profiles.temp_ion
    temp_el = temp_el or core_profiles.temp_el
    ne = ne if ne is not None else core_profiles.ne
    ni = ni if ni is not None else core_profiles.ni
    jtot_face = (
        jtot_face if jtot_face is not None else core_profiles.currents.jtot_face
    )
    psi = psi or core_profiles.psi
    # pytype: enable=attribute-error
    return calc_neoclassical(
        dynamic_config_slice,
        geo,
        temp_ion=temp_ion,
        temp_el=temp_el,
        ne=ne,
        ni=ni,
        jtot_face=jtot_face,
        psi=psi,
    )

  def get_source_profile_for_affected_core_profile(
      self,
      profile: chex.ArrayTree,
      affected_core_profile: int,
      geo: geometry.Geometry,
  ) -> jnp.ndarray:
    return jnp.where(
        affected_core_profile in self.affected_core_profiles_ints,
        profile['j_bootstrap'],
        jnp.zeros_like(geo.r),
    )

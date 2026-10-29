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
"""Ohmic heat source."""

# pylint: disable=invalid-name

import dataclasses
from typing import Annotated, ClassVar, Literal
import jax.numpy as jnp
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.physics import psi_calculations
from torax._src.sources import base
from torax._src.sources import source as source_lib
from torax._src.sources import source_profiles
from torax._src.sources import source_state as source_state_lib
from torax._src.torax_pydantic import torax_pydantic


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class OhmicHeatSource(source_lib.Source):
  """Ohmic heat source for electron heat equation.

  P_ohm = sigma * <E . B>^2 / <B^2>, the flux-surface-averaged collisional
  Joule dissipation of the Ohmic (inductive) current. sigma and <E . B> are
  those of the flux-surface-averaged parallel Ohm's law
  <j . B> = sigma * <E . B> + <j_ni . B> solved in the psi equation; see e.g.
  Hinton & Hazeltine, Rev. Mod. Phys. 48, 239 (1976) and Pereverzev &
  Yushmanov, ASTRA Report IPP 5/98 (2002), Eq. (44).
  """

  SOURCE_ID: ClassVar[str] = 'ohmic'
  AFFECTED_CORE_PROFILES: ClassVar[
      tuple[source_lib.AffectedCoreProfile, ...]
  ] = (source_lib.AffectedCoreProfile.TEMP_EL,)

  def _get_model_value(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      calculated_source_profiles: source_profiles.SourceProfiles | None,
      conductivity: conductivity_base.Conductivity | None,
      source_state: source_state_lib.SourceState,
  ) -> tuple[source_profiles.SourceOutput, source_state_lib.SourceState]:
    """Returns the Ohmic source for electron heat equation.

    Does not modify source_state; returns it unchanged.

    Args:
      runtime_params: Runtime parameters.
      geo: Geometry of the torus.
      core_profiles: Core plasma profiles.
      calculated_source_profiles: Source profiles calculated so far.
      conductivity: Conductivity model output.
      source_state: Source state carried over from the previous timestep. Unused
        by this model and returned unchanged.

    Returns:
      A tuple of (SourceOutput(power_density_e=electron_heat_profile),
      source_state).
    """
    if calculated_source_profiles is None:
      raise ValueError(
          'calculated_source_profiles is a required argument for'
          ' OhmicHeatSource._get_model_value. This can occur if this source'
          ' is used in an explicit source.'
      )

    if conductivity is None:
      raise ValueError(
          'conductivity is a required argument for'
          ' OhmicHeatSource._get_model_value. This can occur if this source'
          ' is used in an explicit source.'
      )

    if (
        not runtime_params.numerics.evolve_current
        and runtime_params.profile_conditions.psidot is not None
    ):
      # If psidot is prescribed and current does not evolve, use
      # prescribed value
      psidot = runtime_params.profile_conditions.psidot
    else:
      psi_sources = calculated_source_profiles.total_psi_sources(geo)
      psidot = psi_calculations.calculate_psidot_from_psi_sources(
          psi_sources=psi_sources,
          sigma=conductivity.sigma,
          psi=core_profiles.psi,
          geo=geo,
      )

    # Joule dissipation of the Ohmic current, P_ohm = sigma <E.B>^2 / <B^2>,
    # with <E.B> from dpsi/dt at fixed toroidal flux (psidot is at fixed
    # rho_norm). See the "Ohmic power" section of docs/physics_models.rst.
    psidot_fixed_phi = (
        psidot
        - geo.rho_norm
        * geo.Phi_b_dot
        / (2.0 * geo.Phi_b)
        * core_profiles.psi.grad()
    )
    E_dot_B = geo.F * geo.g3 / (2.0 * jnp.pi) * psidot_fixed_phi
    P_ohm = conductivity.sigma * E_dot_B**2 / geo.gm5
    return source_profiles.SourceOutput(power_density_e=P_ohm), source_state


class OhmicHeatSourceConfig(base.SourceConfigBase):
  """Configuration for the OhmicHeatSource."""

  model_name: Annotated[Literal['standard'], torax_pydantic.JAX_STATIC] = (
      'standard'
  )

  def build_source(self) -> OhmicHeatSource:
    return OhmicHeatSource()

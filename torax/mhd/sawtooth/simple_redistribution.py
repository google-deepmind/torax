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

"""Simple redistribution model for sawteeth."""

import dataclasses
from typing import Literal
import chex
from jax import numpy as jnp
from torax import array_typing
from torax import state
from torax.config import runtime_params_slice
from torax.core_profiles import getters
from torax.geometry import geometry
from torax.mhd.sawtooth import base_pydantic_model
from torax.mhd.sawtooth import flatten_profile
from torax.mhd.sawtooth import runtime_params
from torax.mhd.sawtooth import sawtooth_model
from torax.physics import psi_calculations
from torax.torax_pydantic import torax_pydantic


# pylint: disable=invalid-name
class SimpleRedistribution(sawtooth_model.RedistributionModel):
  """Simple redistribution model."""

  def __call__(
      self,
      rho_norm_q1: array_typing.ScalarFloat,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> state.CoreProfiles:
    """Applies redistribution of profiles with a user-predefined mixing radius.

    Redistributes the profiles to a radius of
    mixing_radius_multiplier * rho_norm_q1. Two linear profiles are created for
    the density, temperature, and current profiles in the mixing zone:
    1. A flattened profile up to the q=1 surface.
    2. A linear profile up to the mixing radius.

    The value of the redistributed profile at q=1 is set by particle, energy
    and current conservation.

    Args:
      rho_norm_q1: The radius of the q=1 surface.
      static_runtime_params_slice: Static runtime parameters.
      dynamic_runtime_params_slice: Dynamic runtime parameters.
      geo: Geometry object.
      core_profiles: Core plasma profiles *before* redistribution.

    Returns:
      Core plasma profiles *after* redistribution.
    """

    # No sawtooth redistribution if current is not being evolved.
    if not static_runtime_params_slice.current_eq:
      return core_profiles

    assert dynamic_runtime_params_slice.mhd.sawtooth is not None
    assert isinstance(
        dynamic_runtime_params_slice.mhd.sawtooth.redistribution_params,
        DynamicRuntimeParams,
    )
    redistribution_params = (
        dynamic_runtime_params_slice.mhd.sawtooth.redistribution_params
    )

    mixing_radius = redistribution_params.mixing_radius_multiplier * rho_norm_q1

    idx_mixing = jnp.searchsorted(geo.rho_norm, mixing_radius, side='left')

    # Construct masks for different profile domains.
    # The redistribution mask is for all cells up to the mixing radius, since
    # those are the only locations where the modified values contribute to the
    # volume integrals.
    indices = jnp.arange(geo.rho_norm.shape[0])
    redistribution_mask = indices < idx_mixing

    if static_runtime_params_slice.dens_eq:
      ne_redistributed = flatten_profile.flatten_density_profile(
          rho_norm_q1,
          mixing_radius,
          redistribution_mask,
          redistribution_params.flattening_factor,
          core_profiles.ne,
          geo,
      )
    else:
      ne_redistributed = core_profiles.ne
    if static_runtime_params_slice.el_heat_eq:
      te_redistributed = flatten_profile.flatten_temperature_profile(
          rho_norm_q1,
          mixing_radius,
          redistribution_mask,
          redistribution_params.flattening_factor,
          core_profiles.temp_el,
          core_profiles.ne,
          ne_redistributed,
          geo,
      )
    else:
      te_redistributed = core_profiles.temp_el
    if (
        static_runtime_params_slice.dens_eq
        or static_runtime_params_slice.el_heat_eq
    ):
      ni_redistributed, nimp_redistributed, Zi, Zi_face, Zimp, Zimp_face = (
          getters.get_ion_density_and_charge_states(
              static_runtime_params_slice,
              dynamic_runtime_params_slice,
              geo,
              ne_redistributed,
              te_redistributed,
          )
      )
    else:
      ni_redistributed = core_profiles.ni
      nimp_redistributed = core_profiles.nimp
      Zi = core_profiles.Zi
      Zi_face = core_profiles.Zi_face
      Zimp = core_profiles.Zimp
      Zimp_face = core_profiles.Zimp_face

    if static_runtime_params_slice.ion_heat_eq:
      ti_redistributed = flatten_profile.flatten_temperature_profile(
          rho_norm_q1,
          mixing_radius,
          redistribution_mask,
          redistribution_params.flattening_factor,
          core_profiles.temp_ion,
          core_profiles.ni,
          ni_redistributed,
          geo,
      )
    else:
      ti_redistributed = core_profiles.temp_ion
    psi_redistributed = flatten_profile.flatten_current_profile(
        rho_norm_q1,
        mixing_radius,
        redistribution_mask,
        redistribution_params.flattening_factor,
        core_profiles.psi,
        core_profiles.currents.jtot,
        core_profiles.currents.Ip_profile_face[-1],
        geo,
    )

    return dataclasses.replace(
        core_profiles,
        temp_ion=ti_redistributed,
        temp_el=te_redistributed,
        psi=psi_redistributed,
        ne=ne_redistributed,
        ni=ni_redistributed,
        nimp=nimp_redistributed,
        Zi=Zi,
        Zi_face=Zi_face,
        Zimp=Zimp,
        Zimp_face=Zimp_face,
        q_face=psi_calculations.calc_q_face(geo, psi_redistributed),
        s_face=psi_calculations.calc_s_face(geo, psi_redistributed),
    )

  def __hash__(self) -> int:
    return hash(self.__class__.__name__)

  def __eq__(self, other: object) -> bool:
    return isinstance(other, SimpleRedistribution)


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params.RedistributionDynamicRuntimeParams):
  """Dynamic runtime params for simple redistribution model.

  Attributes:
    mixing_radius_multiplier: Profile modification will be limited to a radius
      of mixing_radius_multiplier * rho_norm_at_q1.
  """

  mixing_radius_multiplier: array_typing.ScalarFloat


class SimpleRedistributionConfig(base_pydantic_model.RedistributionConfig):
  """Pydantic model for simple redistribution configuration."""

  redistribution_model_type: Literal['simple'] = 'simple'
  mixing_radius_multiplier: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.1)
  )

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> DynamicRuntimeParams:
    base_kwargs = dataclasses.asdict(super().build_dynamic_params(t))
    return DynamicRuntimeParams(
        **base_kwargs,
        mixing_radius_multiplier=self.mixing_radius_multiplier.get_value(t)
    )

  def build_redistribution_model(
      self,
  ) -> SimpleRedistribution:
    return SimpleRedistribution()

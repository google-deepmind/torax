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
from typing import Annotated, Literal

import chex
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import state
from torax._src import static_dataclass
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import getters
from torax._src.geometry import geometry
from torax._src.mhd.sawtooth import flatten_profile
from torax._src.mhd.sawtooth import redistribution_base
from torax._src.mhd.sawtooth import runtime_params as sawtooth_runtime_params
from torax._src.physics import psi_calculations
from torax._src.torax_pydantic import torax_pydantic


# pylint: disable=invalid-name
@dataclasses.dataclass(frozen=True, eq=False)
class SimpleRedistribution(
    redistribution_base.RedistributionModel, static_dataclass.StaticDataclass
):
  """Simple redistribution model."""

  def __call__(
      self,
      rho_norm_q1: array_typing.FloatScalar,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles_t: state.CoreProfiles,
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
      runtime_params: Runtime parameters.
      geo: Geometry object.
      core_profiles_t: Core plasma profiles *before* redistribution.

    Returns:
      Core plasma profiles *after* redistribution.
    """

    # No sawtooth redistribution if current is not being evolved.
    if not runtime_params.numerics.evolve_current:
      return core_profiles_t

    assert runtime_params.mhd.sawtooth is not None
    assert isinstance(
        runtime_params.mhd.sawtooth.redistribution_params,
        RuntimeParams,
    )
    redistribution_params = runtime_params.mhd.sawtooth.redistribution_params

    mixing_radius = redistribution_params.mixing_radius_multiplier * rho_norm_q1

    idx_mixing = jnp.searchsorted(geo.rho_norm, mixing_radius, side='left')

    # Construct masks for different profile domains.
    # The redistribution mask is for all cells up to the mixing radius, since
    # those are the only locations where the modified values contribute to the
    # volume integrals.
    indices = jnp.arange(geo.rho_norm.shape[0])
    redistribution_mask = indices < idx_mixing

    if runtime_params.numerics.evolve_density:
      n_e_redistributed = flatten_profile.flatten_density_profile(
          rho_norm_q1,
          mixing_radius,
          redistribution_mask,
          redistribution_params.flattening_factor,
          core_profiles_t.n_e,
          geo,
      )
    else:
      n_e_redistributed = core_profiles_t.n_e
    if runtime_params.numerics.evolve_electron_heat:
      te_redistributed = flatten_profile.flatten_temperature_profile(
          rho_norm_q1,
          mixing_radius,
          redistribution_mask,
          redistribution_params.flattening_factor,
          core_profiles_t.T_e,
          core_profiles_t.n_e,
          n_e_redistributed,
          geo,
      )
    else:
      te_redistributed = core_profiles_t.T_e
    if (
        runtime_params.numerics.evolve_density
        or runtime_params.numerics.evolve_electron_heat
    ):
      ions_redistributed = getters.get_updated_ions(
          runtime_params,
          geo,
          n_e_redistributed,
          te_redistributed,
      )
    else:
      ions_redistributed = getters.Ions(
          n_i=core_profiles_t.n_i,
          n_impurity=core_profiles_t.n_impurity,
          impurity_fractions=core_profiles_t.impurity_fractions,
          main_ion_fractions=core_profiles_t.main_ion_fractions,
          Z_i=core_profiles_t.Z_i,
          Z_i_face=core_profiles_t.Z_i_face,
          Z_impurity=core_profiles_t.Z_impurity,
          Z_impurity_face=core_profiles_t.Z_impurity_face,
          A_i=core_profiles_t.A_i,
          A_impurity=core_profiles_t.A_impurity,
          A_impurity_face=core_profiles_t.A_impurity_face,
          Z_eff=core_profiles_t.Z_eff,
          Z_eff_face=core_profiles_t.Z_eff_face,
      )
    if runtime_params.numerics.evolve_ion_heat:
      ti_redistributed = flatten_profile.flatten_temperature_profile(
          rho_norm_q1,
          mixing_radius,
          redistribution_mask,
          redistribution_params.flattening_factor,
          core_profiles_t.T_i,
          core_profiles_t.n_i,
          ions_redistributed.n_i,
          geo,
      )
    else:
      ti_redistributed = core_profiles_t.T_i
    psi_redistributed = flatten_profile.flatten_current_profile(
        rho_norm_q1,
        mixing_radius,
        redistribution_mask,
        redistribution_params.flattening_factor,
        core_profiles_t.psi,
        core_profiles_t.j_total,
        core_profiles_t.Ip_profile_face[-1],
        geo,
    )

    return dataclasses.replace(
        core_profiles_t,
        T_i=ti_redistributed,
        T_e=te_redistributed,
        psi=psi_redistributed,
        n_e=n_e_redistributed,
        n_i=ions_redistributed.n_i,
        n_impurity=ions_redistributed.n_impurity,
        Z_i=ions_redistributed.Z_i,
        Z_i_face=ions_redistributed.Z_i_face,
        A_impurity=ions_redistributed.A_impurity,
        A_impurity_face=ions_redistributed.A_impurity_face,
        Z_impurity=ions_redistributed.Z_impurity,
        Z_impurity_face=ions_redistributed.Z_impurity_face,
        q_face=psi_calculations.calc_q_face(geo, psi_redistributed),
        s_face=psi_calculations.calc_s_face(geo, psi_redistributed),
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(sawtooth_runtime_params.RedistributionRuntimeParams):
  """Runtime params for simple redistribution model.

  Attributes:
    mixing_radius_multiplier: Profile modification will be limited to a radius
      of mixing_radius_multiplier * rho_norm_at_q1.
  """

  mixing_radius_multiplier: array_typing.FloatScalar


class SimpleRedistributionConfig(redistribution_base.RedistributionConfig):
  """Pydantic model for simple redistribution configuration."""

  model_name: Annotated[Literal['simple'], torax_pydantic.JAX_STATIC] = 'simple'
  mixing_radius_multiplier: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.1)
  )

  def build_runtime_params(
      self,
      t: chex.Numeric,
  ) -> RuntimeParams:
    base_kwargs = dataclasses.asdict(super().build_runtime_params(t))
    return RuntimeParams(
        **base_kwargs,
        mixing_radius_multiplier=self.mixing_radius_multiplier.get_value(t)
    )

  def build_redistribution_model(
      self,
  ) -> SimpleRedistribution:
    return SimpleRedistribution()

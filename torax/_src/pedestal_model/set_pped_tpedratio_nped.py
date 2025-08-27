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
"""Pedestal model that specifies pressure, temperature ratio, and density."""
import dataclasses

import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model
from torax._src.pedestal_model import runtime_params as runtime_params_lib
from torax._src.physics import formulas
from typing_extensions import override


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  """Dynamic runtime params for the BgB transport model."""

  P_ped: array_typing.FloatScalar
  n_e_ped: array_typing.FloatScalar
  T_i_T_e_ratio: array_typing.FloatScalar
  rho_norm_ped_top: array_typing.FloatScalar
  n_e_ped_is_fGW: array_typing.BoolScalar


class SetPressureTemperatureRatioAndDensityPedestalModel(
    pedestal_model.PedestalModel
):
  """Pedestal model with specification of pressure, temp ratio, and density."""

  def __init__(self):
    super().__init__()
    self._frozen = True

  @override
  def _call_implementation(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> pedestal_model.PedestalModelOutput:
    assert isinstance(
        dynamic_runtime_params_slice.pedestal, DynamicRuntimeParams
    )
    # Convert n_e_ped to reference units.
    # Ip in MA. a_minor in m. nGW in m^-3.
    nGW = (
        dynamic_runtime_params_slice.profile_conditions.Ip
        / 1e6  # Convert to MA.
        / (jnp.pi * geo.a_minor**2)
        * 1e20
    )
    n_e_ped = jnp.where(
        dynamic_runtime_params_slice.pedestal.n_e_ped_is_fGW,
        dynamic_runtime_params_slice.pedestal.n_e_ped * nGW,
        dynamic_runtime_params_slice.pedestal.n_e_ped,
    )

    # Calculate T_e_ped.
    temperature_ratio = dynamic_runtime_params_slice.pedestal.T_i_T_e_ratio
    Z_impurity = core_profiles.Z_impurity
    Z_i = core_profiles.Z_i
    # Find the value of Z_eff at the pedestal top.
    rho_norm_ped_top = dynamic_runtime_params_slice.pedestal.rho_norm_ped_top
    Z_eff = core_profiles.Z_eff

    ped_idx = jnp.abs(geo.rho_norm - rho_norm_ped_top).argmin()
    Z_eff_ped = jnp.take(Z_eff, ped_idx)
    Z_i_ped = jnp.take(Z_i, ped_idx)
    Z_impurity_ped = jnp.take(Z_impurity, ped_idx)
    dilution_factor_ped = formulas.calculate_main_ion_dilution_factor(
        Z_i_ped, Z_impurity_ped, Z_eff_ped
    )
    # Calculate n_i and n_impurity.
    n_i_ped = dilution_factor_ped * n_e_ped
    n_impurity_ped = (n_e_ped - Z_i_ped * n_i_ped) / Z_impurity_ped
    # Assumption that impurity is at the same temperature as the ion AND
    # the pressure P = P_e + P_i + P_imp.
    # P = T_e*n_e + T_i*n_i + T_i*n_imp.
    T_e_ped = (
        dynamic_runtime_params_slice.pedestal.P_ped
        / (
            n_e_ped  # Electron pressure contribution.
            + temperature_ratio * n_i_ped  # Ion pressure contribution.
            + temperature_ratio
            * n_impurity_ped  # Impurity pressure contribution.
        )
        / constants.CONSTANTS.keV2J
    )

    # Calculate T_i_ped
    T_i_ped = temperature_ratio * T_e_ped

    return pedestal_model.PedestalModelOutput(
        n_e_ped=n_e_ped,
        T_i_ped=T_i_ped,
        T_e_ped=T_e_ped,
        rho_norm_ped_top=dynamic_runtime_params_slice.pedestal.rho_norm_ped_top,
        rho_norm_ped_top_idx=ped_idx,
    )

  def __hash__(self) -> int:
    return hash('SetPressureTemperatureRatioAndDensityPedestalModel')

  def __eq__(self, other) -> bool:
    return isinstance(other, SetPressureTemperatureRatioAndDensityPedestalModel)

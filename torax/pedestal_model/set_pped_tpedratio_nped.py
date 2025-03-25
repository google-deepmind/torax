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
import chex
from jax import numpy as jnp
from torax import array_typing
from torax import constants
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.pedestal_model import pedestal_model
from torax.pedestal_model import runtime_params as runtime_params_lib
from torax.physics import formulas
from typing_extensions import override


# pylint: disable=invalid-name
@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  """Dynamic runtime params for the BgB transport model."""

  Pped: array_typing.ScalarFloat
  neped: array_typing.ScalarFloat
  ion_electron_temperature_ratio: array_typing.ScalarFloat
  rho_norm_ped_top: array_typing.ScalarFloat
  neped_is_fGW: array_typing.ScalarBool


class SetPressureTemperatureRatioAndDensityPedestalModel(
    pedestal_model.PedestalModel
):
  """Pedestal model with specification of pressure, temp ratio, and density."""

  def __init__(
      self,
  ):
    super().__init__()
    self._frozen = True

  @override
  def _call_implementation(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> pedestal_model.PedestalModelOutput:
    assert isinstance(
        dynamic_runtime_params_slice.pedestal, DynamicRuntimeParams
    )
    # Convert neped to reference units.
    nGW = (
        dynamic_runtime_params_slice.profile_conditions.Ip_tot
        / (jnp.pi * geo.Rmin**2)
        * 1e20
        / dynamic_runtime_params_slice.numerics.nref
    )
    neped_ref = jnp.where(
        dynamic_runtime_params_slice.pedestal.neped_is_fGW,
        dynamic_runtime_params_slice.pedestal.neped * nGW,
        dynamic_runtime_params_slice.pedestal.neped,
    )

    # Calculate Teped.
    temperature_ratio = (
        dynamic_runtime_params_slice.pedestal.ion_electron_temperature_ratio
    )
    Zimp = core_profiles.Zimp
    Zi = core_profiles.Zi
    # Find the value of Zeff at the pedestal top.
    rho_norm_ped_top = dynamic_runtime_params_slice.pedestal.rho_norm_ped_top
    Zeff = dynamic_runtime_params_slice.plasma_composition.Zeff

    ped_idx = jnp.abs(geo.rho_norm - rho_norm_ped_top).argmin()
    Zeff_ped = Zeff[ped_idx]
    Zi_ped = Zi[ped_idx]
    Zimp_ped = Zimp[ped_idx]
    dilution_factor_ped = formulas.calculate_main_ion_dilution_factor(
        Zi_ped, Zimp_ped, Zeff_ped
    )
    # Calculate ni and nimp.
    ni_ped = dilution_factor_ped * neped_ref
    nimp_ped = (neped_ref - Zi_ped * ni_ped) / Zimp_ped
    # Assumption that impurity is at the same temperature as the ion AND
    # the pressure P = P_e + P_i + P_imp.
    # P = T_e*n_e + T_i*n_i + T_i*n_imp.
    prefactor = constants.CONSTANTS.keV2J * core_profiles.nref
    Teped = (
        dynamic_runtime_params_slice.pedestal.Pped
        / (
            neped_ref  # Electron pressure contribution.
            + temperature_ratio * ni_ped  # Ion pressure contribution.
            + temperature_ratio * nimp_ped  # Impurity pressure contribution.
        )
        / prefactor
    )

    # Calculate Tiped
    Tiped = temperature_ratio * Teped

    return pedestal_model.PedestalModelOutput(
        neped=neped_ref,
        Tiped=Tiped,
        Teped=Teped,
        rho_norm_ped_top=dynamic_runtime_params_slice.pedestal.rho_norm_ped_top,
    )

  def __hash__(self) -> int:
    return hash('SetPressureTemperatureRatioAndDensityPedestalModel')

  def __eq__(self, other) -> bool:
    return isinstance(other, SetPressureTemperatureRatioAndDensityPedestalModel)

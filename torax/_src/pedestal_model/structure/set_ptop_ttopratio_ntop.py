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
from typing import override

import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import core_profiles as core_profiles_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_state as pedestal_state_lib
from torax._src.pedestal_model import (
    runtime_params as pedestal_runtime_params_lib,
)
from torax._src.pedestal_model.structure import base as structure_base
from torax._src.pedestal_model.structure import pedestal_structure
from torax._src.physics import formulas


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(pedestal_runtime_params_lib.StructureRuntimeParams):
  """Runtime params for SetPressureTemperatureRatioAndDensityPedestalModel."""

  p_top: array_typing.FloatScalar
  n_e_top: array_typing.FloatScalar
  T_i_T_e_ratio: array_typing.FloatScalar
  rho_norm_top: array_typing.FloatScalar
  n_e_top_is_f_GW: array_typing.BoolScalar
  p_top_multiplier: array_typing.FloatScalar = 1.0


@dataclasses.dataclass(frozen=True, eq=False)
class SetPressureTemperatureRatioAndDensityPedestalModel(
    structure_base.PedestalStructureModel
):
  """Pedestal structure model specifying pressure, temp ratio, and density."""

  @override
  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: core_profiles_lib.CoreProfiles,
      structure_state: pedestal_state_lib.StructureState,
  ) -> tuple[
      pedestal_structure.PedestalStructure,
      pedestal_state_lib.StructureState,
  ]:
    structure_params = runtime_params.pedestal.structure
    assert isinstance(structure_params, RuntimeParams)

    # Convert n_e_top to reference units.
    # I_p in MA. a_minor in m. nGW in m^-3.
    nGW = (
        runtime_params.profile_conditions.I_p
        / 1e6  # Convert to MA.
        / (jnp.pi * geo.a_minor**2)
        * 1e20
    )
    n_e_top = jnp.where(
        structure_params.n_e_top_is_f_GW,
        structure_params.n_e_top * nGW,
        structure_params.n_e_top,
    )

    # Calculate T_e_top.
    temperature_ratio = structure_params.T_i_T_e_ratio
    Z_impurity = core_profiles.ions.Z_impurity
    Z_i = core_profiles.ions.Z_i
    # Find the value of Z_eff at the pedestal top.
    rho_norm_top = structure_params.rho_norm_top
    Z_eff = core_profiles.ions.Z_eff

    top_idx = jnp.abs(geo.rho_norm - rho_norm_top).argmin()
    Z_eff_top = jnp.take(Z_eff, top_idx)
    Z_i_top = jnp.take(Z_i, top_idx)
    Z_impurity_top = jnp.take(Z_impurity, top_idx)
    dilution_factor_top = jnp.where(
        Z_eff_top == 1.0,
        1.0,
        formulas.calculate_main_ion_dilution_factor(
            Z_i_top, Z_impurity_top, Z_eff_top
        ),
    )
    # Calculate n_i and n_impurity.
    n_i_top = dilution_factor_top * n_e_top
    # Guard against division by zero when Z_impurity_top is zero or
    # degenerate (pure plasma case).
    safe_Z_impurity_top = jnp.where(Z_eff_top == 1.0, 1.0, Z_impurity_top)
    n_impurity_top = jnp.where(
        Z_eff_top == 1.0,
        0.0,
        (n_e_top - Z_i_top * n_i_top) / safe_Z_impurity_top,
    )
    # Assumption that impurity is at the same temperature as the ion AND
    # the pressure p = p_e + p_i + p_imp.
    # p = T_e*n_e + T_i*n_i + T_i*n_imp.
    T_e_top = (
        (
            structure_params.p_top
            * structure_params.p_top_multiplier
        )
        / (
            n_e_top  # Electron pressure contribution.
            + temperature_ratio * n_i_top  # Ion pressure contribution.
            + temperature_ratio
            * n_impurity_top  # Impurity pressure contribution.
        )
        / constants.CONSTANTS.keV_to_J
    )

    # Calculate T_i_top
    T_i_top = temperature_ratio * T_e_top

    return (
        pedestal_structure.PedestalStructure(
            n_e_top=n_e_top,
            T_i_top=T_i_top,
            T_e_top=T_e_top,
            rho_norm_top=structure_params.rho_norm_top,
        ),
        structure_state,
    )

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
"""A basic version of the pedestal model that uses direct specification."""

import dataclasses
from typing import override

import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_state as pedestal_state_lib
from torax._src.pedestal_model import (
    runtime_params as pedestal_runtime_params_lib,
)
from torax._src.pedestal_model.structure import base as structure_base
from torax._src.pedestal_model.structure import pedestal_structure


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(pedestal_runtime_params_lib.StructureRuntimeParams):
  """Runtime params for the SetTemperatureDensityPedestalModel."""

  n_e_ped: array_typing.FloatScalar
  T_i_ped: array_typing.FloatScalar
  T_e_ped: array_typing.FloatScalar
  rho_norm_ped_top: array_typing.FloatScalar
  n_e_ped_is_fGW: array_typing.BoolScalar


@dataclasses.dataclass(frozen=True, eq=False)
class SetTemperatureDensityPedestalModel(
    structure_base.PedestalStructureModel
):
  """Pedestal structure model using direct temperature and density targets."""

  @override
  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      structure_state: pedestal_state_lib.StructureState,
  ) -> tuple[
      pedestal_structure.PedestalStructure,
      pedestal_state_lib.StructureState,
  ]:
    del core_profiles
    structure_params = runtime_params.pedestal.structure
    assert isinstance(structure_params, RuntimeParams)

    # Calculate n_e_ped in m^-3, potentially converting from nGW.
    nGW = (
        runtime_params.profile_conditions.I_p
        / 1e6  # Convert to MA.
        / (jnp.pi * geo.a_minor**2)
        * 1e20
    )
    n_e_ped = jnp.where(
        structure_params.n_e_ped_is_fGW,
        structure_params.n_e_ped * nGW,
        structure_params.n_e_ped,
    )

    return (
        pedestal_structure.PedestalStructure(
            n_e_ped=n_e_ped,
            T_i_ped=structure_params.T_i_ped,
            T_e_ped=structure_params.T_e_ped,
            rho_norm_ped_top=structure_params.rho_norm_ped_top,
        ),
        structure_state,
    )

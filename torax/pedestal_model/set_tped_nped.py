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

from __future__ import annotations

import chex
from jax import numpy as jnp
from torax import array_typing
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.pedestal_model import pedestal_model
from torax.pedestal_model import runtime_params as runtime_params_lib
from typing_extensions import override


# pylint: disable=invalid-name
@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  """Dynamic runtime params for the BgB transport model."""

  neped: array_typing.ScalarFloat
  Tiped: array_typing.ScalarFloat
  Teped: array_typing.ScalarFloat
  rho_norm_ped_top: array_typing.ScalarFloat
  neped_is_fGW: array_typing.ScalarBool


class SetTemperatureDensityPedestalModel(pedestal_model.PedestalModel):
  """A basic version of the pedestal model that uses direct specification."""

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
    nGW = (
        dynamic_runtime_params_slice.profile_conditions.Ip_tot
        / (jnp.pi * geo.Rmin**2)
        * 1e20
        / dynamic_runtime_params_slice.numerics.nref
    )
    # Calculate neped in reference units.
    neped_ref = jnp.where(
        dynamic_runtime_params_slice.pedestal.neped_is_fGW,
        dynamic_runtime_params_slice.pedestal.neped * nGW,
        dynamic_runtime_params_slice.pedestal.neped,
    )
    return pedestal_model.PedestalModelOutput(
        neped=neped_ref,
        Tiped=dynamic_runtime_params_slice.pedestal.Tiped,
        Teped=dynamic_runtime_params_slice.pedestal.Teped,
        rho_norm_ped_top=dynamic_runtime_params_slice.pedestal.rho_norm_ped_top,
    )

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

import dataclasses
from typing import Callable

import chex
from jax import numpy as jnp
from torax import array_typing
from torax import geometry
from torax import interpolated_param
from torax import state
from torax.config import runtime_params_slice
from torax.pedestal_model import pedestal_model
from torax.pedestal_model import runtime_params as runtime_params_lib
from typing_extensions import override


# pylint: disable=invalid-name
@chex.dataclass
class RuntimeParams(runtime_params_lib.RuntimeParams):
  """Extends the base runtime params with additional params for this model.

  See base class runtime_params.RuntimeParams docstring for more info.
  """

  neped: interpolated_param.TimeInterpolatedInput = 0.7
  neped_is_fGW: bool = False
  Tiped: interpolated_param.TimeInterpolatedInput = 5.0
  Teped: interpolated_param.TimeInterpolatedInput = 5.0
  rho_norm_ped_top: interpolated_param.TimeInterpolatedInput = 0.91

  def make_provider(
      self, torax_mesh: geometry.Grid1D | None = None
  ) -> RuntimeParamsProvider:
    return RuntimeParamsProvider(**self.get_provider_kwargs(torax_mesh))


@chex.dataclass
class RuntimeParamsProvider(runtime_params_lib.RuntimeParamsProvider):
  """Provides a RuntimeParams to use during time t of the sim."""

  runtime_params_config: RuntimeParams
  neped: interpolated_param.InterpolatedVarSingleAxis
  Tiped: interpolated_param.InterpolatedVarSingleAxis
  Teped: interpolated_param.InterpolatedVarSingleAxis
  rho_norm_ped_top: interpolated_param.InterpolatedVarSingleAxis

  def build_dynamic_params(self, t: chex.Numeric) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(**self.get_dynamic_params_kwargs(t))


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  """Dynamic runtime params for the BgB transport model."""

  neped: array_typing.ScalarFloat
  Tiped: array_typing.ScalarFloat
  Teped: array_typing.ScalarFloat
  rho_norm_ped_top: array_typing.ScalarFloat
  neped_is_fGW: array_typing.ScalarBool


class BasicPedestalModel(pedestal_model.PedestalModel):
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


def _default_basic_pedestal_builder() -> BasicPedestalModel:
  return BasicPedestalModel()


@dataclasses.dataclass(kw_only=True)
class BasicPedestalModelBuilder(pedestal_model.PedestalModelBuilder):
  """Builds a class BasicPedestalModel."""

  runtime_params: RuntimeParams = dataclasses.field(
      default_factory=RuntimeParams
  )

  builder: Callable[
      [],
      BasicPedestalModel,
  ] = _default_basic_pedestal_builder

  def __call__(
      self,
  ) -> BasicPedestalModel:
    return self.builder()

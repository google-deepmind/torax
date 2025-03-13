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

"""The ConstantTransportModel class.

A simple model assuming constant transport.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import chex
from jax import numpy as jnp
from torax import array_typing
from torax import interpolated_param
from torax import jax_utils
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.pedestal_model import pedestal_model as pedestal_model_lib
from torax.torax_pydantic import torax_pydantic
from torax.transport_model import runtime_params as runtime_params_lib
from torax.transport_model import transport_model


# pylint: disable=invalid-name
@chex.dataclass
class RuntimeParams(runtime_params_lib.RuntimeParams):
  """Extends the base runtime params with additional params for this model.

  See base class runtime_params.RuntimeParams docstring for more info.
  """

  # coefficient in ion heat equation diffusion term in m^2/s
  chii_const: runtime_params_lib.TimeInterpolatedInput = 1.0
  # coefficient in electron heat equation diffusion term in m^2/s
  chie_const: runtime_params_lib.TimeInterpolatedInput = 1.0
  # diffusion coefficient in electron density equation in m^2/s
  De_const: runtime_params_lib.TimeInterpolatedInput = 1.0
  # convection coefficient in electron density equation in m^2/s
  Ve_const: runtime_params_lib.TimeInterpolatedInput = -0.33

  def make_provider(
      self, torax_mesh: torax_pydantic.Grid1D | None = None
  ) -> RuntimeParamsProvider:
    return RuntimeParamsProvider(**self.get_provider_kwargs(torax_mesh))


@chex.dataclass
class RuntimeParamsProvider(runtime_params_lib.RuntimeParamsProvider):
  """Provides a RuntimeParams to use during time t of the sim."""

  runtime_params_config: RuntimeParams
  chii_const: interpolated_param.InterpolatedVarSingleAxis
  chie_const: interpolated_param.InterpolatedVarSingleAxis
  De_const: interpolated_param.InterpolatedVarSingleAxis
  Ve_const: interpolated_param.InterpolatedVarSingleAxis

  def build_dynamic_params(self, t: chex.Numeric) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(**self.get_dynamic_params_kwargs(t))


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  """Extends the base runtime params with additional params for this model.

  See base class runtime_params.DynamicRuntimeParams docstring for more info.
  """

  chii_const: array_typing.ScalarFloat
  chie_const: array_typing.ScalarFloat
  De_const: array_typing.ScalarFloat
  Ve_const: array_typing.ScalarFloat

  def sanity_check(self):
    """Make sure all the parameters are valid."""
    runtime_params_lib.DynamicRuntimeParams.sanity_check(self)
    jax_utils.error_if_negative(self.De_const, 'De_const')
    jax_utils.error_if_negative(self.chii_const, 'chii_const')
    jax_utils.error_if_negative(self.chie_const, 'chie_const')


class ConstantTransportModel(transport_model.TransportModel):
  """Calculates various coefficients related to particle transport."""

  def __init__(self):
    super().__init__()
    self._frozen = True

  def _call_implementation(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_outputs: pedestal_model_lib.PedestalModelOutput,
  ) -> state.CoreTransport:
    del (
        core_profiles,
        pedestal_model_outputs,
    )  # Not needed for this transport model

    assert isinstance(
        dynamic_runtime_params_slice.transport, DynamicRuntimeParams
    )

    chi_face_ion = (
        dynamic_runtime_params_slice.transport.chii_const
        * jnp.ones_like(geo.rho_face)
    )
    chi_face_el = (
        dynamic_runtime_params_slice.transport.chie_const
        * jnp.ones_like(geo.rho_face)
    )
    d_face_el = dynamic_runtime_params_slice.transport.De_const * jnp.ones_like(
        geo.rho_face
    )
    v_face_el = dynamic_runtime_params_slice.transport.Ve_const * jnp.ones_like(
        geo.rho_face
    )

    return state.CoreTransport(
        chi_face_ion=chi_face_ion,
        chi_face_el=chi_face_el,
        d_face_el=d_face_el,
        v_face_el=v_face_el,
    )

  def __hash__(self):
    # All ConstantTransportModels are equivalent and can hash the same
    return hash('ConstantTransportModel')

  def __eq__(self, other):
    # All ConstantTransportModels are equivalent
    return isinstance(other, ConstantTransportModel)


def _default_constant_builder() -> ConstantTransportModel:
  return ConstantTransportModel()


@dataclasses.dataclass(kw_only=True)
class ConstantTransportModelBuilder(transport_model.TransportModelBuilder):
  """Builds a class ConstantTransportModel."""

  runtime_params: RuntimeParams = dataclasses.field(
      default_factory=RuntimeParams
  )

  builder: Callable[
      [],
      ConstantTransportModel,
  ] = _default_constant_builder

  def __call__(
      self,
  ) -> ConstantTransportModel:
    return self.builder()

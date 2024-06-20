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
from torax import geometry
from torax import jax_utils
from torax import state
from torax import versioning
from torax.config import config_args
from torax.config import runtime_params_slice
from torax.transport_model import runtime_params as runtime_params_lib
from torax.transport_model import transport_model


# pylint: disable=invalid-name
@chex.dataclass
class RuntimeParams(runtime_params_lib.RuntimeParams):
  """Extends the base runtime params with additional params for this model.

  See base class runtime_params.RuntimeParams docstring for more info.
  """

  # coefficient in ion heat equation diffusion term in m^2/s
  chii_const: runtime_params_lib.TimeInterpolatedScalar = 1.0
  # coefficient in electron heat equation diffusion term in m^2/s
  chie_const: runtime_params_lib.TimeInterpolatedScalar = 1.0
  # diffusion coefficient in electron density equation in m^2/s
  De_const: runtime_params_lib.TimeInterpolatedScalar = 1.0
  # convection coefficient in electron density equation in m^2/s
  Ve_const: runtime_params_lib.TimeInterpolatedScalar = -0.33

  def build_dynamic_params(self, t: chex.Numeric) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(
        **config_args.get_init_kwargs(
            input_config=self,
            output_type=DynamicRuntimeParams,
            t=t,
        )
    )


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  """Extends the base runtime params with additional params for this model.

  See base class runtime_params.DynamicRuntimeParams docstring for more info.
  """

  chii_const: float
  chie_const: float
  De_const: float
  Ve_const: float

  def sanity_check(self):
    """Make sure all the parameters are valid."""
    runtime_params_lib.DynamicRuntimeParams.sanity_check(self)
    # Using the object.__setattr__ call to get around the fact that this
    # dataclass is frozen.
    object.__setattr__(
        self, 'De_const', jax_utils.error_if_negative(self.De_const, 'De_const')
    )
    object.__setattr__(
        self,
        'chii_const',
        jax_utils.error_if_negative(self.chii_const, 'chii_const'),
    )
    object.__setattr__(
        self,
        'chie_const',
        jax_utils.error_if_negative(self.chie_const, 'chie_const'),
    )

  def build_dynamic_params(self, t: chex.Numeric) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(
        **config_args.get_init_kwargs(
            input_config=self,
            output_type=DynamicRuntimeParams,
            t=t,
        )
    )


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
  ) -> state.CoreTransport:
    del core_profiles  # Not needed for this transport model

    assert isinstance(
        dynamic_runtime_params_slice.transport, DynamicRuntimeParams
    )

    chi_face_ion = (
        dynamic_runtime_params_slice.transport.chii_const
        * jnp.ones_like(geo.r_face)
    )
    chi_face_el = (
        dynamic_runtime_params_slice.transport.chie_const
        * jnp.ones_like(geo.r_face)
    )
    d_face_el = dynamic_runtime_params_slice.transport.De_const * jnp.ones_like(
        geo.r_face
    )
    v_face_el = jnp.where(
        jnp.logical_and(
            dynamic_runtime_params_slice.profile_conditions.set_pedestal,
            geo.r_face_norm
            > dynamic_runtime_params_slice.profile_conditions.Ped_top,
        ),
        0,
        dynamic_runtime_params_slice.transport.Ve_const,
    )

    return state.CoreTransport(
        chi_face_ion=chi_face_ion,
        chi_face_el=chi_face_el,
        d_face_el=d_face_el,
        v_face_el=v_face_el,
    )

  def __hash__(self):
    # All ConstantTransportModels are equivalent and can hash the same
    return hash(('ConstrantTransportModel', versioning.torax_hash))

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

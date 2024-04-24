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

import chex
from jax import numpy as jnp
from torax import config_slice
from torax import geometry
from torax import jax_utils
from torax import state
from torax.runtime_params import config_slice_args
from torax.transport_model import runtime_params as runtime_params_lib
from torax.transport_model import transport_model


# pylint: disable=invalid-name
@chex.dataclass
class RuntimeParams(runtime_params_lib.RuntimeParams):
  """Extends the base runtime params with additional params for this model.

  See base class runtime_params.RuntimeParams docstring for more info.
  """

  # coefficient in ion heat equation diffusion term in m^2/s
  chii_const: runtime_params_lib.TimeDependentField = 1.0
  # coefficient in electron heat equation diffusion term in m^2/s
  chie_const: runtime_params_lib.TimeDependentField = 1.0
  # diffusion coefficient in electron density equation in m^2/s
  De_const: runtime_params_lib.TimeDependentField = 1.0
  # convection coefficient in electron density equation in m^2/s
  Ve_const: runtime_params_lib.TimeDependentField = -0.33

  def build_dynamic_params(self, t: chex.Numeric) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(
        **config_slice_args.get_init_kwargs(
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
        **config_slice_args.get_init_kwargs(
            input_config=self,
            output_type=DynamicRuntimeParams,
            t=t,
        )
    )


class ConstantTransportModel(transport_model.TransportModel):
  """Calculates various coefficients related to particle transport."""

  def __init__(
      self,
      runtime_params: RuntimeParams | None = None,
  ):
    self._runtime_params = runtime_params or RuntimeParams()

  @property
  def runtime_params(self) -> RuntimeParams:
    return self._runtime_params

  @runtime_params.setter
  def runtime_params(self, runtime_params: RuntimeParams) -> None:
    self._runtime_params = runtime_params

  def _call_implementation(
      self,
      dynamic_config_slice: config_slice.DynamicConfigSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> state.CoreTransport:
    del core_profiles  # Not needed for this transport model

    assert isinstance(dynamic_config_slice.transport, DynamicRuntimeParams)

    chi_face_ion = dynamic_config_slice.transport.chii_const * jnp.ones_like(
        geo.r_face
    )
    chi_face_el = dynamic_config_slice.transport.chie_const * jnp.ones_like(
        geo.r_face
    )
    d_face_el = dynamic_config_slice.transport.De_const * jnp.ones_like(
        geo.r_face
    )
    v_face_el = jnp.where(
        jnp.logical_and(
            dynamic_config_slice.profile_conditions.set_pedestal,
            geo.r_face_norm > dynamic_config_slice.profile_conditions.Ped_top,
        ),
        0,
        dynamic_config_slice.transport.Ve_const,
    )

    return state.CoreTransport(
        chi_face_ion=chi_face_ion,
        chi_face_el=chi_face_el,
        d_face_el=d_face_el,
        v_face_el=v_face_el,
    )

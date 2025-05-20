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
import chex
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.transport_model import runtime_params as runtime_params_lib
from torax._src.transport_model import transport_model


# pylint: disable=invalid-name
@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  """Extends the base runtime params with additional params for this model.

  See base class runtime_params.DynamicRuntimeParams docstring for more info.
  """

  chi_i: array_typing.ScalarFloat
  chi_e: array_typing.ScalarFloat
  D_e: array_typing.ScalarFloat
  V_e: array_typing.ScalarFloat


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
        dynamic_runtime_params_slice.transport.chi_i
        * jnp.ones_like(geo.rho_face)
    )
    chi_face_el = (
        dynamic_runtime_params_slice.transport.chi_e
        * jnp.ones_like(geo.rho_face)
    )
    d_face_el = dynamic_runtime_params_slice.transport.D_e * jnp.ones_like(
        geo.rho_face
    )
    v_face_el = dynamic_runtime_params_slice.transport.V_e * jnp.ones_like(
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

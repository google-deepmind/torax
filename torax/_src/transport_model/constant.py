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

A simple model assuming prescribed transport.

TODO(b/323504363): For the next major release (v2), the name of this model should be updated
to PrescribedTransportModel.
"""
import dataclasses
import jax
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.transport_model import runtime_params as runtime_params_lib
from torax._src.transport_model import transport_model as transport_model_lib


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  """Extends the base runtime params with additional params for this model.

  See base class runtime_params.DynamicRuntimeParams docstring for more info.
  """

  chi_i: array_typing.ArrayFloat
  chi_e: array_typing.ArrayFloat
  D_e: array_typing.ArrayFloat
  V_e: array_typing.ArrayFloat


class ConstantTransportModel(transport_model_lib.TransportModel):
  """Calculates various coefficients related to particle transport."""

  def __init__(self):
    super().__init__()
    self._frozen = True

  def _call_implementation(
      self,
      transport_dynamic_runtime_params: runtime_params_lib.DynamicRuntimeParams,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
  ) -> transport_model_lib.TurbulentTransport:
    r"""Calculates transport coefficients using the Constant model.

    Args:
      transport_dynamic_runtime_params: Input runtime parameters for this
        transport model. Can change without triggering a JAX recompilation.
      dynamic_runtime_params_slice: Input runtime parameters for all components
        of the simulation that can change without triggering a JAX
        recompilation.
      geo: Geometry of the torus.
      core_profiles: Core plasma profiles.
      pedestal_model_output: Output of the pedestal model.

    Returns:
      coeffs: The transport coefficients
    """
    del (
        core_profiles,
        pedestal_model_output,
    )  # Not needed for this transport model

    assert isinstance(transport_dynamic_runtime_params, DynamicRuntimeParams)

    return transport_model_lib.TurbulentTransport(
        chi_face_ion=transport_dynamic_runtime_params.chi_i,
        chi_face_el=transport_dynamic_runtime_params.chi_e,
        d_face_el=transport_dynamic_runtime_params.D_e,
        v_face_el=transport_dynamic_runtime_params.V_e,
    )

  def __hash__(self):
    # All ConstantTransportModels are equivalent and can hash the same
    return hash('ConstantTransportModel')

  def __eq__(self, other):
    # All ConstantTransportModels are equivalent
    return isinstance(other, ConstantTransportModel)

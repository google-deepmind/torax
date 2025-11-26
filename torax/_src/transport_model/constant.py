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
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.transport_model import runtime_params as transport_runtime_params_lib
from torax._src.transport_model import transport_model as transport_model_lib


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(transport_runtime_params_lib.RuntimeParams):
  """Extends the base runtime params with additional params for this model.

  See base class runtime_params_lib.RuntimeParams docstring for more info.
  """

  chi_i: array_typing.FloatVector
  chi_e: array_typing.FloatVector
  D_e: array_typing.FloatVector
  V_e: array_typing.FloatVector


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class ConstantTransportModel(transport_model_lib.TransportModel):
  """Calculates various coefficients related to particle transport."""

  def _call_implementation(
      self,
      transport_runtime_params: transport_runtime_params_lib.RuntimeParams,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
  ) -> transport_model_lib.TurbulentTransport:
    r"""Calculates transport coefficients using the Constant model.

    Args:
      transport_runtime_params: Input runtime parameters for this
        transport model.
      runtime_params: Input runtime parameters at the current time.
      geo: Geometry of the torus.
      core_profiles: Core plasma profiles.
      pedestal_model_output: Output of the pedestal model.

    Returns:
      coeffs: The transport coefficients
    """
    assert isinstance(transport_runtime_params, RuntimeParams)

    return transport_model_lib.TurbulentTransport(
        chi_face_ion=transport_runtime_params.chi_i,
        chi_face_el=transport_runtime_params.chi_e,
        d_face_el=transport_runtime_params.D_e,
        v_face_el=transport_runtime_params.V_e,
    )

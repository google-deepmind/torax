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

"""The PrescribedTransportModel class.

A simple model assuming prescribed transport.
"""

import dataclasses

import jax
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.transport_model import component
from torax._src.transport_model import runtime_params as transport_runtime_params_lib
from torax._src.transport_model import transport_coeffs


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(transport_runtime_params_lib.ComponentRuntimeParams):
  """Extends the base runtime params with additional params for this model.

  See base class runtime_params_lib.RuntimeParams docstring for more info.
  """

  chi_i: array_typing.FloatVector
  chi_e: array_typing.FloatVector
  D_e: array_typing.FloatVector
  V_e: array_typing.FloatVector


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class PrescribedTransportModel(component.ComponentTransportModel):
  """Calculates various coefficients related to particle transport."""

  def call_implementation(
      self,
      transport_runtime_params: (
          transport_runtime_params_lib.ComponentRuntimeParams
      ),
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      two_point_mask: array_typing.BoolVectorFace,
  ) -> transport_coeffs.TransportCoeffs:
    r"""Calculates transport coefficients using the Prescribed model.

    Args:
      transport_runtime_params: Input runtime parameters for this
        transport model.
      runtime_params: Input runtime parameters at the current time.
      geo: Geometry of the torus.
      core_profiles: Core plasma profiles.
      two_point_mask: Boolean mask on the face grid indicating where to use
        2-point central differencing instead of 3-point polynomial interpolation
        for gradients.

    Returns:
      coeffs: The transport coefficients
    """
    assert isinstance(transport_runtime_params, RuntimeParams)
    del two_point_mask

    return transport_coeffs.TransportCoeffs(
        chi_face_ion=transport_runtime_params.chi_i,
        chi_face_el=transport_runtime_params.chi_e,
        d_face_el=transport_runtime_params.D_e,
        v_face_el=transport_runtime_params.V_e,
    )

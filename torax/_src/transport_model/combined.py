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

"""The CombinedTransportModel class.

A class for combining transport models.
"""

from typing import Sequence

import chex
import jax
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.transport_model import runtime_params as runtime_params_lib
from torax._src.transport_model import transport_model as transport_model_lib

# pylint: disable=protected-access


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  transport_model_params: Sequence[runtime_params_lib.DynamicRuntimeParams]


class CombinedTransportModel(transport_model_lib.TransportModel):
  """Combines coefficients from a list of transport models."""

  def __init__(
      self, transport_models: Sequence[transport_model_lib.TransportModel]
  ):
    super().__init__()
    self.transport_models = transport_models
    self._frozen = True

  def _call_implementation(
      self,
      transport_dynamic_runtime_params: runtime_params_lib.DynamicRuntimeParams,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
  ) -> transport_model_lib.TurbulentTransport:
    r"""Calculates transport coefficients using the Combined model.

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
    # Required for pytype
    assert isinstance(transport_dynamic_runtime_params, DynamicRuntimeParams)

    component_transport_coeffs_list = []

    for component_model, component_params in zip(
        self.transport_models,
        transport_dynamic_runtime_params.transport_model_params,
    ):
      # Use the component model's _call_implementation, rather than __call__
      # directly. This ensures postprocessing (clipping, smoothing, patches) are
      # performed on the output of CombinedTransportModel rather than its
      # component models.
      component_transport_coeffs = component_model._call_implementation(
          component_params,
          dynamic_runtime_params_slice,
          geo,
          core_profiles,
          pedestal_model_output,
      )

      # Apply domain restriction
      # This is a property of each component_model, so needs to be applied
      # at the component model level rather than the global level
      component_transport_coeffs = component_model._apply_domain_restriction(
          component_params,
          geo,
          component_transport_coeffs,
          pedestal_model_output,
      )

      component_transport_coeffs_list.append(component_transport_coeffs)

    combined_transport_coeffs = jax.tree.map(
        lambda *leaves: sum(leaves),
        *component_transport_coeffs_list,
    )

    return combined_transport_coeffs

  def __hash__(self):
    return hash(tuple(self.transport_models))

  def __eq__(self, other):
    return (
        isinstance(other, CombinedTransportModel)
        and self.transport_models == other.transport_models
    )

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

import dataclasses
from typing import Sequence

import jax
import chex
from jax import numpy as jnp

from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.transport_model import runtime_params as runtime_params_lib
from torax._src.transport_model import transport_model


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  transport_model_params: Sequence[runtime_params_lib.DynamicRuntimeParams]


class CombinedTransportModel(transport_model.TransportModel):
  """Combines coefficients from a list of transport models."""

  def __init__(
      self, transport_models: Sequence[transport_model.TransportModel]
  ):
    super().__init__()
    self.transport_models = transport_models
    self._frozen = True

  def _call_implementation(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_outputs: pedestal_model_lib.PedestalModelOutput,
  ) -> state.CoreTransport:
    assert isinstance(
        dynamic_runtime_params_slice.transport, DynamicRuntimeParams
    )

    summed_transport_coeffs = state.CoreTransport.zeros(geo)

    for component_model, component_params in zip(
        self.transport_models,
        dynamic_runtime_params_slice.transport.transport_model_params,
    ):
      # Each transport model expects dynamic_runtime_params_slice.transport to
      # be its corresponding DynamicRuntimeParams
      # TODO: This seems like a hacky way of getting round what is potentially
      # a problem with the design [TB 22 May 2025]
      component_dynamic_runtime_params_slice = dataclasses.replace(
          dynamic_runtime_params_slice,
          transport=component_params,
      )

      # Use the component model's _call_implementation, rather than __call__
      # directly. This ensures postprocessing (clipping, smoothing, patches) are
      # performed on the output of CombinedTransportModel rather than its
      # component models.
      transport_component = component_model._call_implementation(
          component_dynamic_runtime_params_slice,
          geo,
          core_profiles,
          pedestal_model_outputs,
      )

      # However, we still need to restrict each model's domain.
      transport_component = component_model._apply_domain_restriction(
          component_dynamic_runtime_params_slice,
          geo,
          transport_component,
          pedestal_model_outputs,
      )

      summed_transport_coeffs = jax.tree.map(
          lambda old, new: old + new,
          summed_transport_coeffs,
          transport_component,
      )

    return summed_transport_coeffs

  def __hash__(self):
    return hash(tuple(self.transport_models))

  def __eq__(self, other):
    return (
        isinstance(other, CombinedTransportModel)
        and self.transport_models == other.transport_models
    )

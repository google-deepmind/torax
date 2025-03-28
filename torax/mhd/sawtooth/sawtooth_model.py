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

"""Sawtooth model."""

import abc
import jax
from torax import array_typing
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry


class TriggerModel(abc.ABC):
  """Abstract base class for sawtooth trigger models."""

  @abc.abstractmethod
  def __call__(
      self,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> array_typing.ScalarBool:
    """Indicates if a crash is triggered."""


class RedistributionModel(abc.ABC):
  """Abstract base class for sawtooth redistribution models."""

  @abc.abstractmethod
  def __call__(
      self,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> state.CoreProfiles:
    """Returns a redistributed core_profiles if sawtooth has been triggered."""


class SawtoothModel:
  """Sawtooth trigger and redistribution, and carries out sawtooth step."""

  def __init__(
      self,
      trigger_model: TriggerModel,
      redistribution_model: RedistributionModel,
  ):
    self.trigger_model = trigger_model
    self.redistribution_model = redistribution_model

  def __call__(
      self,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      input_state: state.ToraxSimState,
  ) -> state.ToraxSimState:
    """Applies the sawtooth model and outputs a new state if triggered.

    If the trigger model indicates a crash has been triggered, the
    redistribution model is applied. A new state following a short
    (configurable) dt is returned, with core_profiles modified by the
    redistribution model.

    Args:
      static_runtime_params_slice: Static runtime parameters.
      dynamic_runtime_params_slice: Dynamic runtime parameters.
      input_state: The input ToraxSimState.

    Returns:
      The output ToraxSimState, which may be modified by the sawtooth model.
    """

    trigger_sawtooth = self.trigger_model(
        static_runtime_params_slice,
        dynamic_runtime_params_slice,
        input_state.geometry,
        input_state.core_profiles,
    )

    jax.lax.cond(
        trigger_sawtooth,
        self.redistribution_model(
            static_runtime_params_slice,
            dynamic_runtime_params_slice,
            input_state.geometry,
            input_state.core_profiles,
        ),
        lambda: input_state.core_profiles,
    )

    # TODO(b/317360481)
    # modify output state with new time, dt, and core_profiles if triggered.

    return input_state

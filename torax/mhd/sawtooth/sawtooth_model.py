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
import dataclasses
import functools
import jax
from torax import array_typing
from torax import jax_utils
from torax import post_processing
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
  ) -> tuple[array_typing.ScalarBool, array_typing.ScalarFloat]:
    """Indicates if a crash is triggered and the radius of the q=1 surface."""

  @abc.abstractmethod
  def __hash__(self) -> int:
    """Returns a hash of the trigger model.

    Should be implemented to support jax.jit caching.
    """

  @abc.abstractmethod
  def __eq__(self, other) -> bool:
    """Returns whether the trigger model is equal to the other.

    Should be implemented to support jax.jit caching.

    Args:
      other: The object to compare to.
    """


class RedistributionModel(abc.ABC):
  """Abstract base class for sawtooth redistribution models."""

  @abc.abstractmethod
  def __call__(
      self,
      rho_norm_q1: array_typing.ScalarFloat,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> state.CoreProfiles:
    """Returns a redistributed core_profiles if sawtooth has been triggered."""

  @abc.abstractmethod
  def __hash__(self) -> int:
    """Returns a hash of the redistribution model.

    Should be implemented to support jax.jit caching.
    """

  @abc.abstractmethod
  def __eq__(self, other) -> bool:
    """Returns whether the redistribution model is equal to the other.

    Should be implemented to support jax.jit caching.

    Args:
      other: The object to compare to.
    """


class SawtoothModel:
  """Sawtooth trigger and redistribution, and carries out sawtooth step."""

  def __init__(
      self,
      trigger_model: TriggerModel,
      redistribution_model: RedistributionModel,
  ):
    self.trigger_model = trigger_model
    self.redistribution_model = redistribution_model

  @functools.partial(
      jax_utils.jit, static_argnames=['self', 'static_runtime_params_slice']
  )
  def __call__(
      self,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      input_state: state.ToraxSimState,
  ) -> tuple[bool, state.ToraxSimState]:
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

    trigger_sawtooth, rho_norm_q1 = self.trigger_model(
        static_runtime_params_slice,
        dynamic_runtime_params_slice,
        input_state.geometry,
        input_state.core_profiles,
    )

    def _true_fun_updated_core_profiles():
      return self.redistribution_model(
          rho_norm_q1,
          static_runtime_params_slice,
          dynamic_runtime_params_slice,
          input_state.geometry,
          input_state.core_profiles,
      )

    new_core_profiles = jax.lax.cond(
        trigger_sawtooth,
        _true_fun_updated_core_profiles,
        lambda: input_state.core_profiles,
    )

    # TODO(b/317360481)
    # Consider being more consistent with updating sources, bootstrap_current,
    # currents, etc at the end of this short step.

    def _true_fun_updated_output_state():
      return _update_output_state(
          input_state,
          dynamic_runtime_params_slice,
          new_core_profiles,
      )

    output_state = jax.lax.cond(
        trigger_sawtooth,
        _true_fun_updated_output_state,
        lambda: input_state,
    )

    return trigger_sawtooth, output_state

  def __hash__(self) -> int:
    return hash((self.trigger_model, self.redistribution_model))

  def __eq__(self, other) -> bool:
    return (
        isinstance(other, SawtoothModel)
        and self.trigger_model == other.trigger_model
        and self.redistribution_model == other.redistribution_model
    )


def _update_output_state(
    input_state: state.ToraxSimState,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    new_core_profiles: state.CoreProfiles,
) -> state.ToraxSimState:
  """Updates the output state with new core profiles and t, dt."""

  assert dynamic_runtime_params_slice.mhd.sawtooth is not None

  output_state = dataclasses.replace(
      input_state,
      core_profiles=new_core_profiles,
      t=input_state.t
      + dynamic_runtime_params_slice.mhd.sawtooth.crash_step_duration,
      dt=dynamic_runtime_params_slice.mhd.sawtooth.crash_step_duration,
  )

  output_state = post_processing.make_outputs(
      sim_state=output_state,
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      previous_sim_state=input_state,
  )
  return output_state


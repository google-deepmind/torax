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

import dataclasses
import functools
import jax
from torax import jax_utils
from torax import post_processing
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.mhd.sawtooth import redistribution_base
from torax.mhd.sawtooth import trigger_base


# TODO(b/414537757). Sawtooth extensions.
# a. Full and incomplete Kadomtsev redistribution model.
# b. Porcelli model with free parameters and fast ion sensitivities.
# c. "Smooth" version that can work with forward-sensitivity-analysis and
#    stationary-state applications without the need for averaging.
class SawtoothModel:
  """Sawtooth trigger and redistribution, and carries out sawtooth step."""

  def __init__(
      self,
      trigger_model: trigger_base.TriggerModel,
      redistribution_model: redistribution_base.RedistributionModel,
  ):
    self._trigger_model = trigger_model
    self._redistribution_model = redistribution_model

  @functools.partial(
      jax_utils.jit,
      static_argnames=['self', 'static_runtime_params_slice'],
  )
  def __call__(
      self,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice_t: runtime_params_slice.DynamicRuntimeParamsSlice,
      input_state: state.ToraxSimState,
      input_post_processed_outputs: state.PostProcessedOutputs,
      dynamic_runtime_params_slice_t_plus_crash_dt: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo_t_plus_crash_dt: geometry.Geometry,
  ) -> tuple[state.ToraxSimState, state.PostProcessedOutputs]:
    """Applies the sawtooth model and outputs a new state if triggered.

    If the trigger model indicates a crash has been triggered, the
    redistribution model is applied. A new state following a short
    (configurable) dt is returned, with core_profiles modified by the
    redistribution model.

    Args:
      static_runtime_params_slice: Static runtime parameters.
      dynamic_runtime_params_slice_t: Dynamic runtime parameters at time t.
      input_state: The input ToraxSimState.
      input_post_processed_outputs: The input PostProcessedOutputs.
      dynamic_runtime_params_slice_t_plus_crash_dt: Dynamic runtime parameters
        at time t + crash_step_duration.
      geo_t_plus_crash_dt: The geometry at time t + crash_step_duration.

    Returns:
      The output ToraxSimState, which may be modified by the sawtooth model.
      The output PostProcessedOutputs, which may be modified by the sawtooth
        model.
    """

    trigger_sawtooth, rho_norm_q1 = self._trigger_model(
        static_runtime_params_slice,
        dynamic_runtime_params_slice_t,
        input_state.geometry,
        input_state.core_profiles,
    )

    # TODO(b/317360481)
    # Consider being more consistent with updating sources, bootstrap_current,
    # currents, etc at the end of this short step.

    def _make_redistributed_state_and_post_processed_outputs() -> (
        tuple[state.ToraxSimState, state.PostProcessedOutputs]
    ):
      # assertion needed for linter
      assert dynamic_runtime_params_slice_t.mhd.sawtooth is not None
      redistributed_core_profiles = self._redistribution_model(
          rho_norm_q1,
          static_runtime_params_slice,
          dynamic_runtime_params_slice_t_plus_crash_dt,
          geo_t_plus_crash_dt,
          input_state.core_profiles,
      )

      output_state = dataclasses.replace(
          input_state,
          core_profiles=redistributed_core_profiles,
          t=input_state.t
          + dynamic_runtime_params_slice_t.mhd.sawtooth.crash_step_duration,
          dt=dynamic_runtime_params_slice_t.mhd.sawtooth.crash_step_duration,
          geometry=geo_t_plus_crash_dt,
          sawtooth_crash=True,
      )
      output_post_processed_outputs = (
          post_processing.make_post_processed_outputs(
              sim_state=output_state,
              dynamic_runtime_params_slice=dynamic_runtime_params_slice_t,
              previous_post_processed_outputs=input_post_processed_outputs,
          )
      )
      return output_state, output_post_processed_outputs

    output_state, output_post_processed_outputs = jax.lax.cond(
        trigger_sawtooth,
        _make_redistributed_state_and_post_processed_outputs,
        lambda: (
            dataclasses.replace(input_state, sawtooth_crash=False),
            input_post_processed_outputs,
        ),
    )

    return output_state, output_post_processed_outputs

  def __hash__(self) -> int:
    return hash((self._trigger_model, self._redistribution_model))

  def __eq__(self, other: object) -> bool:
    return (
        isinstance(other, SawtoothModel)
        and self._trigger_model == other._trigger_model
        and self._redistribution_model == other._redistribution_model
    )

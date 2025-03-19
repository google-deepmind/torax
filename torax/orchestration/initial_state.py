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
"""Code for getting the initial state for a simulation."""
import dataclasses

from absl import logging

from torax import output
from torax import sim
from torax import state
from torax.config import config_args
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.orchestration import step_function
from torax.torax_pydantic import file_restart as file_restart_pydantic_model


def initial_state_from_file_restart(
    file_restart: file_restart_pydantic_model.FileRestart,
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice_for_init: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo_for_init: geometry.Geometry,
    step_fn: step_function.SimulationStepFn,
) -> state.ToraxSimState:
  """Returns the initial state for a file restart."""
  data_tree = output.load_state_file(file_restart.filename)
  # Find the closest time in the given dataset.
  data_tree = data_tree.sel(time=file_restart.time, method='nearest')
  t_restart = data_tree.time.item()
  core_profiles_dataset = data_tree.children[output.CORE_PROFILES].dataset
  # Remap coordinates in saved file to be consistent with expectations of
  # how config_args parses xarrays.
  core_profiles_dataset = core_profiles_dataset.rename(
      {output.RHO_CELL_NORM: config_args.RHO_NORM}
  )
  core_profiles_dataset = core_profiles_dataset.squeeze()
  if t_restart != dynamic_runtime_params_slice_for_init.numerics.t_initial:
    logging.warning(
        'Requested restart time %f not exactly available in state file %s.'
        ' Restarting from closest available time %f instead.',
        file_restart.time,
        file_restart.filename,
        t_restart,
    )
  # Override some of dynamic runtime params slice from t=t_initial.
  dynamic_runtime_params_slice_for_init, geo_for_init = (
      sim._override_initial_runtime_params_from_file(  # pylint: disable=protected-access
          dynamic_runtime_params_slice_for_init,
          geo_for_init,
          t_restart,
          core_profiles_dataset,
      )
  )
  post_processed_dataset = data_tree.children[
      output.POST_PROCESSED_OUTPUTS
  ].dataset
  post_processed_dataset = post_processed_dataset.rename(
      {output.RHO_CELL_NORM: config_args.RHO_NORM}
  )
  post_processed_dataset = post_processed_dataset.squeeze()
  post_processed_outputs = (
      sim._override_initial_state_post_processed_outputs_from_file(  # pylint: disable=protected-access
          geo_for_init,
          post_processed_dataset,
      )
  )

  initial_state = sim.get_initial_state(
      static_runtime_params_slice=static_runtime_params_slice,
      dynamic_runtime_params_slice=dynamic_runtime_params_slice_for_init,
      geo=geo_for_init,
      step_fn=step_fn,
  )
  return dataclasses.replace(
      initial_state,
      post_processed_outputs=post_processed_outputs,
  )

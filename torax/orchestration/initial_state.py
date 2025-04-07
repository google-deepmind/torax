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
import jax.numpy as jnp
from torax import output
from torax import post_processing
from torax import state
from torax.config import build_runtime_params
from torax.config import config_args
from torax.config import runtime_params_slice
from torax.core_profiles import initialization
from torax.geometry import geometry
from torax.geometry import geometry_provider as geometry_provider_lib
from torax.orchestration import step_function
from torax.sources import source_profile_builders
from torax.torax_pydantic import file_restart as file_restart_pydantic_model
import xarray as xr


def get_initial_state_and_post_processed_outputs(
    t: float,
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice_provider: build_runtime_params.DynamicRuntimeParamsSliceProvider,
    geometry_provider: geometry_provider_lib.GeometryProvider,
    step_fn: step_function.SimulationStepFn,
) -> tuple[state.ToraxSimState, state.PostProcessedOutputs]:
  """Returns the initial state and post processed outputs."""
  dynamic_runtime_params_slice_for_init, geo_for_init = (
      build_runtime_params.get_consistent_dynamic_runtime_params_slice_and_geometry(
          t=t,
          dynamic_runtime_params_slice_provider=dynamic_runtime_params_slice_provider,
          geometry_provider=geometry_provider,
      )
  )
  initial_state = _get_initial_state(
      static_runtime_params_slice=static_runtime_params_slice,
      dynamic_runtime_params_slice=dynamic_runtime_params_slice_for_init,
      geo=geo_for_init,
      step_fn=step_fn,
  )
  post_processed_outputs = post_processing.make_post_processed_outputs(
      initial_state,
      dynamic_runtime_params_slice_for_init,
  )
  return initial_state, post_processed_outputs


def _get_initial_state(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    step_fn: step_function.SimulationStepFn,
) -> state.ToraxSimState:
  """Returns the initial state to be used by run_simulation()."""
  initial_core_profiles = initialization.initial_core_profiles(
      static_runtime_params_slice,
      dynamic_runtime_params_slice,
      geo,
      step_fn.stepper.source_models,
  )
  # Populate the starting state with source profiles from the implicit sources
  # before starting the run-loop. The explicit source profiles will be computed
  # inside the loop and will be merged with these implicit source profiles.
  initial_core_sources = source_profile_builders.get_initial_source_profiles(
      static_runtime_params_slice=static_runtime_params_slice,
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      geo=geo,
      core_profiles=initial_core_profiles,
      source_models=step_fn.stepper.source_models,
  )

  return state.ToraxSimState(
      t=jnp.array(dynamic_runtime_params_slice.numerics.t_initial),
      dt=jnp.zeros(()),
      core_profiles=initial_core_profiles,
      # This will be overridden within run_simulation().
      core_sources=initial_core_sources,
      core_transport=state.CoreTransport.zeros(geo),
      stepper_numeric_outputs=state.StepperNumericOutputs(
          stepper_error_state=0,
          outer_stepper_iterations=0,
          inner_solver_iterations=0,
      ),
      geometry=geo,
  )


def get_initial_state_and_post_processed_outputs_from_file(
    t_initial: float,
    file_restart: file_restart_pydantic_model.FileRestart,
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice_provider: build_runtime_params.DynamicRuntimeParamsSliceProvider,
    geometry_provider: geometry_provider_lib.GeometryProvider,
    step_fn: step_function.SimulationStepFn,
) -> tuple[state.ToraxSimState, state.PostProcessedOutputs]:
  """Returns the initial state and post processed outputs from a file."""
  data_tree = output.load_state_file(file_restart.filename)
  # Find the closest time in the given dataset.
  data_tree = data_tree.sel(time=file_restart.time, method='nearest')
  t_restart = data_tree.time.item()
  core_profiles_dataset = data_tree.children[output.CORE_PROFILES].dataset
  # Remap coordinates in saved file to be consistent with expectations of
  # how config_args parses xarrays.
  core_profiles_dataset = core_profiles_dataset.squeeze()
  if t_restart != t_initial:
    logging.warning(
        'Requested restart time %f not exactly available in state file %s.'
        ' Restarting from closest available time %f instead.',
        file_restart.time,
        file_restart.filename,
        t_restart,
    )

  dynamic_runtime_params_slice_for_init, geo_for_init = (
      build_runtime_params.get_consistent_dynamic_runtime_params_slice_and_geometry(
          t=t_initial,
          dynamic_runtime_params_slice_provider=dynamic_runtime_params_slice_provider,
          geometry_provider=geometry_provider,
      )
  )
  dynamic_runtime_params_slice_for_init, geo_for_init = (
      _override_initial_runtime_params_from_file(
          dynamic_runtime_params_slice_for_init,
          geo_for_init,
          t_restart,
          core_profiles_dataset,
      )
  )
  initial_state = _get_initial_state(
      static_runtime_params_slice=static_runtime_params_slice,
      dynamic_runtime_params_slice=dynamic_runtime_params_slice_for_init,
      geo=geo_for_init,
      step_fn=step_fn,
  )
  post_processed_dataset = data_tree.children[
      output.POST_PROCESSED_OUTPUTS
  ].dataset
  post_processed_dataset = post_processed_dataset.rename(
      {output.RHO_CELL_NORM: config_args.RHO_NORM}
  )
  post_processed_dataset = post_processed_dataset.squeeze()
  post_processed_outputs = post_processing.make_post_processed_outputs(
      initial_state,
      dynamic_runtime_params_slice_for_init,
  )
  post_processed_outputs = dataclasses.replace(
      post_processed_outputs,
      E_cumulative_fusion=post_processed_dataset.data_vars[
          'E_cumulative_fusion'
      ].to_numpy(),
      E_cumulative_external=post_processed_dataset.data_vars[
          'E_cumulative_external'
      ].to_numpy(),
  )
  core_profiles = dataclasses.replace(
      initial_state.core_profiles,
      vloop_lcfs=core_profiles_dataset.vloop_lcfs.values,
  )
  return (
      dataclasses.replace(
          initial_state,
          core_profiles=core_profiles,
      ),
      post_processed_outputs,
  )


def _override_initial_runtime_params_from_file(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    t_restart: float,
    ds: xr.Dataset,
) -> tuple[
    runtime_params_slice.DynamicRuntimeParamsSlice,
    geometry.Geometry,
]:
  """Override parts of runtime params slice from state in a file."""
  # pylint: disable=invalid-name
  dynamic_runtime_params_slice.numerics.t_initial = t_restart
  dynamic_runtime_params_slice.profile_conditions.Ip_tot = ds.data_vars[
      output.IP_PROFILE_FACE
  ].to_numpy()[-1]/1e6  # Convert from A to MA.
  dynamic_runtime_params_slice.profile_conditions.Te = ds.data_vars[
      output.TEMP_EL
  ].to_numpy()
  dynamic_runtime_params_slice.profile_conditions.Te_bound_right = ds.data_vars[
      output.TEMP_EL_RIGHT_BC
  ].to_numpy()
  dynamic_runtime_params_slice.profile_conditions.Ti = ds.data_vars[
      output.TEMP_ION
  ].to_numpy()
  dynamic_runtime_params_slice.profile_conditions.Ti_bound_right = ds.data_vars[
      output.TEMP_ION_RIGHT_BC
  ].to_numpy()
  dynamic_runtime_params_slice.profile_conditions.ne = ds.data_vars[
      output.NE
  ].to_numpy()
  dynamic_runtime_params_slice.profile_conditions.ne_bound_right = ds.data_vars[
      output.NE_RIGHT_BC
  ].to_numpy()
  dynamic_runtime_params_slice.profile_conditions.psi = ds.data_vars[
      output.PSI
  ].to_numpy()
  # When loading from file we want ne not to have transformations.
  # Both ne and the boundary condition are given in absolute values (not fGW).
  dynamic_runtime_params_slice.profile_conditions.ne_bound_right_is_fGW = False
  dynamic_runtime_params_slice.profile_conditions.ne_is_fGW = False
  dynamic_runtime_params_slice.profile_conditions.ne_bound_right_is_absolute = (
      True
  )
  # Additionally we want to avoid normalizing to nbar.
  dynamic_runtime_params_slice.profile_conditions.normalize_to_nbar = False
  # pylint: enable=invalid-name

  dynamic_runtime_params_slice, geo = runtime_params_slice.make_ip_consistent(
      dynamic_runtime_params_slice, geo
  )

  return dynamic_runtime_params_slice, geo

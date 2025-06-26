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
import numpy as np
import xarray as xr

from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.config import runtime_params_slice
from torax._src.core_profiles import initialization
from torax._src.geometry import geometry
from torax._src.geometry import geometry_provider as geometry_provider_lib
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.orchestration import sim_state
from torax._src.orchestration import step_function
from torax._src.output_tools import output
from torax._src.output_tools import post_processing
from torax._src.sources import source_profile_builders
from torax._src.torax_pydantic import file_restart as file_restart_pydantic_model


def get_initial_state_and_post_processed_outputs(
    t: float,
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice_provider: build_runtime_params.DynamicRuntimeParamsSliceProvider,
    geometry_provider: geometry_provider_lib.GeometryProvider,
    step_fn: step_function.SimulationStepFn,
) -> tuple[sim_state.ToraxSimState, post_processing.PostProcessedOutputs]:
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
) -> sim_state.ToraxSimState:
  """Returns the initial state to be used by run_simulation()."""
  initial_core_profiles = initialization.initial_core_profiles(
      static_runtime_params_slice,
      dynamic_runtime_params_slice,
      geo,
      step_fn.solver.source_models,
      step_fn.solver.neoclassical_models,
  )
  # Populate the starting state with source profiles from the implicit sources
  # before starting the run-loop. The explicit source profiles will be computed
  # inside the loop and will be merged with these implicit source profiles.
  initial_core_sources = source_profile_builders.get_all_source_profiles(
      static_runtime_params_slice=static_runtime_params_slice,
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      geo=geo,
      core_profiles=initial_core_profiles,
      source_models=step_fn.solver.source_models,
      neoclassical_models=step_fn.solver.neoclassical_models,
      conductivity=conductivity_base.Conductivity(
          sigma=initial_core_profiles.sigma,
          sigma_face=initial_core_profiles.sigma_face,
      ),
  )

  return sim_state.ToraxSimState(
      t=np.array(dynamic_runtime_params_slice.numerics.t_initial),
      dt=np.zeros(()),
      core_profiles=initial_core_profiles,
      # This will be overridden within run_simulation().
      core_sources=initial_core_sources,
      core_transport=state.CoreTransport.zeros(geo),
      solver_numeric_outputs=state.SolverNumericOutputs(
          solver_error_state=0,
          outer_solver_iterations=0,
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
) -> tuple[sim_state.ToraxSimState, post_processing.PostProcessedOutputs]:
  """Returns the initial state and post processed outputs from a file."""
  data_tree = output.load_state_file(file_restart.filename)
  # Find the closest time in the given dataset.
  data_tree = data_tree.sel(time=file_restart.time, method='nearest')
  t_restart = data_tree.time.item()
  profiles_dataset = data_tree.children[output.PROFILES].dataset
  profiles_dataset = profiles_dataset.squeeze()
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
  (
      static_runtime_params_slice_for_init,
      dynamic_runtime_params_slice_for_init,
      geo_for_init,
  ) = _override_initial_runtime_params_from_file(
      static_runtime_params_slice,
      dynamic_runtime_params_slice_for_init,
      geo_for_init,
      t_restart,
      profiles_dataset,
  )
  initial_state = _get_initial_state(
      static_runtime_params_slice=static_runtime_params_slice_for_init,
      dynamic_runtime_params_slice=dynamic_runtime_params_slice_for_init,
      geo=geo_for_init,
      step_fn=step_fn,
  )
  scalars_dataset = data_tree.children[output.SCALARS].dataset
  scalars_dataset = scalars_dataset.squeeze()
  post_processed_outputs = post_processing.make_post_processed_outputs(
      initial_state,
      dynamic_runtime_params_slice_for_init,
  )
  post_processed_outputs = dataclasses.replace(
      post_processed_outputs,
      E_fusion=scalars_dataset.data_vars['E_fusion'].to_numpy(),
      E_aux=scalars_dataset.data_vars['E_aux'].to_numpy(),
  )
  core_profiles = dataclasses.replace(
      initial_state.core_profiles,
      v_loop_lcfs=scalars_dataset.v_loop_lcfs.values,
  )
  numerics_dataset = data_tree.children[output.NUMERICS].dataset
  numerics_dataset = numerics_dataset.squeeze()
  sawtooth_crash = bool(numerics_dataset[output.SAWTOOTH_CRASH])
  outer_solver_iterations = int(
      numerics_dataset[output.OUTER_SOLVER_ITERATIONS]
  )
  inner_solver_iterations = int(
      numerics_dataset[output.INNER_SOLVER_ITERATIONS]
  )
  return (
      dataclasses.replace(
          initial_state,
          core_profiles=core_profiles,
          solver_numeric_outputs=state.SolverNumericOutputs(
              sawtooth_crash=sawtooth_crash,
              outer_solver_iterations=outer_solver_iterations,
              inner_solver_iterations=inner_solver_iterations,
          ),
      ),
      post_processed_outputs,
  )


def _override_initial_runtime_params_from_file(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    t_restart: float,
    profiles_ds: xr.Dataset,
) -> tuple[
    runtime_params_slice.StaticRuntimeParamsSlice,
    runtime_params_slice.DynamicRuntimeParamsSlice,
    geometry.Geometry,
]:
  """Override parts of runtime params slice from state in a file."""
  # pylint: disable=invalid-name
  dynamic_runtime_params_slice.numerics.t_initial = t_restart
  dynamic_runtime_params_slice.profile_conditions.Ip = profiles_ds.data_vars[
      output.IP_PROFILE
  ].to_numpy()[-1]
  dynamic_runtime_params_slice.profile_conditions.T_e = (
      profiles_ds.data_vars[output.T_E]
      .sel(rho_norm=profiles_ds.coords[output.RHO_CELL_NORM])
      .to_numpy()
  )
  dynamic_runtime_params_slice.profile_conditions.T_e_right_bc = (
      profiles_ds.data_vars[output.T_E]
      .sel(rho_norm=profiles_ds.coords[output.RHO_FACE_NORM][-1])
      .to_numpy()
  )
  dynamic_runtime_params_slice.profile_conditions.T_i = (
      profiles_ds.data_vars[output.T_I]
      .sel(rho_norm=profiles_ds.coords[output.RHO_CELL_NORM])
      .to_numpy()
  )
  dynamic_runtime_params_slice.profile_conditions.T_i_right_bc = (
      profiles_ds.data_vars[output.T_I]
      .sel(rho_norm=profiles_ds.coords[output.RHO_FACE_NORM][-1])
      .to_numpy()
  )
  # Density in output is in m^-3.
  dynamic_runtime_params_slice.profile_conditions.n_e = (
      profiles_ds.data_vars[output.N_E]
      .sel(rho_norm=profiles_ds.coords[output.RHO_CELL_NORM])
      .to_numpy()
  )
  dynamic_runtime_params_slice.profile_conditions.n_e_right_bc = (
      profiles_ds.data_vars[output.N_E]
      .sel(rho_norm=profiles_ds.coords[output.RHO_FACE_NORM][-1])
      .to_numpy()
  )
  dynamic_runtime_params_slice.profile_conditions.psi = (
      profiles_ds.data_vars[output.PSI]
      .sel(rho_norm=profiles_ds.coords[output.RHO_CELL_NORM])
      .to_numpy()
  )
  # When loading from file we want ne not to have transformations.
  # Both ne and the boundary condition are given in absolute values (not fGW).
  # Additionally we want to avoid normalizing to nbar.
  dynamic_runtime_params_slice.profile_conditions.n_e_right_bc_is_fGW = False
  dynamic_runtime_params_slice.profile_conditions.n_e_nbar_is_fGW = False
  static_runtime_params_slice = dataclasses.replace(
      static_runtime_params_slice,
      profile_conditions=dataclasses.replace(
          static_runtime_params_slice.profile_conditions,
          n_e_right_bc_is_absolute=True,
          normalize_n_e_to_nbar=False,
      ),
  )
  # pylint: enable=invalid-name

  dynamic_runtime_params_slice, geo = runtime_params_slice.make_ip_consistent(
      dynamic_runtime_params_slice, geo
  )

  return static_runtime_params_slice, dynamic_runtime_params_slice, geo

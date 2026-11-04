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
import json

from absl import logging
import jax
import jax.numpy as jnp
from torax._src import jax_utils
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import initialization
from torax._src.core_profiles import (
    runtime_params as core_profile_runtime_params,
)
from torax._src.core_profiles import solver_state as solver_state_lib
from torax._src.edge import base as edge_base
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.geometry import geometry_provider as geometry_provider_lib
from torax._src.orchestration import sim_state
from torax._src.orchestration import state_evaluation
from torax._src.orchestration import step_function
from torax._src.output_tools import output
from torax._src.output_tools import output_keys
from torax._src.output_tools import post_processing
from torax._src.physics import psi_calculations
from torax._src.torax_pydantic import file_restart as file_restart_config
import xarray as xr


@jax.jit
def get_initial_state_and_post_processed_outputs(
    step_fn: step_function.SimulationStepFn,
    t: float | None = None,
    runtime_params_overrides: (
        build_runtime_params.RuntimeParamsProvider | None
    ) = None,
    geometry_overrides: geometry_provider_lib.GeometryProvider | None = None,
) -> tuple[sim_state.SimState, post_processing.PostProcessedOutputs]:
  """Returns the initial state and post processed outputs.

  Args:
    step_fn: The step function to use for the simulation, this contains the
      physics models, default geometry provider and default runtime params
      provider to use.
    t: The time to use for the simulation. If not provided, the time will be
      chosen based on the values set in the numerics config.
    runtime_params_overrides: Optional runtime params provider to override the
      one set in the step function.
    geometry_overrides: Optional geometry provider to override the one set in
      the step function.

  Returns:
    A tuple of the initial state and post processed outputs.
  """
  runtime_params_provider = (
      runtime_params_overrides or step_fn.runtime_params_provider
  )
  geometry_provider = geometry_overrides or step_fn.geometry_provider
  t = (
      t
      if t is not None
      else runtime_params_provider.time_step_calculator.t_initial
  )

  runtime_params_for_init, geo_for_init = (
      build_runtime_params.get_consistent_runtime_params_and_geometry(
          t=t,
          runtime_params_provider=runtime_params_provider,
          geometry_provider=geometry_provider,
          is_initialization=True,
      )
  )

  initial_state = _get_initial_state(
      runtime_params=runtime_params_for_init,
      geo=geo_for_init,
      step_fn=step_fn,
  )

  post_processed_outputs = post_processing.make_post_processed_outputs(
      sim_state=initial_state,
      runtime_params=runtime_params_for_init,
      previous_post_processed_outputs=(
          post_processing.PostProcessedOutputs.zeros(geo_for_init)
      ),
  )
  return initial_state, post_processed_outputs


def _get_initial_state(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    step_fn: step_function.SimulationStepFn,
) -> sim_state.SimState:
  """Returns the initial state to be used by run_simulation()."""
  models = step_fn.solver.models
  solver_state = initialization.initial_solver_state(
      runtime_params=runtime_params,
      geo=geo,
      source_models=models.source_models,
      neoclassical_model=models.neoclassical_model,
      source_states=models.source_models.initial_state(runtime_params),
  )
  evaluated_state, initial_model_state = (
      state_evaluation.evaluate_initial_state(
          runtime_params=runtime_params,
          geo=geo,
          solver_state=solver_state,
          models=models,
      )
  )

  if models.edge_model is not None:
    assert runtime_params.edge is not None
    seed_edge_state = models.edge_model.initial_state(runtime_params.edge)
    edge_outputs, initial_edge_state = models.edge_model(
        runtime_params.edge,
        runtime_params,
        geo,
        evaluated_state.core_profiles,
        evaluated_state.core_sources,
        edge_state=seed_edge_state,
    )
    initial_model_state = dataclasses.replace(
        initial_model_state,
        edge=initial_edge_state,
    )
  else:
    edge_outputs = None
  return sim_state.SimState(
      t=jnp.array(
          runtime_params.time_step_calculator.t_initial,
          dtype=jax_utils.get_dtype(),
      ),
      dt=jnp.zeros((), dtype=jax_utils.get_dtype()),
      core_profiles=evaluated_state.core_profiles,
      core_sources=evaluated_state.core_sources,
      core_transport=evaluated_state.core_transport,
      solver_numeric_outputs=state.SolverNumericOutputs(
          solver_error_state=jnp.zeros((), jax_utils.get_int_dtype()),
          outer_solver_iterations=jnp.zeros((), jax_utils.get_int_dtype()),
          inner_solver_iterations=jnp.zeros((), jax_utils.get_int_dtype()),
          sawtooth_crash=False,
      ),
      geometry=geo,
      edge_outputs=edge_outputs,
      pedestal_output=evaluated_state.pedestal_output,
      model_state=initial_model_state,
      time_step_calculator_state=step_fn.time_step_calculator.initial_state(
          runtime_params
      ),
  )


def _cell_variable_from_profiles_dataset(
    profiles_ds: xr.Dataset,
    variable_name: str,
    geo: geometry.Geometry,
) -> cell_variable.CellVariable:
  """Builds a CellVariable from a profiles dataset variable."""
  values = jnp.asarray(
      profiles_ds.data_vars[variable_name]
      .sel(rho_norm=profiles_ds.coords[output_keys.RHO_CELL_NORM])
      .to_numpy(),
      dtype=jax_utils.get_dtype(),
  )
  right_bc = jnp.asarray(
      profiles_ds.data_vars[variable_name]
      .sel(rho_norm=profiles_ds.coords[output_keys.RHO_FACE_NORM][-1])
      .to_numpy(),
      dtype=jax_utils.get_dtype(),
  )
  return cell_variable.CellVariable(
      value=values,
      face_centers=geo.rho_face_norm,
      right_face_constraint=right_bc,
      right_face_grad_constraint=None,
  )


def _solver_state_from_profiles_dataset(
    profiles_ds: xr.Dataset,
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
) -> solver_state_lib.SolverState:
  """Builds a SolverState from a profiles dataset."""
  # pylint: disable=invalid-name
  T_i = _cell_variable_from_profiles_dataset(profiles_ds, output_keys.T_I, geo)
  T_e = _cell_variable_from_profiles_dataset(profiles_ds, output_keys.T_E, geo)
  n_e = _cell_variable_from_profiles_dataset(profiles_ds, output_keys.N_E, geo)
  toroidal_angular_velocity = _cell_variable_from_profiles_dataset(
      profiles_ds, output_keys.TOROIDAL_ANGULAR_VELOCITY, geo
  )

  psi_values = jnp.asarray(
      profiles_ds.data_vars[output_keys.PSI]
      .sel(rho_norm=profiles_ds.coords[output_keys.RHO_CELL_NORM])
      .to_numpy(),
      dtype=jax_utils.get_dtype(),
  )
  if (
      runtime_params.profile_conditions.psi_bc_source
      == core_profile_runtime_params.PsiBoundaryConditionSource.V_LOOP
  ):
    right_face_constraint = jnp.asarray(
        profiles_ds.data_vars[output_keys.PSI]
        .sel(rho_norm=profiles_ds.coords[output_keys.RHO_FACE_NORM][-1])
        .to_numpy(),
        dtype=jax_utils.get_dtype(),
    )
    right_face_grad_constraint = None
  else:
    right_face_constraint = None
    if output_keys.I_P_PROFILE in profiles_ds.data_vars:
      I_p = jnp.asarray(
          profiles_ds.data_vars[output_keys.I_P_PROFILE].to_numpy()[-1],
          dtype=jax_utils.get_dtype(),
      )
    else:
      I_p = runtime_params.profile_conditions.I_p
    right_face_grad_constraint = (
        psi_calculations.calculate_psi_grad_constraint_from_I_p(I_p, geo)
    )

  psi = cell_variable.CellVariable(
      value=psi_values,
      face_centers=geo.rho_face_norm,
      right_face_constraint=right_face_constraint,
      right_face_grad_constraint=right_face_grad_constraint,
  )
  return solver_state_lib.SolverState(
      T_i=T_i,
      T_e=T_e,
      psi=psi,
      n_e=n_e,
      toroidal_angular_velocity=toroidal_angular_velocity,
  )


def get_initial_state_and_post_processed_outputs_from_file(
    file_restart: file_restart_config.FileRestartConfig,
    step_fn: step_function.SimulationStepFn,
    runtime_params_overrides: (
        build_runtime_params.RuntimeParamsProvider | None
    ) = None,
    geometry_overrides: geometry_provider_lib.GeometryProvider | None = None,
) -> tuple[sim_state.SimState, post_processing.PostProcessedOutputs]:
  """Returns the initial state and post processed outputs from a file.

  Args:
    file_restart: The file restart config to use for the simulation.
    step_fn: The step function to use for the simulation, this contains the
      physics models, default geometry provider and default runtime params
      provider to use.
    runtime_params_overrides: Optional runtime params provider to override the
      one set in the step function.
    geometry_overrides: Optional geometry provider to override the one set in
      the step function.

  Returns:
    A tuple of the initial state and post processed outputs.
  """
  runtime_params_provider = (
      runtime_params_overrides or step_fn.runtime_params_provider
  )
  geometry_provider = geometry_overrides or step_fn.geometry_provider

  full_data_tree = output.load_state_file(str(file_restart.filename))
  first_time_in_file = float(full_data_tree.time.values[0])
  # Find the closest time in the given dataset.
  data_tree = full_data_tree.sel(time=file_restart.time, method='nearest')
  t_restart = data_tree.time.item()
  profiles_dataset = data_tree.children[output_keys.PROFILES].dataset
  profiles_dataset = profiles_dataset.squeeze()
  if t_restart != file_restart.time:
    logging.warning(
        'Requested restart time %f not exactly available in state file %s.'
        ' Restarting from closest available time %f instead.',
        file_restart.time,
        file_restart.filename,
        t_restart,
    )

  models = step_fn.solver.models
  if models.edge_model is not None:
    if output_keys.EDGE not in data_tree.children:
      raise ValueError(
          f'Missing expected {output_keys.EDGE!r} in restart DataTree.'
      )
    edge_datatree = data_tree.children[output_keys.EDGE]
    source_is_unstitched_restart = False
    raw_config = full_data_tree.attrs.get(output_keys.CONFIG)
    if raw_config is None and hasattr(full_data_tree, 'dataset'):
      raw_config = full_data_tree.dataset.attrs.get(output_keys.CONFIG)
    if raw_config:
      try:
        config_dict = json.loads(raw_config)
        restart_cfg = config_dict.get('restart')
        if isinstance(restart_cfg, dict):
          source_is_unstitched_restart = bool(
              restart_cfg.get('do_restart', False)
              and not restart_cfg.get('stitch', True)
          )
      except (ValueError, TypeError):
        pass
    skip_initial_edge_bcs = (
        t_restart == first_time_in_file and not source_is_unstitched_restart
    )
    initial_edge_bcs = (
        None
        if skip_initial_edge_bcs
        else edge_base.EdgeModelOutputs.from_datatree(edge_datatree)
    )
  else:
    edge_datatree = None
    initial_edge_bcs = None

  runtime_params_for_init, geo_for_init = (
      build_runtime_params.get_consistent_runtime_params_and_geometry(
          t=t_restart,
          runtime_params_provider=runtime_params_provider,
          geometry_provider=geometry_provider,
          edge_outputs=initial_edge_bcs,
          is_initialization=True,
      )
  )

  solver_state = _solver_state_from_profiles_dataset(
      profiles_dataset, runtime_params_for_init, geo_for_init
  )
  model_state = models.initial_state_from_datatree(
      data_tree.children.get(output_keys.MODEL_STATE),
      runtime_params_for_init,
  )
  scalars_dataset = data_tree.children[output_keys.SCALARS].dataset
  scalars_dataset = scalars_dataset.squeeze()
  v_loop_lcfs = jnp.asarray(
      scalars_dataset.v_loop_lcfs.values, dtype=jax_utils.get_dtype()
  )
  evaluated_state, _ = state_evaluation.evaluate_state(
      runtime_params=runtime_params_for_init,
      geo=geo_for_init,
      solver_state=solver_state,
      models=models,
      explicit_model_outputs=None,
      model_state=model_state,
      v_loop_lcfs=v_loop_lcfs,
  )

  if models.edge_model is not None:
    assert runtime_params_for_init.edge is not None
    assert model_state.edge is not None
    assert edge_datatree is not None
    # Evaluate edge_model to obtain the concrete EdgeModelOutputs PyTree
    # structure (needed to stack initial_state with subsequent SimStates in
    # jit_run_loop / StateHistory) and diagnostic fields at t_restart, then
    # overwrite the boundary conditions with the exact values saved in /edge.
    template_edge_outputs, _ = models.edge_model(
        runtime_params_for_init.edge,
        runtime_params_for_init,
        geo_for_init,
        evaluated_state.core_profiles,
        evaluated_state.core_sources,
        edge_state=model_state.edge,
    )
    edge_outputs = template_edge_outputs.replace_bcs_from_datatree(
        edge_datatree
    )
  else:
    edge_outputs = None

  numerics_dataset = data_tree.children[output_keys.NUMERICS].dataset
  numerics_dataset = numerics_dataset.squeeze()
  sawtooth_crash = bool(numerics_dataset[output_keys.SAWTOOTH_CRASH])
  outer_solver_iterations = int(
      numerics_dataset[output_keys.OUTER_SOLVER_ITERATIONS]
  )
  inner_solver_iterations = int(
      numerics_dataset[output_keys.INNER_SOLVER_ITERATIONS]
  )

  sim_state_instance = sim_state.SimState(
      t=jnp.array(t_restart, dtype=jax_utils.get_dtype()),
      dt=jnp.zeros((), dtype=jax_utils.get_dtype()),
      core_profiles=evaluated_state.core_profiles,
      core_sources=evaluated_state.core_sources,
      core_transport=evaluated_state.core_transport,
      solver_numeric_outputs=state.SolverNumericOutputs(
          sawtooth_crash=sawtooth_crash,
          solver_error_state=jnp.zeros((), jax_utils.get_int_dtype()),
          outer_solver_iterations=jnp.asarray(
              outer_solver_iterations, jax_utils.get_int_dtype()
          ),
          inner_solver_iterations=jnp.asarray(
              inner_solver_iterations, jax_utils.get_int_dtype()
          ),
      ),
      geometry=geo_for_init,
      edge_outputs=edge_outputs,
      pedestal_output=evaluated_state.pedestal_output,
      model_state=model_state,
      time_step_calculator_state=step_fn.time_step_calculator.initial_state(
          runtime_params_for_init
      ),
  )

  post_processed_outputs = post_processing.make_post_processed_outputs(
      sim_state=sim_state_instance,
      runtime_params=runtime_params_for_init,
      previous_post_processed_outputs=(
          post_processing.PostProcessedOutputs.zeros(geo_for_init)
      ),
  )
  post_processed_outputs = dataclasses.replace(
      post_processed_outputs,
      E_fusion=scalars_dataset.data_vars['E_fusion'].to_numpy(),
      E_aux_total=scalars_dataset.data_vars['E_aux_total'].to_numpy(),
      E_ohmic_e=scalars_dataset.data_vars['E_ohmic_e'].to_numpy(),
      E_external_injected=scalars_dataset.data_vars[
          'E_external_injected'
      ].to_numpy(),
      E_external_total=scalars_dataset.data_vars['E_external_total'].to_numpy(),
      dW_thermal_dt_smoothed=scalars_dataset.data_vars[
          'dW_thermal_dt_smoothed'
      ].to_numpy(),
      dW_thermal_i_dt_smoothed=scalars_dataset.data_vars[
          'dW_thermal_i_dt_smoothed'
      ].to_numpy(),
      dW_thermal_e_dt_smoothed=scalars_dataset.data_vars[
          'dW_thermal_e_dt_smoothed'
      ].to_numpy(),
  )
  return sim_state_instance, post_processed_outputs

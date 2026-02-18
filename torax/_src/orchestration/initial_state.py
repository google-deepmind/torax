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
import jax
import jax.numpy as jnp
from torax._src import jax_utils
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.config import runtime_params as runtime_params_lib
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
from torax._src.transport_model import transport_coefficients_builder
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
  t = t or runtime_params_provider.numerics.t_initial

  runtime_params_for_init, geo_for_init = (
      build_runtime_params.get_consistent_runtime_params_and_geometry(
          t=t,
          runtime_params_provider=runtime_params_provider,
          geometry_provider=geometry_provider,
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
      previous_post_processed_outputs=post_processing.PostProcessedOutputs.zeros(
          geo_for_init
      ),
  )
  return initial_state, post_processed_outputs


def _get_initial_state(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    step_fn: step_function.SimulationStepFn,
) -> sim_state.SimState:
  """Returns the initial state to be used by run_simulation()."""
  physics_models = step_fn.solver.physics_models
  initial_core_profiles = initialization.initial_core_profiles(
      runtime_params,
      geo,
      source_models=physics_models.source_models,
      neoclassical_models=physics_models.neoclassical_models,
  )
  initial_core_sources = source_profile_builders.get_all_source_profiles(
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=initial_core_profiles,
      source_models=physics_models.source_models,
      neoclassical_models=physics_models.neoclassical_models,
      conductivity=conductivity_base.Conductivity(
          sigma=initial_core_profiles.sigma,
          sigma_face=initial_core_profiles.sigma_face,
      ),
  )

  if physics_models.edge_model is not None:
    # If `runtime_params.edge.use_enrichment_model` is True, then the
    # `runtime_params.edge.enrichment_factor`
    # in this initialization step was calculated with a guess for the divertor
    # neutral pressure. This is only used to set edge fixed impurities if
    # PlasmaComposition.impurity_source_of_truth == CORE. All subsequent
    # calls to the edge model will use the divertor neutral pressure from a
    # previous calculation.
    edge_outputs = physics_models.edge_model(
        runtime_params,
        geo,
        initial_core_profiles,
        initial_core_sources,
        previous_edge_outputs=None,
    )
  else:
    edge_outputs = None

  transport_coeffs = (
      transport_coefficients_builder.calculate_all_transport_coeffs(
          physics_models.pedestal_model,
          physics_models.transport_model,
          physics_models.neoclassical_models,
          runtime_params,
          geo,
          initial_core_profiles,
      )
  )

  return sim_state.SimState(
      t=jnp.array(
          runtime_params.numerics.t_initial, dtype=jax_utils.get_dtype()
      ),
      dt=jnp.zeros((), dtype=jax_utils.get_dtype()),
      core_profiles=initial_core_profiles,
      core_sources=initial_core_sources,
      core_transport=transport_coeffs,
      solver_numeric_outputs=state.SolverNumericOutputs(
          solver_error_state=jnp.zeros((), jax_utils.get_int_dtype()),
          outer_solver_iterations=jnp.zeros((), jax_utils.get_int_dtype()),
          inner_solver_iterations=jnp.zeros((), jax_utils.get_int_dtype()),
          sawtooth_crash=False,
      ),
      geometry=geo,
      edge_outputs=edge_outputs,
  )


def get_initial_state_and_post_processed_outputs_from_file(
    file_restart: file_restart_pydantic_model.FileRestart,
    step_fn: step_function.SimulationStepFn,
    t: float | None = None,
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
  t = t or runtime_params_provider.numerics.t_initial

  data_tree = output.load_state_file(file_restart.filename)
  # Find the closest time in the given dataset.
  data_tree = data_tree.sel(time=file_restart.time, method='nearest')
  t_restart = data_tree.time.item()
  profiles_dataset = data_tree.children[output.PROFILES].dataset
  profiles_dataset = profiles_dataset.squeeze()
  if t_restart != t:
    logging.warning(
        'Requested restart time %f not exactly available in state file %s.'
        ' Restarting from closest available time %f instead.',
        file_restart.time,
        file_restart.filename,
        t_restart,
    )

  # No need for edge_outputs since file will contain all needed overrides.
  runtime_params_for_init, geo_for_init = (
      build_runtime_params.get_consistent_runtime_params_and_geometry(
          t=t,
          runtime_params_provider=runtime_params_provider,
          geometry_provider=geometry_provider,
      )
  )
  runtime_params_for_init, geo_for_init = (
      _override_initial_runtime_params_from_file(
          runtime_params_for_init,
          geo_for_init,
          t_restart,
          profiles_dataset,
      )
  )
  initial_state = _get_initial_state(
      runtime_params=runtime_params_for_init,
      geo=geo_for_init,
      step_fn=step_fn,
  )
  scalars_dataset = data_tree.children[output.SCALARS].dataset
  scalars_dataset = scalars_dataset.squeeze()
  post_processed_outputs = post_processing.make_post_processed_outputs(
      sim_state=initial_state,
      runtime_params=runtime_params_for_init,
      previous_post_processed_outputs=post_processing.PostProcessedOutputs.zeros(
          geo_for_init
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
              solver_error_state=jnp.zeros((), jax_utils.get_int_dtype()),
              outer_solver_iterations=jnp.asarray(
                  outer_solver_iterations, jax_utils.get_int_dtype()
              ),
              inner_solver_iterations=jnp.asarray(
                  inner_solver_iterations, jax_utils.get_int_dtype()
              ),
          ),
      ),
      post_processed_outputs,
  )


def _override_initial_runtime_params_from_file(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    t_restart: float,
    profiles_ds: xr.Dataset,
) -> tuple[runtime_params_lib.RuntimeParams, geometry.Geometry]:
  """Override parts of runtime params slice from state in a file."""
  # pylint: disable=invalid-name

  # TODO(b/446608829): Implement passing of impurity content needed when
  # restarting with inverse model extended-lengyel.

  runtime_params.numerics.t_initial = t_restart
  runtime_params.profile_conditions.Ip = profiles_ds.data_vars[
      output.IP_PROFILE
  ].to_numpy()[-1]
  runtime_params.profile_conditions.T_e = (
      profiles_ds.data_vars[output.T_E]
      .sel(rho_norm=profiles_ds.coords[output.RHO_CELL_NORM])
      .to_numpy()
  )
  runtime_params.profile_conditions.T_e_right_bc = (
      profiles_ds.data_vars[output.T_E]
      .sel(rho_norm=profiles_ds.coords[output.RHO_FACE_NORM][-1])
      .to_numpy()
  )
  runtime_params.profile_conditions.T_i = (
      profiles_ds.data_vars[output.T_I]
      .sel(rho_norm=profiles_ds.coords[output.RHO_CELL_NORM])
      .to_numpy()
  )
  runtime_params.profile_conditions.T_i_right_bc = (
      profiles_ds.data_vars[output.T_I]
      .sel(rho_norm=profiles_ds.coords[output.RHO_FACE_NORM][-1])
      .to_numpy()
  )
  # Density in output is in m^-3.
  runtime_params.profile_conditions.n_e = (
      profiles_ds.data_vars[output.N_E]
      .sel(rho_norm=profiles_ds.coords[output.RHO_CELL_NORM])
      .to_numpy()
  )
  runtime_params.profile_conditions.n_e_right_bc = (
      profiles_ds.data_vars[output.N_E]
      .sel(rho_norm=profiles_ds.coords[output.RHO_FACE_NORM][-1])
      .to_numpy()
  )
  runtime_params.profile_conditions.psi = (
      profiles_ds.data_vars[output.PSI]
      .sel(rho_norm=profiles_ds.coords[output.RHO_CELL_NORM])
      .to_numpy()
  )
  # When loading from file we want ne not to have transformations.
  # Both ne and the boundary condition are given in absolute values (not fGW).
  # Additionally we want to avoid normalizing to nbar.
  runtime_params.profile_conditions.n_e_right_bc_is_fGW = False
  runtime_params.profile_conditions.n_e_nbar_is_fGW = False
  runtime_params.profile_conditions.normalize_n_e_to_nbar = False
  runtime_params.profile_conditions.n_e_right_bc_is_absolute = True

  return runtime_params, geo

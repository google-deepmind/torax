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

"""Functions for performing a sawtooth step."""

import dataclasses
import functools
import jax
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import convertors
from torax._src.core_profiles import getters
from torax._src.core_profiles import updaters
from torax._src.edge import base as edge_base
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.geometry import geometry_provider as geometry_provider_lib
from torax._src.mhd.sawtooth import sawtooth_solver as sawtooth_solver_lib
from torax._src.orchestration import sim_state
from torax._src.orchestration import step_function_processing
from torax._src.output_tools import post_processing
from torax._src.physics import formulas
from torax._src.sources import source_profiles as source_profiles_lib

# pylint: disable=invalid-name


@functools.partial(
    jax.jit,
    static_argnames=[
        'sawtooth_solver',
    ],
)
def sawtooth_step(
    *,
    sawtooth_solver: sawtooth_solver_lib.SawtoothSolver,
    runtime_params_t: runtime_params_lib.RuntimeParams,
    runtime_params_provider: build_runtime_params.RuntimeParamsProvider,
    geo_t: geometry.Geometry,
    geometry_provider: geometry_provider_lib.GeometryProvider,
    explicit_source_profiles: source_profiles_lib.SourceProfiles,
    edge_outputs: edge_base.EdgeModelOutputs | None,
    input_state: sim_state.ToraxSimState,
    input_post_processed_outputs: post_processing.PostProcessedOutputs,
) -> tuple[sim_state.ToraxSimState, post_processing.PostProcessedOutputs]:
  """Checks for and handles a sawtooth crash.

  If a sawtooth model is provided and a crash is triggered, this method
  computes the post-crash state and returns it. Otherwise, returns the input
  state and post-processed outputs unchanged.

  Consecutive sawtooth crashes are not allowed since standard PDE steps
  may then not take place. Therefore if the input state has sawtooth_crash set
  to True, then no crash is triggered.

  Args:
    sawtooth_solver: Sawtooth model which carries out sawtooth step..
    runtime_params_t: Runtime params at time t.
    runtime_params_provider: Provider for runtime params.
    geo_t: Geometry at time t.
    geometry_provider: Provider for geometry.
    explicit_source_profiles: Explicit source profiles at time t.
    edge_outputs: Explicit edge outputs at time t.
    input_state: State at the start of the time step.
    input_post_processed_outputs: Post-processed outputs from the previous step.

  Returns:
    Returns a tuple (output_state, post_processed_outputs).
  """
  # Asserts needed for linter.
  assert runtime_params_t.mhd.sawtooth is not None
  dt_crash = runtime_params_t.mhd.sawtooth.crash_step_duration
  runtime_params_t_plus_crash_dt, geo_t_plus_crash_dt = (
      build_runtime_params.get_consistent_runtime_params_and_geometry(
          t=input_state.t + dt_crash,
          runtime_params_provider=runtime_params_provider,
          geometry_provider=geometry_provider,
      )
  )

  # Prepare core_profiles_t_plus_crash_dt with new boundary conditions
  # and prescribed profiles if present.
  core_profiles_t_plus_crash_dt = updaters.provide_core_profiles_t_plus_dt(
      dt=dt_crash,
      runtime_params_t=runtime_params_t,
      runtime_params_t_plus_dt=runtime_params_t_plus_crash_dt,
      geo_t_plus_dt=geo_t_plus_crash_dt,
      core_profiles_t=input_state.core_profiles,
  )

  (
      x_candidate,
      solver_numeric_outputs,
  ) = sawtooth_solver(
      t=input_state.t,
      dt=dt_crash,
      runtime_params_t=runtime_params_t,
      runtime_params_t_plus_dt=runtime_params_t_plus_crash_dt,
      geo_t=geo_t,
      geo_t_plus_dt=geo_t_plus_crash_dt,
      core_profiles_t=input_state.core_profiles,
      core_profiles_t_plus_dt=core_profiles_t_plus_crash_dt,
      explicit_source_profiles=explicit_source_profiles,
  )

  def _make_post_crash_state_and_post_processed_outputs():
    """Returns the post-crash state and post-processed outputs."""

    # We also update the temperature profiles over the sawtooth time to
    # maintain constant dW/dt over the sawtooth period. While not strictly
    # realistic this avoids non-physical dW/dt=perturbations in
    # post-processing.
    # Following the sawtooth redistribution, the PDE will take over the
    # energy evolution and the physical dW/dt corresponding to the new profile
    # distribution will be calculated.
    # This must be done here and not in the sawtooth model since the Solver
    # API does not include the post-processed outputs.
    x_evolved = _evolve_x_after_sawtooth(
        x_redistributed=x_candidate,
        runtime_params_t_plus_crash_dt=runtime_params_t_plus_crash_dt,
        core_profiles_redistributed=core_profiles_t_plus_crash_dt,
        geo_t_plus_crash_dt=geo_t_plus_crash_dt,
        previous_post_processed_outputs=input_post_processed_outputs,
        evolving_names=runtime_params_t.numerics.evolving_names,
        dt_crash=dt_crash,
    )

    return step_function_processing.finalize_outputs(
        t=input_state.t,
        dt=dt_crash,
        x_new=x_evolved,
        solver_numeric_outputs=solver_numeric_outputs,
        runtime_params_t_plus_dt=runtime_params_t_plus_crash_dt,
        geometry_t_plus_dt=geo_t_plus_crash_dt,
        core_profiles_t=input_state.core_profiles,
        core_profiles_t_plus_dt=core_profiles_t_plus_crash_dt,
        explicit_source_profiles=explicit_source_profiles,
        edge_outputs=edge_outputs,
        physics_models=sawtooth_solver.physics_models,
        evolving_names=runtime_params_t.numerics.evolving_names,
        input_post_processed_outputs=input_post_processed_outputs,
    )

  return jax.lax.cond(
      solver_numeric_outputs.sawtooth_crash,
      _make_post_crash_state_and_post_processed_outputs,
      lambda: (
          input_state,
          input_post_processed_outputs,
      ),
  )


def _evolve_x_after_sawtooth(
    x_redistributed: tuple[cell_variable.CellVariable, ...],
    runtime_params_t_plus_crash_dt: runtime_params_lib.RuntimeParams,
    core_profiles_redistributed: state.CoreProfiles,
    geo_t_plus_crash_dt: geometry.Geometry,
    previous_post_processed_outputs: post_processing.PostProcessedOutputs,
    evolving_names: tuple[str, ...],
    dt_crash: jax.Array,
) -> tuple[cell_variable.CellVariable, ...]:
  """Evolves the x_redistributed after the sawtooth redistribution."""

  updated_core_profiles = convertors.solver_x_tuple_to_core_profiles(
      x_new=x_redistributed,
      evolving_names=evolving_names,
      core_profiles=core_profiles_redistributed,
  )

  ions = getters.get_updated_ions(
      runtime_params_t_plus_crash_dt,
      geo_t_plus_crash_dt,
      updated_core_profiles.n_e,
      updated_core_profiles.T_e,
  )

  updated_core_profiles = dataclasses.replace(
      updated_core_profiles,
      n_i=ions.n_i,
      n_impurity=ions.n_impurity,
  )

  _, _, W_thermal_tot = formulas.calculate_stored_thermal_energy(
      updated_core_profiles.pressure_thermal_e,
      updated_core_profiles.pressure_thermal_i,
      updated_core_profiles.pressure_thermal_total,
      geo_t_plus_crash_dt,
  )

  # Update temperatures to maintain constant dW/dt over the sawtooth period.
  dW_target = previous_post_processed_outputs.dW_thermal_dt * dt_crash

  factor = 1 + dW_target / W_thermal_tot

  updated_core_profiles = dataclasses.replace(
      updated_core_profiles,
      T_e=dataclasses.replace(
          updated_core_profiles.T_e,
          value=updated_core_profiles.T_e.value * factor,
      ),
      T_i=dataclasses.replace(
          updated_core_profiles.T_i,
          value=updated_core_profiles.T_i.value * factor,
      ),
  )

  x_evolved = convertors.core_profiles_to_solver_x_tuple(
      updated_core_profiles,
      evolving_names,
  )

  return x_evolved

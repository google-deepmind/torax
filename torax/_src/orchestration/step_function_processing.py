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

"""Functions for pre and post processing used in the step function call."""

import functools

import jax
from torax._src import physics_models as physics_models_lib
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.config import runtime_params_slice
from torax._src.core_profiles import updaters
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.geometry import geometry_provider as geometry_provider_lib
from torax._src.orchestration import sim_state
from torax._src.output_tools import post_processing
from torax._src.sources import source_profile_builders
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.transport_model import transport_coefficients_builder


def pre_step(
    input_state: sim_state.ToraxSimState,
    runtime_params_provider: build_runtime_params.RuntimeParamsProvider,
    geometry_provider: geometry_provider_lib.GeometryProvider,
    physics_models: physics_models_lib.PhysicsModels,
) -> tuple[
    runtime_params_slice.RuntimeParams,
    geometry.Geometry,
    source_profiles_lib.SourceProfiles,
]:
  """Performs the pre-step operations for the step function."""
  runtime_params_t, geo_t = (
      build_runtime_params.get_consistent_runtime_params_and_geometry(
          t=input_state.t,
          runtime_params_provider=runtime_params_provider,
          geometry_provider=geometry_provider,
      )
  )

  # This only computes sources set to explicit in the
  # SourceConfig. All implicit sources will have their profiles
  # set to 0.
  explicit_source_profiles = source_profile_builders.build_source_profiles(
      runtime_params=runtime_params_t,
      geo=geo_t,
      core_profiles=input_state.core_profiles,
      source_models=physics_models.source_models,
      neoclassical_models=physics_models.neoclassical_models,
      explicit=True,
  )
  return runtime_params_t, geo_t, explicit_source_profiles


@functools.partial(
    jax.jit,
    static_argnames=[
        'physics_models',
        'evolving_names',
    ],
)
def finalize_outputs(
    t: jax.Array,
    dt: jax.Array,
    x_new: tuple[cell_variable.CellVariable, ...],
    solver_numeric_outputs: state.SolverNumericOutputs,
    geometry_t_plus_dt: geometry.Geometry,
    runtime_params_t_plus_dt: runtime_params_slice.RuntimeParams,
    core_profiles_t: state.CoreProfiles,
    core_profiles_t_plus_dt: state.CoreProfiles,
    explicit_source_profiles: source_profiles_lib.SourceProfiles,
    physics_models: physics_models_lib.PhysicsModels,
    evolving_names: tuple[str, ...],
    input_post_processed_outputs: post_processing.PostProcessedOutputs,
) -> tuple[sim_state.ToraxSimState, post_processing.PostProcessedOutputs]:
  """Returns the final state and post-processed outputs."""
  final_core_profiles, final_source_profiles = (
      updaters.update_core_and_source_profiles_after_step(
          dt=dt,
          x_new=x_new,
          runtime_params_t_plus_dt=runtime_params_t_plus_dt,
          geo=geometry_t_plus_dt,
          core_profiles_t=core_profiles_t,
          core_profiles_t_plus_dt=core_profiles_t_plus_dt,
          explicit_source_profiles=explicit_source_profiles,
          source_models=physics_models.source_models,
          neoclassical_models=physics_models.neoclassical_models,
          evolving_names=evolving_names,
      )
  )
  final_total_transport = (
      transport_coefficients_builder.calculate_total_transport_coeffs(
          physics_models.pedestal_model,
          physics_models.transport_model,
          physics_models.neoclassical_models,
          runtime_params_t_plus_dt,
          geometry_t_plus_dt,
          final_core_profiles,
      )
  )

  output_state = sim_state.ToraxSimState(
      t=t + dt,
      dt=dt,
      core_profiles=final_core_profiles,
      core_sources=final_source_profiles,
      core_transport=final_total_transport,
      geometry=geometry_t_plus_dt,
      solver_numeric_outputs=solver_numeric_outputs,
  )
  post_processed_outputs = post_processing.make_post_processed_outputs(
      sim_state=output_state,
      runtime_params=runtime_params_t_plus_dt,
      previous_post_processed_outputs=input_post_processed_outputs,
  )
  return output_state, post_processed_outputs

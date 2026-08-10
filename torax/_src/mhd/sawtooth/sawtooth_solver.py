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

"""A solver that implements the sawtooth trigger and redistribution."""

import dataclasses

import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import jax_utils
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import convertors
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib
from torax._src.solver import solver
from torax._src.sources import source_profiles as source_profiles_lib


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SawtoothPreparedStepState(solver.PreparedStepState):
  trigger_sawtooth: array_typing.BoolScalar
  rho_norm_q1: array_typing.FloatScalar
  runtime_params_t: runtime_params_lib.RuntimeParams
  geo_t: geometry.Geometry


# TODO(b/414537757). Sawtooth extensions.
# a. Full and incomplete Kadomtsev redistribution model.
# b. Porcelli model with free parameters and fast ion sensitivities.
# c. "Smooth" version that can work with forward-sensitivity-analysis and
#    stationary-state applications without the need for averaging.
class SawtoothSolver(solver.Solver):
  """Sawtooth trigger and redistribution, and carries out sawtooth step."""

  @jax.jit(static_argnames=['self'])
  def prepare_step(
      self,
      t: jax.Array,
      runtime_params_t: runtime_params_lib.RuntimeParams,
      geo_t: geometry.Geometry,
      core_profiles_t: state.CoreProfiles,
      explicit_source_profiles: source_profiles_lib.SourceProfiles,
      pedestal_transition_state: pedestal_transition_state_lib.PedestalTransitionState,
  ) -> SawtoothPreparedStepState:
    evolving_names = runtime_params_t.numerics.evolving_names
    x_old = convertors.core_profiles_to_solver_x_tuple(
        core_profiles_t, evolving_names
    )
    sawtooth_models = self.models.mhd_models.sawtooth_models
    if sawtooth_models is None:
      raise ValueError('Sawtooth model is None.')

    trigger_sawtooth, rho_norm_q1 = sawtooth_models.trigger_model(
        runtime_params_t,
        geo_t,
        core_profiles_t,
    )
    # When no sawtooth is triggered, rho_norm_q1=0. Clamp to eps so that the
    # redistribution branch (traced but not selected by jax.lax.cond) does not
    # encounter division-by-zero.
    rho_norm_q1 = jnp.maximum(rho_norm_q1, constants.CONSTANTS.eps)

    return SawtoothPreparedStepState(
        x_old=x_old,
        core_profiles_t=core_profiles_t,
        explicit_source_profiles=explicit_source_profiles,
        pedestal_transition_state=pedestal_transition_state,
        trigger_sawtooth=trigger_sawtooth,
        rho_norm_q1=rho_norm_q1,
        runtime_params_t=runtime_params_t,
        geo_t=geo_t,
    )

  @jax.jit(static_argnames=['self'])
  def solve_step(
      self,
      prepared_state: SawtoothPreparedStepState,
      dt: jax.Array,
      runtime_params_t_plus_dt: runtime_params_lib.RuntimeParams,
      geo_t_plus_dt: geometry.Geometry,
      core_profiles_t_plus_dt: state.CoreProfiles,
  ) -> tuple[
      tuple[cell_variable.CellVariable, ...],
      state.SolverNumericOutputs,
  ]:
    evolving_names = runtime_params_t_plus_dt.numerics.evolving_names
    sawtooth_models = self.models.mhd_models.sawtooth_models
    if sawtooth_models is None:
      raise ValueError('Sawtooth model is None.')

    def _redistribute_state() -> tuple[
        tuple[cell_variable.CellVariable, ...],
        state.SolverNumericOutputs,
    ]:
      redistributed_core_profiles = sawtooth_models.redistribution_model(
          prepared_state.rho_norm_q1,
          prepared_state.runtime_params_t,
          prepared_state.geo_t,
          prepared_state.core_profiles_t,
      )

      # Evolve the psi profile over the sawtooth time.
      # Redistribution maintains the same psi boundary condition. However,
      # over the course of the sawtooth time, the central solenoid must still
      # modify the psi profile. Since we don't calculate the psi PDE here, we
      # assume that for the short sawtooth time we can use the psidot from the
      # beginning of the step interval. This updates the bulk values. Later, the
      # boundary conditions are also updated at time t_plus_dt when
      # using `updaters.update_all_core_profiles_after_step`.
      evolved_psi_redistributed_value = (
          redistributed_core_profiles.psi.value
          + prepared_state.core_profiles_t.psidot.value * dt
      )
      evolved_core_profiles = dataclasses.replace(
          redistributed_core_profiles,
          psi=dataclasses.replace(
              redistributed_core_profiles.psi,
              value=evolved_psi_redistributed_value,
          ),
      )

      x_post_step = convertors.core_profiles_to_solver_x_tuple(
          evolved_core_profiles, evolving_names
      )

      solver_numeric_outputs_post_step = state.SolverNumericOutputs(
          sawtooth_crash=True,
          solver_error_state=jnp.array(0, jax_utils.get_int_dtype()),
          inner_solver_iterations=jnp.array(0, jax_utils.get_int_dtype()),
          outer_solver_iterations=jnp.array(0, jax_utils.get_int_dtype()),
      )

      return (
          x_post_step,
          solver_numeric_outputs_post_step,
      )

    # Return redistributed state attributes if triggered, otherwise return
    # unchanged state attributes.
    return jax.lax.cond(
        prepared_state.trigger_sawtooth,
        _redistribute_state,
        lambda: (
            tuple([
                getattr(prepared_state.core_profiles_t, name)
                for name in evolving_names
            ]),
            state.SolverNumericOutputs(
                sawtooth_crash=False,
                solver_error_state=jnp.array(0, jax_utils.get_int_dtype()),
                inner_solver_iterations=jnp.array(0, jax_utils.get_int_dtype()),
                outer_solver_iterations=jnp.array(0, jax_utils.get_int_dtype()),
            ),
        ),
    )

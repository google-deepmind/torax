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
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.core_profiles import convertors
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.solver import solver
from torax._src.sources import source_profiles as source_profiles_lib


# TODO(b/414537757). Sawtooth extensions.
# a. Full and incomplete Kadomtsev redistribution model.
# b. Porcelli model with free parameters and fast ion sensitivities.
# c. "Smooth" version that can work with forward-sensitivity-analysis and
#    stationary-state applications without the need for averaging.
class SawtoothSolver(solver.Solver):
  """Sawtooth trigger and redistribution, and carries out sawtooth step."""

  def _x_new(
      self,
      dt: jax.Array,
      runtime_params_t: runtime_params_slice.RuntimeParams,
      runtime_params_t_plus_dt: runtime_params_slice.RuntimeParams,
      geo_t: geometry.Geometry,
      geo_t_plus_dt: geometry.Geometry,
      core_profiles_t: state.CoreProfiles,
      core_profiles_t_plus_dt: state.CoreProfiles,
      explicit_source_profiles: source_profiles_lib.SourceProfiles,
      evolving_names: tuple[str, ...],
  ) -> tuple[
      tuple[cell_variable.CellVariable, ...],
      state.SolverNumericOutputs,
  ]:
    """Applies the sawtooth model and outputs new state attributes if triggered.

    If the trigger model indicates a crash has been triggered, an
    instantaneous redistribution model is applied. New state attributes
    following a short (configurable) dt are returned. Beyond the sawtooth
    redistribution, core_profiles are further updated by: the psidot assumed at
    time t; the new boundary conditions; sources, transport, geo, and pedestal
    outputs consistent with the new core_profiles at time t_plus_crash_dt.

    Args:
      dt: Sawtooth step duration.
      runtime_params_t: Runtime parameters at time t.
      runtime_params_t_plus_dt: Runtime parameters at time t + crash_dt.
      geo_t: Geometry at time t.
      geo_t_plus_dt: Geometry at time t + crash_dt.
      core_profiles_t: Core profiles at time t.
      core_profiles_t_plus_dt: Core profiles containing boundary conditions and
        prescribed profiles at time t + crash_dt.
      explicit_source_profiles: Explicit source profiles at time t.
      evolving_names: Names of evolving variables.

    Returns:
      Updated tuple of evolving CellVariables from CoreProfiles
      SolverNumericOutputs indicating a sawtooth crash.
    """
    sawtooth_models = self.physics_models.mhd_models.sawtooth_models
    if sawtooth_models is None:
      raise ValueError('Sawtooth model is None.')

    trigger_sawtooth, rho_norm_q1 = sawtooth_models.trigger_model(
        runtime_params_t,
        geo_t,
        core_profiles_t,
    )

    def _redistribute_state() -> tuple[
        tuple[cell_variable.CellVariable, ...],
        state.SolverNumericOutputs,
    ]:

      redistributed_core_profiles = sawtooth_models.redistribution_model(
          rho_norm_q1,
          runtime_params_t,
          geo_t,
          core_profiles_t,
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
          + core_profiles_t.psidot.value * dt
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
          sawtooth_crash=True
      )

      return (
          x_post_step,
          solver_numeric_outputs_post_step,
      )

    # Return redistributed state attributes if triggered, otherwise return
    # unchanged state attributes.
    return jax.lax.cond(
        trigger_sawtooth,
        _redistribute_state,
        lambda: (
            tuple([getattr(core_profiles_t, name) for name in evolving_names]),
            state.SolverNumericOutputs(),
        ),
    )

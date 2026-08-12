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

"""The LinearThetaMethod solver class."""

import dataclasses
import jax
from jax import numpy as jnp
from torax._src import jax_utils
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import convertors
from torax._src.fvm import block_1d_coeffs
from torax._src.fvm import calc_coeffs
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib
from torax._src.solver import predictor_corrector_method
from torax._src.solver import solver as solver_lib
from torax._src.sources import source_profiles


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class LinearPreparedStepState(solver_lib.PreparedStepState):
  coeffs_exp: block_1d_coeffs.Block1DCoeffs


class LinearThetaMethod(solver_lib.Solver):
  """Time step update using theta method, linearized on coefficients at t."""

  @jax.jit(static_argnames=['self'])
  def prepare_step(
      self,
      t: jax.Array,
      runtime_params_t: runtime_params_lib.RuntimeParams,
      geo_t: geometry.Geometry,
      core_profiles_t: state.CoreProfiles,
      explicit_source_profiles: source_profiles.SourceProfiles,
      pedestal_transition_state: pedestal_transition_state_lib.PedestalTransitionState,
  ) -> LinearPreparedStepState:
    evolving_names = runtime_params_t.numerics.evolving_names
    x_old = convertors.core_profiles_to_solver_x_tuple(
        core_profiles_t, evolving_names
    )
    coeffs_callback = calc_coeffs.CoeffsCallback(
        models=self.models,
        evolving_names=evolving_names,
    )
    coeffs_exp = coeffs_callback(
        runtime_params_t,
        geo_t,
        core_profiles_t,
        prev_core_profiles=None,
        dt=None,
        x=x_old,
        explicit_source_profiles=explicit_source_profiles,
        allow_pereverzev=True,
        explicit_call=True,
        pedestal_transition_state=pedestal_transition_state,
    )
    return LinearPreparedStepState(
        x_old=x_old,
        core_profiles_t=core_profiles_t,
        explicit_source_profiles=explicit_source_profiles,
        pedestal_transition_state=pedestal_transition_state,
        coeffs_exp=coeffs_exp,
    )

  @jax.jit(static_argnames=['self'])
  def solve_step(
      self,
      prepared_state: LinearPreparedStepState,
      dt: jax.Array,
      runtime_params_t_plus_dt: runtime_params_lib.RuntimeParams,
      geo_t_plus_dt: geometry.Geometry,
      core_profiles_t_plus_dt: state.CoreProfiles,
  ) -> tuple[
      tuple[cell_variable.CellVariable, ...],
      state.SolverNumericOutputs,
  ]:
    evolving_names = runtime_params_t_plus_dt.numerics.evolving_names
    x_old = prepared_state.x_old
    x_new_guess = convertors.core_profiles_to_solver_x_tuple(
        core_profiles_t_plus_dt, evolving_names
    )
    coeffs_callback = calc_coeffs.CoeffsCallback(
        models=self.models,
        evolving_names=evolving_names,
    )
    x_new = predictor_corrector_method.predictor_corrector_method(
        dt=dt,
        runtime_params_t_plus_dt=runtime_params_t_plus_dt,
        geo_t_plus_dt=geo_t_plus_dt,
        x_old=x_old,
        x_new_guess=x_new_guess,
        core_profiles_t=prepared_state.core_profiles_t,
        core_profiles_t_plus_dt=core_profiles_t_plus_dt,
        coeffs_exp=prepared_state.coeffs_exp,
        coeffs_callback=coeffs_callback,
        explicit_source_profiles=prepared_state.explicit_source_profiles,
        pedestal_transition_state=prepared_state.pedestal_transition_state,
    )

    if runtime_params_t_plus_dt.solver.use_predictor_corrector:
      inner_solver_iterations = (
          1 + runtime_params_t_plus_dt.solver.n_corrector_steps
      )
    else:
      inner_solver_iterations = 1

    solver_numeric_outputs = state.SolverNumericOutputs(
        inner_solver_iterations=jnp.array(
            inner_solver_iterations, jax_utils.get_int_dtype()
        ),
        outer_solver_iterations=jnp.array(1, jax_utils.get_int_dtype()),
        solver_error_state=jnp.array(0, jax_utils.get_int_dtype()),
        sawtooth_crash=False,
    )

    return (
        x_new,
        solver_numeric_outputs,
    )

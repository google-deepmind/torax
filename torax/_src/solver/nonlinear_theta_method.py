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

"""The NonLinearThetaMethod class."""
import abc
import dataclasses

import jax
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import convertors
from torax._src.fvm import block_1d_coeffs
from torax._src.fvm import calc_coeffs
from torax._src.fvm import cell_variable
from torax._src.fvm import enums
from torax._src.fvm import newton_raphson_solve_block
from torax._src.fvm import optimizer_solve_block
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib
from torax._src.solver import runtime_params as solver_runtime_params_lib
from torax._src.solver import solver
from torax._src.sources import source_profiles


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class OptimizerRuntimeParams(solver_runtime_params_lib.RuntimeParams):
  n_max_iterations: int
  loss_tol: float
  initial_guess_mode: int = dataclasses.field(metadata={'static': True})


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class NewtonRaphsonRuntimeParams(solver_runtime_params_lib.RuntimeParams):
  maxiter: int
  residual_tol: float
  residual_coarse_tol: float
  tau_min: float
  initial_guess_mode: int = dataclasses.field(metadata={'static': True})
  log_iterations: bool = dataclasses.field(metadata={'static': True})


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class NonlinearPreparedStepState(solver.PreparedStepState):
  coeffs_old: block_1d_coeffs.Block1DCoeffs
  coeffs_exp_linear: block_1d_coeffs.Block1DCoeffs | None


class NonlinearThetaMethod(solver.Solver):
  """Time step update using nonlinear solvers and the theta method."""

  @jax.jit(static_argnames=['self'])
  def prepare_step(
      self,
      t: jax.Array,
      runtime_params_t: runtime_params_lib.RuntimeParams,
      geo_t: geometry.Geometry,
      core_profiles_t: state.CoreProfiles,
      explicit_source_profiles: source_profiles.SourceProfiles,
      pedestal_transition_state: pedestal_transition_state_lib.PedestalTransitionState,
  ) -> NonlinearPreparedStepState:
    evolving_names = runtime_params_t.numerics.evolving_names
    x_old = convertors.core_profiles_to_solver_x_tuple(
        core_profiles_t, evolving_names
    )
    coeffs_callback = calc_coeffs.CoeffsCallback(
        models=self.models,
        evolving_names=evolving_names,
    )
    coeffs_old = coeffs_callback(
        runtime_params_t,
        geo_t,
        core_profiles_t,
        prev_core_profiles=None,
        dt=None,
        x=x_old,
        explicit_source_profiles=explicit_source_profiles,
        explicit_call=True,
        pedestal_transition_state=pedestal_transition_state,
    )

    solver_params = runtime_params_t.solver
    assert isinstance(
        solver_params, (OptimizerRuntimeParams, NewtonRaphsonRuntimeParams)
    )
    if solver_params.initial_guess_mode == enums.InitialGuessMode.LINEAR:
      coeffs_exp_linear = coeffs_callback(
          runtime_params_t,
          geo_t,
          core_profiles=core_profiles_t,
          prev_core_profiles=None,
          dt=None,
          x=x_old,
          explicit_source_profiles=explicit_source_profiles,
          allow_pereverzev=True,
          explicit_call=True,
          pedestal_transition_state=pedestal_transition_state,
      )
    else:
      coeffs_exp_linear = None

    return NonlinearPreparedStepState(
        x_old=x_old,
        core_profiles_t=core_profiles_t,
        explicit_source_profiles=explicit_source_profiles,
        pedestal_transition_state=pedestal_transition_state,
        coeffs_old=coeffs_old,
        coeffs_exp_linear=coeffs_exp_linear,
    )

  @jax.jit(static_argnames=['self'])
  def solve_step(
      self,
      prepared_state: NonlinearPreparedStepState,
      dt: jax.Array,
      runtime_params_t_plus_dt: runtime_params_lib.RuntimeParams,
      geo_t_plus_dt: geometry.Geometry,
      core_profiles_t_plus_dt: state.CoreProfiles,
  ) -> tuple[
      tuple[cell_variable.CellVariable, ...],
      state.SolverNumericOutputs,
  ]:
    evolving_names = runtime_params_t_plus_dt.numerics.evolving_names
    coeffs_callback = calc_coeffs.CoeffsCallback(
        models=self.models,
        evolving_names=evolving_names,
    )
    return self._solve_step_helper(
        prepared_state=prepared_state,
        dt=dt,
        runtime_params_t_plus_dt=runtime_params_t_plus_dt,
        geo_t_plus_dt=geo_t_plus_dt,
        core_profiles_t_plus_dt=core_profiles_t_plus_dt,
        coeffs_callback=coeffs_callback,
    )

  @abc.abstractmethod
  def _solve_step_helper(
      self,
      prepared_state: NonlinearPreparedStepState,
      dt: jax.Array,
      runtime_params_t_plus_dt: runtime_params_lib.RuntimeParams,
      geo_t_plus_dt: geometry.Geometry,
      core_profiles_t_plus_dt: state.CoreProfiles,
      coeffs_callback: calc_coeffs.CoeffsCallback,
  ) -> tuple[
      tuple[cell_variable.CellVariable, ...],
      state.SolverNumericOutputs,
  ]:
    """Abstract method for subclasses to implement the specific nonlinear solve."""
    ...


class OptimizerThetaMethod(NonlinearThetaMethod):
  """Minimize the squared norm of the residual of the theta method equation."""

  def _solve_step_helper(
      self,
      prepared_state: NonlinearPreparedStepState,
      dt: jax.Array,
      runtime_params_t_plus_dt: runtime_params_lib.RuntimeParams,
      geo_t_plus_dt: geometry.Geometry,
      core_profiles_t_plus_dt: state.CoreProfiles,
      coeffs_callback: calc_coeffs.CoeffsCallback,
  ) -> tuple[
      tuple[cell_variable.CellVariable, ...],
      state.SolverNumericOutputs,
  ]:
    evolving_names = runtime_params_t_plus_dt.numerics.evolving_names
    solver_params = runtime_params_t_plus_dt.solver
    assert isinstance(solver_params, OptimizerRuntimeParams)
    (
        x_new,
        solver_numeric_outputs,
    ) = optimizer_solve_block.optimizer_solve_block(
        dt=dt,
        runtime_params_t_plus_dt=runtime_params_t_plus_dt,
        geo_t_plus_dt=geo_t_plus_dt,
        x_old=prepared_state.x_old,
        core_profiles_t=prepared_state.core_profiles_t,
        core_profiles_t_plus_dt=core_profiles_t_plus_dt,
        models=self.models,
        explicit_source_profiles=prepared_state.explicit_source_profiles,
        coeffs_callback=coeffs_callback,
        evolving_names=evolving_names,
        initial_guess_mode=enums.InitialGuessMode(
            solver_params.initial_guess_mode,
        ),
        maxiter=solver_params.n_max_iterations,
        tol=solver_params.loss_tol,
        pedestal_transition_state=prepared_state.pedestal_transition_state,
        coeffs_old=prepared_state.coeffs_old,
        coeffs_exp_linear=prepared_state.coeffs_exp_linear,
    )
    return (
        x_new,
        solver_numeric_outputs,
    )


class NewtonRaphsonThetaMethod(NonlinearThetaMethod):
  """Nonlinear theta method using Newton Raphson."""

  def _solve_step_helper(
      self,
      prepared_state: NonlinearPreparedStepState,
      dt: jax.Array,
      runtime_params_t_plus_dt: runtime_params_lib.RuntimeParams,
      geo_t_plus_dt: geometry.Geometry,
      core_profiles_t_plus_dt: state.CoreProfiles,
      coeffs_callback: calc_coeffs.CoeffsCallback,
  ) -> tuple[
      tuple[cell_variable.CellVariable, ...],
      state.SolverNumericOutputs,
  ]:
    evolving_names = runtime_params_t_plus_dt.numerics.evolving_names
    solver_params = runtime_params_t_plus_dt.solver
    assert isinstance(solver_params, NewtonRaphsonRuntimeParams)

    (
        x_new,
        solver_numeric_outputs,
    ) = newton_raphson_solve_block.newton_raphson_solve_block(
        dt=dt,
        runtime_params_t_plus_dt=runtime_params_t_plus_dt,
        geo_t_plus_dt=geo_t_plus_dt,
        x_old=prepared_state.x_old,
        core_profiles_t=prepared_state.core_profiles_t,
        core_profiles_t_plus_dt=core_profiles_t_plus_dt,
        explicit_source_profiles=prepared_state.explicit_source_profiles,
        models=self.models,
        coeffs_callback=coeffs_callback,
        evolving_names=evolving_names,
        log_iterations=solver_params.log_iterations,
        initial_guess_mode=enums.InitialGuessMode(
            solver_params.initial_guess_mode
        ),
        maxiter=solver_params.maxiter,
        tol=solver_params.residual_tol,
        coarse_tol=solver_params.residual_coarse_tol,
        delta_reduction_factor=solver_params.delta_reduction_factor,
        tau_min=solver_params.tau_min,
        pedestal_transition_state=prepared_state.pedestal_transition_state,
        coeffs_old=prepared_state.coeffs_old,
        coeffs_exp_linear=prepared_state.coeffs_exp_linear,
    )
    return (
        x_new,
        solver_numeric_outputs,
    )

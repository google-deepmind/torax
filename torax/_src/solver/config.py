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

"""Pydantic config for Solver."""

import abc
import functools
from typing import Annotated, Literal
import pydantic
from torax._src import models as models_lib
from torax._src import tridiagonal
from torax._src.fvm import enums
from torax._src.solver import linear_theta_method
from torax._src.solver import nonlinear_theta_method
from torax._src.solver import runtime_params
from torax._src.solver import solver as solver_lib
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name


class SolverConfigBase(torax_pydantic.ConfigBase):
  """Base class for solver configs.

  Attributes:
    theta_implicit: The theta value in the theta method of time discretization.
      0 = explicit, 1 = fully implicit, 0.5 = Crank-Nicolson.
    use_fixed_point: Enables fixed-point iterations with the linear solver. If
      `False`, compilation is faster.
    n_fixed_point_iterations: The number of iterations for the fixed-point
      linear solver. Must be a positive integer.
    use_pereverzev: Use Pereverzev-Corrigan terms in the heat and particle flux
      when using the linear solver. Critical for stable calculation of stiff
      transport, at the cost of introducing non-physical lag during transients.
      Also used for the `'linear'` initial guess mode in the nonlinear solvers.
    implicit_solver_type: Selects the algorithm used to solve the
      block-tridiagonal linear system that arises at each implicit time step.
      The available options are:

      - `'thomas'`: The Thomas algorithm (block LU factorization), which
        solves the system in linear complexity in the number of grid cells by
        exploiting the tridiagonal sparsity structure. This is the recommended
        default for most simulations.
      - `'dense'`: Converts the block-tridiagonal system to a dense matrix
        and solves it with a general-purpose linear solver. This is primarily
        useful for debugging, validation, or small systems, as it does not
        exploit sparsity and scales cubically with the number of grid cells.
    chi_pereverzev: Large heat conductivity used for the Pereverzev-Corrigan
      term.
    D_pereverzev: Large particle diffusion used for the Pereverzev-Corrigan
      term.
    fixed_point_atol: Absolute tolerance on the residual norm for fixed-point
      iterations in the predictor-corrector solver. When
      `fixed_point_termination_criterion` is `'tolerance'`, iterations can exit
      early when the normalized residual falls below this threshold.
    fixed_point_rtol: Relative tolerance on the residual norm for fixed-point
      iterations in the predictor-corrector solver. When
      `fixed_point_termination_criterion` is `'tolerance'`, iterations can exit
      early when the normalized residual falls below this fraction of the
      initial residual.
    fixed_point_termination_criterion: Termination criterion for the fixed-point
      iterations in the predictor-corrector solver. Must be either `'tolerance'`
      (exit when tolerances are met) or `'max_iterations'` (always run for the
      specified number of iterations).
    fixed_point_sufficient_decrease: Sufficient decrease threshold used in the
      fixed-point (predictor-corrector) backtracking line search.
    fixed_point_use_backtracking: Enables backtracking linesearch for
      fixed-point iterations (predictor-corrector). Can be used with the linear
      solver, or in the initial guess for nonlinear solvers.
    delta_reduction_factor: Reduction factor by which the iteration step size is
      reduced during backtracking line search (for both fixed-point and
      Newton-Raphson iterations). If in a given iteration, the new state is
      unphysical (e.g., negative temperatures) or the residual increases in
      magnitude, a smaller step will be iteratively taken until the above
      conditions are met.
  """  # fmt: skip

  theta_implicit: Annotated[
      torax_pydantic.UnitInterval, torax_pydantic.JAX_STATIC
  ] = 1.0
  use_fixed_point: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  n_fixed_point_iterations: Annotated[
      pydantic.PositiveInt, torax_pydantic.JAX_STATIC
  ] = 10
  use_pereverzev: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  implicit_solver_type: Annotated[
      tridiagonal.SolverType, torax_pydantic.JAX_STATIC
  ] = tridiagonal.SolverType.THOMAS
  chi_pereverzev: pydantic.PositiveFloat = 30.0
  D_pereverzev: pydantic.NonNegativeFloat = 15.0
  fixed_point_atol: float = 1e-8
  fixed_point_rtol: float = 1e-6
  fixed_point_termination_criterion: Annotated[
      Literal['tolerance', 'max_iterations'], torax_pydantic.JAX_STATIC
  ] = 'max_iterations'
  fixed_point_sufficient_decrease: torax_pydantic.UnitInterval = 0.5
  fixed_point_use_backtracking: Annotated[bool, torax_pydantic.JAX_STATIC] = (
      False
  )
  delta_reduction_factor: float = 0.5

  @functools.cached_property
  def build_runtime_params(self) -> runtime_params.RuntimeParams:
    """Builds runtime params from the config."""
    return runtime_params.RuntimeParams(
        theta_implicit=self.theta_implicit,
        use_pereverzev=self.use_pereverzev,
        use_fixed_point=self.use_fixed_point,
        implicit_solver_type=self.implicit_solver_type,
        chi_pereverzev=self.chi_pereverzev,
        D_pereverzev=self.D_pereverzev,
        n_fixed_point_iterations=self.n_fixed_point_iterations,
        fixed_point_atol=self.fixed_point_atol,
        fixed_point_rtol=self.fixed_point_rtol,
        fixed_point_termination_criterion=self.fixed_point_termination_criterion,
        fixed_point_sufficient_decrease=self.fixed_point_sufficient_decrease,
        fixed_point_use_backtracking=self.fixed_point_use_backtracking,
        delta_reduction_factor=self.delta_reduction_factor,
    )

  @abc.abstractmethod
  def build_solver(
      self,
      models: models_lib.Models,
  ) -> solver_lib.Solver:
    """Builds a solver from the config."""


class LinearThetaMethodConfig(SolverConfigBase):
  """Linear solver where PDE coefficients are set at fixed values of the state.

  An approximation of the nonlinear solution is optionally carried out with a
  predictor-corrector method, i.e. fixed-point iteration of the PDE
  coefficients.

  Attributes:
    solver_type: Selected PDE solver algorithm.
  """

  solver_type: Annotated[Literal['linear'], torax_pydantic.JAX_STATIC] = (
      'linear'
  )

  def build_solver(
      self,
      models: models_lib.Models,
  ) -> solver_lib.Solver:
    return linear_theta_method.LinearThetaMethod(
        models=models,
    )


class NewtonRaphsonThetaMethodConfig(SolverConfigBase):
  """Nonlinear solver using the Newton-Raphson iterative algorithm.

  Includes backtracking line search and timestep backtracking for increased
  robustness.

  Attributes:
    solver_type: The type of solver to use, hardcoded to `'newton_raphson'`.
    log_iterations: If `True`, logs information about the internal state of the
      Newton-Raphson solver. For the first iteration, this contains the initial
      residual value and time-step size. For subsequent iterations, this
      contains the iteration step number, the current value of the residual, and
      the current value of `tau`, which is the relative reduction in Newton step
      size compared to the original Newton step size. If the solver does not
      converge, then these inner iterations will restart at a smaller timestep
      size if `allow_backtracking=True` in the `time_step_calculator`
      config dict.
    initial_guess_mode: Sets the approach taken for the initial guess into the
      Newton-Raphson solver for the first iteration. Two options are available:

      - `'x_old'`: Use the state at the beginning of the timestep.
      - `'linear'`: Use the linear solver to obtain an initial guess to
        warm-start the nonlinear solver. If used, it is recommended to do so
        with `use_fixed_point=True` and several corrector steps. It is also
        strongly recommended to set `use_pereverzev=True` if a stiff
        transport model like `qlknn` is used.
    n_max_iterations: Maximum number of allowed Newton iterations. If the number
      of iterations surpasses `n_max_iterations`, then the solver will exit in
      an unconverged state. The step will still be accepted if `residual <
      residual_coarse_tol`, otherwise `dt` backtracking will take place if
      enabled.
    residual_tol: PDE residual magnitude tolerance for successfully exiting the
      iterative solver.
    residual_coarse_tol: If the solver hits an exit criterion due to small steps
      or many iterations, but the residual is still below `residual_coarse_tol`,
      then the step is allowed to successfully pass, and a warning is passed to
      the user. Note that differentiating through the Newton-Raphson solver uses
      implicit differentiation (`jax.lax.custom_root`), which is only valid at a
      converged root. When taking gradients through a simulation, consider
      setting `residual_coarse_tol` equal to `residual_tol` to disable this
      fallback and force `dt` backtracking until every step converges to
      `residual_tol`.
    tau_min: `tau` is the relative reduction in step size (`delta /
      delta_original`) following backtracking line search, where
      `delta_original` is the step in state $x$ that minimizes the linearized
      PDE system. If following some iterations `tau < tau_min`, then the solver
      will exit in an unconverged state. The step will still be accepted if
      `residual < residual_coarse_tol`, otherwise `dt` backtracking will take
      place if enabled.
    vmap_linesearch: If `True`, use a vmapped (parallel) implementation of the
      linesearch that eagerly evaluates the residual at all linesearch step
      sizes and returns the first acceptable one. This can provide a significant
      speedup when many linesearch steps are required (especially on GPU), at
      the cost of higher peak memory usage and compilation time.
    max_linesearch_steps: Maximum number of backtracking linesearch steps per
      Newton iteration.
  """  # fmt: skip

  solver_type: Annotated[
      Literal['newton_raphson'], torax_pydantic.JAX_STATIC
  ] = 'newton_raphson'
  log_iterations: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  initial_guess_mode: Annotated[
      enums.InitialGuessMode, torax_pydantic.JAX_STATIC
  ] = enums.InitialGuessMode.LINEAR
  n_max_iterations: pydantic.NonNegativeInt = 30
  residual_tol: float = 1e-5
  residual_coarse_tol: float = 1e-2
  tau_min: float = 0.01
  vmap_linesearch: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  max_linesearch_steps: Annotated[
      pydantic.PositiveInt, torax_pydantic.JAX_STATIC
  ] = 100

  @functools.cached_property
  def build_runtime_params(
      self,
  ) -> nonlinear_theta_method.NewtonRaphsonRuntimeParams:
    base_params = super().build_runtime_params
    return nonlinear_theta_method.NewtonRaphsonRuntimeParams(
        **vars(base_params),
        maxiter=self.n_max_iterations,
        residual_tol=self.residual_tol,
        residual_coarse_tol=self.residual_coarse_tol,
        tau_min=self.tau_min,
        initial_guess_mode=self.initial_guess_mode.value,  # pyrefly: ignore[bad-argument-type]
        log_iterations=self.log_iterations,
        vmap_linesearch=self.vmap_linesearch,
        max_linesearch_steps=self.max_linesearch_steps,
    )

  def build_solver(
      self,
      models: models_lib.Models,
  ) -> nonlinear_theta_method.NewtonRaphsonThetaMethod:
    return nonlinear_theta_method.NewtonRaphsonThetaMethod(
        models=models,
    )


class OptimizerThetaMethodConfig(SolverConfigBase):
  """Nonlinear solver using the `jaxopt` library.

  Attributes:
    solver_type: The type of solver to use, hardcoded to `'optimizer'`.
    initial_guess_mode: Sets the approach taken for the initial guess into the
      optimizer for the first iteration. Two options are available:

      - `'x_old'`: Use the state at the beginning of the timestep.
      - `'linear'`: Use the linear solver to obtain an initial guess to
        warm-start the nonlinear solver. If used, it is recommended to do so
        with `use_fixed_point=True` and several corrector steps. It is also
        strongly recommended to set `use_pereverzev=True` if a stiff
        transport model like `qlknn` is used.
    n_max_iterations: Maximum number of allowed optimizer iterations.
    loss_tol: PDE loss magnitude tolerance for successfully exiting the
      iterative solver. Note: the default tolerance here is smaller than the
      default tolerance for the Newton-Raphson solver because it is a tolerance
      on the loss (square of the residual).
  """  # fmt: skip

  solver_type: Annotated[Literal['optimizer'], torax_pydantic.JAX_STATIC] = (
      'optimizer'
  )
  initial_guess_mode: Annotated[
      enums.InitialGuessMode, torax_pydantic.JAX_STATIC
  ] = enums.InitialGuessMode.LINEAR
  n_max_iterations: pydantic.NonNegativeInt = 100
  loss_tol: float = 1e-10

  @functools.cached_property
  def build_runtime_params(
      self,
  ) -> nonlinear_theta_method.OptimizerRuntimeParams:
    base_params = super().build_runtime_params
    return nonlinear_theta_method.OptimizerRuntimeParams(
        **vars(base_params),
        n_max_iterations=self.n_max_iterations,
        loss_tol=self.loss_tol,
        initial_guess_mode=self.initial_guess_mode.value,  # pyrefly: ignore[bad-argument-type]
    )

  def build_solver(
      self,
      models: models_lib.Models,
  ) -> nonlinear_theta_method.OptimizerThetaMethod:
    return nonlinear_theta_method.OptimizerThetaMethod(
        models=models,
    )


SolverConfig = Annotated[
    LinearThetaMethodConfig
    | NewtonRaphsonThetaMethodConfig
    | OptimizerThetaMethodConfig,
    pydantic.Discriminator('solver_type'),
]

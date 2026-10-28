.. _solver_config:

solver
######

Select and configure the ``Solver`` object, which evolves the PDE system by one
timestep. See :ref:`solver_details` for further details. The dictionary consists
of keys common to all solvers. Additional fields for parameters pertaining to a
specific solver are defined in the relevant section below.

``solver_type`` (str [default = 'linear'])
  Selected PDE solver algorithm. The current options are:

* ``'linear'``
    Linear solver where PDE coefficients are set at fixed values of the state.
    An approximation of the nonlinear solution is optionally carried out with a
    predictor-corrector method, i.e. fixed point iteration of the PDE
    coefficients.

* ``'newton_raphson'``
    Nonlinear solver using the Newton-Raphson iterative algorithm, with
    backtracking line search, and timestep backtracking, for increased
    robustness.

* ``'optimizer'``
    Nonlinear solver using the jaxopt library.

``theta_implicit`` (float [default = 1.0])
  theta value in the theta method of time discretization. 0 = explicit, 1 =
  fully implicit, 0.5 = Crank-Nicolson.

``use_fixed_point`` (bool [default = True])
  Enables use_fixed_point iterations with the linear solver.

``n_fixed_point_iterations`` (int [default = 10])
  The number of iterations for the fixed-point linear solver. Must be a positive integer.

``atol`` (float | None [default = None])
  Absolute tolerance for fixed-point iterations in the predictor-corrector solver.
  If specified, iterations can exit early when the normalized residual falls below this threshold.

``rtol`` (float | None [default = None])
  Relative tolerance for fixed-point iterations in the predictor-corrector solver.
  If specified, iterations can exit early when the normalized residual falls below this fraction of the initial residual.

``fixed_point_termination_criterion`` (str [default = 'max_iterations'])
  Termination criterion for the fixed-point iterations in the predictor-corrector solver.
  Must be either 'tolerance' (exit when tolerances are met) or 'max_iterations' (always run for the specified number of iterations).

``fixed_point_use_backtracking`` (bool [default = False])
  Enables backtracking linesearch for fixed-point iterations (predictor-corrector). Can be used with the linear solver, or in the initial guess for nonlinear solvers.

``fixed_point_sufficient_decrease`` (float [default = 0.5])
  Sufficient decrease threshold used in the fixed point (predictor-corrector) backtracking line search.

``delta_reduction_factor`` (float [default = 0.5])
  Factor by which the step size is reduced during backtracking.

``use_pereverzev`` (bool [default = False])
  Use Pereverzev-Corrigan terms in the heat and particle flux when using the
  linear solver. Critical for stable calculation of stiff transport, at the cost
  of introducing non-physical lag during transient. Also used for the
  ``linear_step`` initial guess mode in the nonlinear solvers.

``chi_pereverzev`` (float [default = 30.0])
  Large heat conductivity used for the Pereverzev-Corrigan term.

``D_pereverzev`` (float [default = 15.0])
  Large particle diffusion used for the Pereverzev-Corrigan term.

``implicit_solver_type`` (str [default = 'THOMAS'])
  Selects the algorithm used to solve the block-tridiagonal linear system that
  arises at each implicit time step. The available options are:

* ``'THOMAS'``
    The Thomas algorithm (block LU factorization), which solves the system in
    linear complexity in the number of grid cells by exploiting the tridiagonal
    sparsity structure. This is the recommended default for most simulations.

* ``'DENSE'``
    Converts the block-tridiagonal system to a dense matrix and solves it with a
    general-purpose linear solver. This is primarily useful for debugging,
    validation or small systems, as it does not exploit sparsity and scales
    as cubicly with the number of grid cells.

linear
======

No extra parameters are defined for the ``linear`` solver.

newton_raphson
==============

.. _log_iterations:

``log_iterations`` (bool [default = False])
  If ``True``, logs information about the internal state of the Newton-Raphson
  solver. For the first iteration, this contains the initial residual value and
  time-step size. For subsequent iterations, this contains the iteration step
  number, the current value of the residual, and the current value of ``tau``,
  which is the relative reduction in Newton step size compared to the original
  Newton step size. If the solver does not converge, then these inner iterations
  will restart at a smaller timestep size if ``adaptive_dt=True`` in the
  ``numerics`` config dict.

``initial_guess_mode`` (str [default = 'linear_step'])
  Sets the approach taken for the initial guess into the Newton-Raphson solver
  for the first iteration. Two options are available:

* ``x_old``
    Use the state at the beginning of the timestep.

* ``linear_step``
    Use the linear solver to obtain an initial guess to warm-start the nonlinear
    solver. If used, is recommended to do so with the use_fixed_point
    solver and several corrector steps. It is also strongly recommended to
    ``use_pereverzev=True`` if a stiff transport model like qlknn is used.

``residual_tol`` (float [default = 1e-5])
  PDE residual magnitude tolerance for successfully exiting the iterative
  solver.

``residual_coarse_tol`` (float [default = 1e-2])
  If the solver hits an exit criterion due to small steps or many iterations,
  but the residual is still below ``coarse_tol``, then the step is allowed to
  successfully pass, and a warning is passed to the user. Note that
  differentiating through the Newton-Raphson solver uses implicit
  differentiation (``jax.lax.custom_root``), which is only valid at a converged
  root. When taking gradients through a simulation, consider setting
  ``residual_coarse_tol`` equal to ``residual_tol`` to disable this fallback
  and force ``dt`` backtracking until every step converges to ``residual_tol``.

``n_max_iterations`` (int [default = 30])
  Maximum number of allowed Newton iterations. If the number of iterations
  surpasses ``maxiter``, then the solver will exit in an unconverged state.
  The step will still be accepted if ``residual < coarse_tol``, otherwise dt
  backtracking will take place if enabled.

``delta_reduction_factor`` (float [default = 0.5])
  Reduction of Newton iteration step size in the backtracking line search.
  If in a given iteration, the new state is unphysical (e.g. negative
  temperatures) or the residual increases in magnitude, then a smaller step will
  be iteratively taken until the above conditions are met.

``tau_min`` (float [default = 0.01])
  tau is the relative reduction in step size: delta/delta_original, following
  backtracking line search, where delta_original is the step in state :math:`x`
  that minimizes the linearized PDE system. If following some iterations,
  ``tau < tau_min``, then the solver will exit in an unconverged state. The step
  will still be accepted if ``residual < coarse_tol``, otherwise dt backtracking
  will take place if enabled.

``vmap_linesearch`` (bool [default = False])
  If True, use a vmapped implementation of the linesearch. This can provide a
  significant speedup when many linesearch steps are required, at the cost of
  higher peak memory usage and compilation time.

optimizer
=========

``initial_guess_mode`` (str [default = 'linear_step'])
  Sets the approach taken for the initial guess into the Newton-Raphson solver
  for the first iteration. Two options are available:

* ``x_old``
    Use the state at the beginning of the timestep.

* ``linear_step``
    Use the linear solver to obtain an initial guess to warm-start the nonlinear
    solver. If used, is recommended to do so with the use_fixed_point
    solver and several corrector steps. It is also strongly recommended to
    use_pereverzev=True if a stiff transport model like qlknn is used.

``loss_tol`` (float [default = 1e-10])
  PDE loss magnitude tolerance for successfully exiting the iterative solver.
  Note: the default tolerance here is smaller than the default tolerance for
  the Newton-Raphson solver because it's a tolerance on the loss (square of the
  residual).

``n_max_iterations`` (int [default = 100])
  Maximum number of allowed optimizer iterations.

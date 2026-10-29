.. _time_step_calculator_config:

time_step_calculator
####################

``calculator_type`` (str [default = 'chi'])
  The name of the ``time_step_calculator``, a method which calculates ``dt`` at
  every timestep. Three methods are currently available:

* ``'fixed'``
    ``dt`` is equal to the configured ``dt`` parameter.
    If the Newton-Raphson solver is being used and ``allow_backtracking==True``
    in the ``time_step_calculator`` config dict, then in practice some steps may
    have lower ``dt`` if the solver needed to backtrack.

* ``'chi'``
    adaptive dt method, where ``dt`` is a multiple of a base dt inspired by the
    explicit stability limit for parabolic PDEs:
    :math:`dt_{base}=\frac{dx^2}{2\chi}`, where :math:`dx` is the grid
    resolution  and :math:`\chi=max(\chi_i, \chi_e)`.
    ``dt=chi_timestep_prefactor * dt_base``, where ``chi_timestep_prefactor``
    can be significantly larger than unity for implicit solvers. Scaling the
    timestep to be :math:`\propto \chi` helps protect against traversing through
    fast transients, if there is a desire for them to be fully resolved.

* ``'from_previous_dt'``
    On the first step, ``dt`` is equal to ``initial_dt``. On subsequent steps,
    ``dt`` is equal to the previous non-sawtooth time step duration multiplied
    by ``dt_backtracking_factor`` (capped at ``max_dt``). Requires
    ``allow_backtracking=True``. This can speed up simulations that take many dt
    reduction steps, as the dt is initialized to a smaller value, rather than
    starting from max_dt every time. It can also speed up simulations where a
    small dt is required for a short section, but then a larger dt is
    appropriate for the remainder of the simulation.

``t_initial`` (float [default = 0.0])
  Simulation start time, in units of seconds.

``t_final`` (float [default = 5.0])
  Simulation end time, in units of seconds.

``tolerance`` (float [default = 1e-7])
  The tolerance within the final time for which the simulation will be
  considered done.

``exact_t_final`` (bool [default = True])
  If True, ensures that the simulation end time is exactly ``t_final``, by
  adapting the final ``dt`` to match.

``min_dt`` (float [default = 1e-8])
  Minimum timestep allowed in simulation.

``dt_backtracking_factor`` (float [default = 3.0])
  Used only if ``allow_backtracking`` is True (or for
  ``'from_previous_dt'``). Factor by which the timestep is reduced if a
  nonlinear solver does not converge for a given timestep.

``allow_backtracking`` (bool [default = True])
  If True, then if a nonlinear solver does not converge for a given timestep,
  then dt-backtracking is applied and a new Solver call is made where the
  timestep is reduced by a factor of ``dt_backtracking_factor``. This is applied
  iteratively until either the solver converges, or ``min_dt`` is reached.

``max_dt`` (float [default = 2.0])
  Maximum size of timesteps allowed in the simulation. Only available when
  ``calculator_type`` is ``'chi'`` or ``'from_previous_dt'``.

``chi_timestep_prefactor`` (float [default = 50.0])
  Prefactor in front of ``'chi'`` time_step_calculator base timestep
  :math:`dt_{base}=\frac{dx^2}{2\chi}`. Only available when
  ``calculator_type`` is ``'chi'``.

``dt`` ( **time-varying-scalar** [default = 1e-1])
  Timestep used for ``'fixed'`` time_step_calculator. All values must be
  non-negative. Only available when ``calculator_type`` is ``'fixed'``.
  Specifying a time-dependent ``dt`` can be useful in cases where there are
  known transitions (e.g. ramp-up, ramp-down, and/or LH transitions) and a
  different granularity of simulation is required for each phase. Note that the
  default interpolation mode for ``dt`` has been internally imposed as
  ``'STEP'``.
  E.g.:

  .. code-block:: python

    # Setting a higher time resolution between 5s and 10s.
    'dt': {0.0: 1e-1, 5.0: 1e-2, 10.0: 1e-1}

``initial_dt`` (float [default = 1e-1])
  Initial timestep duration (must be positive) used for the first step of the
  ``'from_previous_dt'`` time_step_calculator. Only available when
  ``calculator_type`` is ``'from_previous_dt'``.

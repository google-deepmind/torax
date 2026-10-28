.. _numerics_dataclass:

numerics
########

Configures simulation control such as time settings and timestep calculation,
equations being solved, constant numerical variables.

``t_initial`` (float [default = 0.0])
  Simulation start time, in units of seconds.

``t_final`` (float [default = 5.0])
  Simulation end time, in units of seconds.

``exact_t_final`` (bool [default = True])
  If True, ensures that the simulation end time is exactly ``t_final``, by
  adapting the final ``dt`` to match.

``max_dt`` (float [default = 2.0])
  Maximum size of timesteps allowed in the simulation. This is only used with
  the ``chi_time_step_calculator`` time_step_calculator.

``min_dt`` (float [default = 1e-8])
  Minimum timestep allowed in simulation.

``chi_timestep_prefactor`` (float [default = 50.0])
  Prefactor in front of ``chi_timestep_calculator`` base timestep
  :math:`dt_{base}=\frac{dx^2}{2\chi}` (see :ref:`time_step_calculator`).

``fixed_dt`` ( **time-varying-scalar** [default = 1e-1])
  Timestep used for ``fixed_time_step_calculator`` (see
  :ref:`time_step_calculator`). All values must be non-negative.
  Specifying a time-dependent fixed_dt can be useful in cases where there are
  known transitions (e.g. ramp-up, ramp-down, and/or LH transitions) and a
  different granularity of simulation is required for each phase. Note that the
  default interpolation mode for `fixed_dt` has been internally imposed as
  `'STEP'`.
  E.g.:

  .. code-block:: python

    # Setting a higher time resolution between 5s and 10s.
    'fixed_dt': {0.0: 1e-1, 5.0: 1e-2, 10.0: 1e-1}

``adaptive_dt`` (bool [default = True])
  If True, then if a nonlinear solver does not converge for a given timestep,
  then dt-backtracking is applied and a new Solver call is made where the
  timestep is reduced by a factor of ``dt_reduction_factor``. This is applied
  iteratively until either the solver converges, or ``min_dt`` is reached.

``dt_reduction_factor`` (float [default = 3.0])
  Used only if ``adaptive_dt`` is True. Factor by which the timestep is reduced
  if a nonlinear solver does not converge for a given timestep.

``evolve_ion_heat`` (bool [default = True])
  Solve the ion heat equation in the time-dependent PDE.

``evolve_electron_heat`` (bool [default = True])
  Solve the electron heat equation in the time-dependent PDE.

``evolve_current`` (bool [default = False])
  Solve the current diffusion equation (evolving :math:`\psi`) in the
  time-dependent PDE.

``evolve_density`` (bool [default = False])
  Solve the electron density equation in the time-dependent PDE.

``enable_fast_ions`` (bool [default = False])
  Enables the tracking and computation of fast ion density and temperature
  profiles. Required to be True to use ``profile_conditions.fast_ions``,
  ``transport.fast_ion_stabilization``, and to allow sources like ``toric-nn``
  to provide fast ion profiles to the simulation.

``current_diffusion_time_reduction_factor`` (**positive-time-varying-scalar** [default = 1.0])
  Factor dividing the transient term in the poloidal flux (:math:`\psi`) PDE
  to reduce the current diffusion timescale toward the energy confinement
  timescale, for testing purposes. Does not modify physical conductivity,
  loop voltage, or Ohmic heating.

``dW_dt_smoothing_time_scale`` (float [default = 0.3])
  Time scale [s] for the exponential moving average smoothing of dW/dt terms
  used in P_SOL and confinement time calculations. If 0.0, no smoothing is
  applied and raw dW/dt is used.

``min_rho_norm`` (float [default = 0.015])
  Minimum rho_norm value below which current profile values are extrapolated to
  the axis in psi calculations, to avoid numerical artifacts near rho=0.

``T_min_eV`` (float [default = 5.0])
  Minimum temperature threshold in eV. If the ion or electron temperature drops
  below this value anywhere in the domain, the simulation will stop with a
  ``LOW_TEMPERATURE_COLLAPSE`` error (error code 5). This is to catch radiative
  collapse or other unphysical low-temperature states early.

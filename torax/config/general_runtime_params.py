class NumericSettings(torax_pydantic.BaseModel):
  """Settings for numeric solvers and runtime operations.
  
  Attributes:
    t_initial: Initial time for the simulation [s].
    t_final: Final time for the simulation [s].
    exact_t_final: True to stop the simulation at exactly t_final, false to
      stop at the next dt step boundary.
    nref: Reference density [1e19 m^-3].
    fref: Reference parallel force density [N/m^3].
    resistivity_mult: Multiplicator for the resistivity, used to speed up
      simulation by shortening the current diffusion time [dimensionless].
    mindt: Minimum allowable dt value for time step calculator [s].
    fixed_dt: Fixed dt value for fixed time step calculator [s].
    maxdt: Maximum allowable dt value for time step calculator [s].
    dt_reduction_factor: Factor to reduce dt by between failed steps when
      using adaptive nonlinear solver.
    dtmult: Multiplier for chi-based time stepper [dimensionless].
    ptot_to_pthermal: Multiplier to convert total plasma pressure to thermal
      pressure. Used for L-H transition power calculation [dimensionless].
    vloop_lcfs_init: If loop voltage BC used for current diffusion equation,
      this is the initial value [V].
    ion_heat_eq: Solve the ion heat equation (ion temperature evolves over time).
    el_heat_eq: Solve the electron heat equation (electron temperature evolves
      over time).
    current_eq: Solve the current equation (psi evolves over time driven by
      the solver; q and s evolve over time as a function of psi).
    dens_eq: Solve the density equation (n evolves over time).
    use_bootstrap_calc: Compute the bootstrap current from the plasma pressure,
      density and temperature during the sim.
    use_vloop_lcfs_boundary_condition: For the current diffusion, imposes a
      loop voltage boundary condition at LCFS instead of boundary condition on
      total plasma current.
    adaptive_dt: Whether to use adaptive time stepping when a nonlinear step
      is not converging.
    progress_bar: Whether to show a progress bar.
    output_dir: Directory to output results, like state_history.nc.
    enable_sanity_checks: Whether to perform sanity checks during simulation
      to detect issues such as NaN values, negative temperatures, etc.
  """

  # Simulation times.
  t_initial: float = 0.0
  t_final: float = 5.0
  exact_t_final: bool = True

  # Reference values.
  nref: float = 1.0
  fref: float = 1.0
  resistivity_mult: float = 10.0
  mindt: float = 1e-4
  fixed_dt: float = 0.05
  # used for chi-based time_step_calculator.
  maxdt: float = 1.0
  dt_reduction_factor: float = 3.0
  # used for chi-based time_step_calculator
  dtmult: float = 10.0
  ptot_to_pthermal: float = 0.5
  vloop_lcfs_init: float = 0.0

  # What to evolve with transport equations
  ion_heat_eq: bool = True
  el_heat_eq: bool = True
  current_eq: bool = True
  dens_eq: bool = True

  # Source helpers.
  use_bootstrap_calc: bool = True
  use_vloop_lcfs_boundary_condition: bool = False

  # Nonlinear stepper handling.
  adaptive_dt: bool = True

  # Whether to show the progress bar during simulations
  progress_bar: bool = True

  # Directory to output results
  output_dir: str = '/tmp/torax_results'
  
  # Whether to perform sanity checks during simulation
  enable_sanity_checks: bool = False 
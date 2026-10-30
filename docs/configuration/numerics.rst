.. _numerics_dataclass:

numerics
########

Configures simulation control such as equations being solved and constant
numerical variables.

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
  ``transport.fast_ion_stabilization``, and to allow sources to provide fast ion
  profiles to the simulation.

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

.. _overview:

.. include:: links.rst

TORAX Overview
##############

TORAX is an auto-differentiable tokamak core transport simulator aimed for fast
and accurate forward modelling, pulse-design, trajectory optimization, and
controller design workflows. TORAX is written in Python-JAX, with the following
motivations:

  * Open-source and extensible, aiding with flexible workflow coupling.
  * JAX provides auto-differentiation capabilities and code compilation for fast
    runtimes. Differentiability allows for gradient-based nonlinear PDE solvers
    for fast and accurate modelling, and for sensitivity analysis of simulation
    results to arbitrary parameter inputs, enabling applications such as
    trajectory optimization and data-driven parameter identification for
    semi-empirical models. Auto-differentiability allows for these applications
    to be easily extended with the addition of new physics models including
    ML-surrogates, or new parameter inputs, by avoiding the need to hand-derive
    Jacobians
  * Python-JAX is a natural framework for the coupling of ML-surrogates of
    physics models.

TORAX, at v1.0.0, has the following physics and numerics feature set:

  * Coupled PDEs of ion and electron heat transport, electron particle
    transport, and current diffusion, solved numerically with:

    * Finite-volume-method spatial discretization
    * Multiple solver options for PDE time evolution

      * Linear with Pereverzev-Corrigan terms, and predictor-corrector steps for
        approximating the nonlinear solution.
      * Nonlinear solver with Newton-Raphson iterations, with line search and
        timestep backtracking.
      * Nonlinear solver with optimization using the jaxopt library.
      * Poloidal flux boundary conditions based on either total current or loop
        voltage at the last-closed-flux-surface.

  * Ohmic power, ion-electron heat exchange, fusion power, Bremsstrahlung,
    impurity line radiation, an ICRH ML-surrogate |toricnn| (as-yet covering
    limited regimes).
  * Bootstrap current and neoclassical conductivity with the analytical Sauter
    model.
  * Coupling to the |qlknn_7_11| and QLKNN10D |qlknn10d| QuaLiKiz-neural-network
    surrogates for physics-based turbulent transport. The semi-empirical
    Bohm-GyroBohm model is also available.
  * General geometry, provided via CHEASE, FBT, or EQDSK equilibrium files.
  * Sawtooth triggering and profile redistribution.
  * Simple pedestal models using a local adaptive source to set internal
    boundary conditions.
  * API for setting time-dependent boundary conditions, prescribed sources, and
    prescribed profiles optionally evolved by the PDE, from a variety of input
    data structures or user-provided analytical models.

    * For testing and demonstration purposes, a single CHEASE equilibrium file
      is available in the data/geo directory. It corresponds to an ITER hybrid
      scenario equilibrium based on simulations in |citrin2010| and was obtained
      from |pint|. A PINT license file is available in data/third_party/geo.

Model implementation was verified through direct comparison of simulation
outputs to the RAPTOR |raptor| tokamak transport simulator.

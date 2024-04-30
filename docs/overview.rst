.. _overview:

TORAX Overview
##############

TORAX is an auto-differentiable tokamak core transport simulator aimed for fast and
accurate forward modelling, pulse-design, trajectory optimization, and controller
design workflows. TORAX is written in Python-JAX, with the following motivations:

  * Open-source and extensible, aiding with flexible workflow coupling.
  * JAX provides auto-differentiation capabilities and code compilation for fast runtimes.
    Differentiability allows for gradient-based nonlinear PDE solvers for fast and accurate modelling,
    and for sensitivity analysis of simulation results to arbitrary parameter inputs, enabling applications
    such as trajectory optimization and data-driven parameter identification for semi-empirical models.
    Auto-differentiability allows for these applications to be easily extended with the addition of new
    physics models, or new parameter inputs, by avoiding the need to hand-derive Jacobians.
  * Python-JAX is a natural framework for the coupling of ML-surrogates of physics models.

TORAX's physics and solver feature set includes the following:

  * Coupled PDEs of ion and electron heat transport, electron particle transport, and current diffusion, solved
    numerically with:

    * Finite-volume-method spatial discretization
    * Multiple solver options for PDE time evolution

      * Linear with Pereverzev-Corrigan terms, and predictor-corrector steps for approximating the nonlinear solution
      * Nonlinear solver with Newton-Raphson iterations, with line search and timestep backtracking
      * Nonlinear solver with optimization using the jaxopt library

  * Ohmic power, ion-electron heat exchange, fusion power, bootstrap current with the analytical Sauter model
  * Time dependent boundary conditions and sources
  * Coupling to the QLKNN10D `[van de Plassche et al, Phys. Plasmas 2020] <https://doi.org/10.1063/1.5134126>`_
    QuaLiKiz-neural-network surrogate for physics-based turbulent transport
  * General geometry, provided via CHEASE equilibrium files

    * For testing and demonstration purposes, a single CHEASE equilibrium file is available in the
      data/geo directory. It corresponds to an ITER hybrid scenario equilibrium based on simulations
      in `[Citrin et al, Nucl. Fusion 2010] <https://doi.org/10.1088/0029-5515/50/11/115007>`_,
      and was obtained from `[PINT] <https://gitlab.com/qualikiz-group/pyntegrated_model>`_. A PINT
      license file is available in data/geo.

Additional heating and current drive sources can be provided by prescribed formulas, or user-provided analytical models.

Model implementation was verified through direct comparison of simulation outputs to the RAPTOR
`[Felici et al, Plasma Phys. Control. Fusion 2012] <https://iopscience.iop.org/article/10.1088/0741-3335/54/2/025002>`_
tokamak transport simulator.

.. include:: ../links.rst

.. _physics_plasma_composition:

Plasma composition, initial and prescribed conditions
#####################################################

Presently, TORAX accommodates a single main ion species and single impurity
species, which can be comprised of time-dependent mixtures of ions with
fractional abundances summing to 1. This is useful for example for simulating
isotope mixes. Based on the ion symbols and fractional abundances, the average
mass of each species is determined. The average charge state of each ion in each
mixture is determined by Mavrin polynomial fits to ADAS atomic data |mavrin|.
For the temperature ranges of interest in the tokamak core, these are well
approximated as 1D functions of electron temperature. All ions with atomic
numbers below Carbon are assumed to be fully ionized.

The impurity and main ion densities are constrained by the plasma effective
charge, :math:`Z_\mathrm{eff}`, which is a user-provided 2D array in both time
and space, as well as quasineutrality.

:math:`n_i`, and :math:`n_{imp}`, are solved from the
following system of equations, where :math:`Z_\mathrm{eff}` and the electron
density are known, and :math:`Z_\mathrm{imp}` is the average impurity charge of
the impurity mixture, with the average charge state for each ion determined from
the Mavrin polynomials.

.. math::

  n_\mathrm{i}Z_\mathrm{i}^2 + n_\mathrm{imp}Z_\mathrm{imp}^2 =
  n_\mathrm{e}Z_\mathrm{eff}

  n_\mathrm{i}Z_\mathrm{i} + n_\mathrm{imp}Z_\mathrm{imp} = n_\mathrm{e}

Initial conditions for the evolving profiles :math:`T_i`, :math:`T_e`,
:math:`n_e`, and :math:`\psi` are user-configurable.

Additionally, if any of the :math:`T_{i,e}`, :math:`n_e` equations are set to be
non-evolving (i.e., not evolved by the PDE solver), then time-dependent
prescribed profiles are user-configurable.

For the poloidal flux :math:`\psi(\hat{\rho})`, the user specifies if the
initial condition is based on a prescribed current profile,
:math:`j_\mathrm{tor}=j_0(1-\hat{\rho}^2)^\nu` (with :math:`j_0`
scaled to match :math:`I_p`, and :math:`\nu` is user-configurable), or from the
:math:`\psi` provided in a geometry file. The prescribed current profile
option is always used for the circular geometry. The total current :math:`I_p`
can be user-configured or determined by the geometry data. In the latter
case, the :math:`\psi` provided by the geometry data can still be used, but is
scaled by the ratio of the desired :math:`I_p` and the geometry-provided
:math:`I_p`.

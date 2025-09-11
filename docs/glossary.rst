.. _glossary:

Glossary of Terms
#################

:math:`T_i`: Ion temperature

:math:`T_e`: Electron temperature

:math:`n_e`: electron density

:math:`\psi`: poloidal flux

:math:`\hat{\rho}`: The normalized toroidal flux coordinate. used as the radial
coordinate in the TORAX 1D mesh.

:math:`\hat{\rho}` is a flux surface label, being constant on a closed surface
of magnetic flux. The toroidal flux coordinate is defined as
:math:`\rho=\sqrt{\frac{\Phi}{\pi B}}`, where :math:`\Phi` is the toroidal
magnetic flux enclosed by the magnetic flux surface being labelled. :math:`B` is
the magnetic field at the magnetic axis.
The :math:`\hat{\rho}=\rho/\rho_{LCFS}`, where :math:`\rho_{LCFS}` is the
toroidal flux coordinate at the last-closed-flux-surface, in units of meters.

**evolving profiles**: ``core_profile`` variables
(:math:`T_i`, :math:`T_e`, :math:`n_e`, :math:`\psi`) which are evolved by the
PDE, as determined in the **numerics** configuration (see :ref:`configuration`).

**prescribed profiles**: ``core_profile`` variables that are not being evolved
by the PDE, but have time-dependent variables in the **profile_conditions**
configuration (see :ref:`configuration`) that determine their values
throughout the simulation.

**runtime parameters**: configuration variables that are inputs to the
simulation. Some of these are time-dependent, and some are not.

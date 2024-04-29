.. _glossary:

Glossary of Terms
#################

:math:`T_i`: Ion temperature

:math:`T_e`: Electron temperature

:math:`n_e`: electron density

:math:`\psi`: poloidal flux

:math:`\hat{\rho}`: The normalized toroidal flux coordinate. used as the radial coordinate in the TORAX 1D mesh.

**evolving profiles**: ``core_profile`` variables (:math:`T_i`, :math:`T_e`, :math:`n_e`, :math:`\psi`)
which are evolved by the PDE, as determined in the **numerics** dataclass (see :ref:`configuration`).

**prescribed profiles**: ``core_profile`` variables that are not being evolved by the PDE, but have time-dependent
variables in the **profile_conditions** dataclass (see :ref:`configuration`) that determine their values
throughout the simulation.


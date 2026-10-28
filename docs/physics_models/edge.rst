.. include:: ../links.rst

.. _physics_edge:

Edge models
###########

TORAX supports coupling to reduced edge/divertor models to provide physically
consistent boundary conditions for the core transport solver. Currently, only
the Extended Lengyel model is supported.

Extended Lengyel Model
======================
The Extended Lengyel model describes the 1D parallel heat and particle transport
in the Scrape-Off Layer (SOL) and divertor. It extends the classic Lengyel model
by including cross-field transport in the divertor, power and momentum loss due
to neutral ionization close to the divertor target and turbulent broadening of
the upstream heat flux channel. The implementation follows |body2025|.

The model relates upstream quantities (e.g. power crossing separatrix
:math:`P_{SOL}`, separatrix density :math:`n_{e,sep}`) and divertor impurity
content, to downstream quantities such as target temperature :math:`T_{e,t}`.
The separatrix temperature is calculated from a 2-point model.

**Modes of Operation:**

*   **Forward Mode**: Given the impurity mix and upstream conditions, calculate
    the target temperature :math:`T_{e,t}`.

*   **Inverse Mode**: Given a desired target temperature :math:`T_{e,t}`
    (e.g., 5 eV to represent detachment onset), calculate the required
    concentration of a given mix of seeded impurity species.

**Physics Features:**

*   **Impurity Radiation**: Radiative cooling rates :math:`L_z(T_e)` are
    calculated using polynomial fits from |mavrin2017|, which are valid at low
    edge temperatures (down to ~1 eV).

*   **Enrichment**: Impurity enrichment (:math:`c_{div}/c_{core}`) can be
    specified manually or calculated using the empirical scaling from
    |kallenbach2024|:
    :math:`E \propto Z^{-0.5} p_0^{-0.4} (E_{ion,z}/E_{ion,D})^{-5.8}`.

**Coupling to Core:**

When enabled, the edge model updates the following TORAX runtime parameters for
the next time step. This is done via explicit coupling: the parameters at time
:math:`t` are passed to the edge model, which calculates the updated parameters
at time :math:`t+\Delta t`.

1.  **Boundary Conditions**: The separatrix electron temperature calculated by
    the model sets the core :math:`T_e` boundary condition. :math:`T_i` boundary
    condition is set via a configured ratio.

2.  **Impurity Composition**:

    *   In **Inverse Mode**, the required seeded impurity concentration updates
        the core impurity profile (scaled by enrichment factor).

    *   The model also enforces consistency for fixed background impurities
        between core and edge.

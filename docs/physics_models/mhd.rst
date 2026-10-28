.. include:: ../links.rst

.. _physics_mhd:

MHD models
##########

Currently only a sawtooth model is implemented, although the TORAX config and
internal APIs are designed to accommodate other models in the future.

Sawtooth Model
==============

Sawteeth are periodic oscillations in the core plasma temperature, density, and
current caused by the growth and subsequent rapid reconnection of an m=1, n=1
kink instability within the plasma volume where the safety factor, :math:`q`,
drops below unity. The sawtooth crash is triggered by a state-dependent critical
magnetic shear at the :math:`q=1` surface.

The TORAX sawtooth model comprises two components:

* Trigger Model: determines the conditions under which a sawtooth crash is
  initiated.

* Redistribution Model: Modifies the plasma profiles (temperature, density,
  poloidal flux) following a crash to simulate the rapid transport event.

Currently only simple Trigger and Redistribution models are implemented.

The ``simple`` trigger model checks for the following conditions at each time
step:

1.  **Existence of a q=1 surface:** The safety factor profile `q` must drop
    below 1.

2.  **Minimum radius:** The normalized radius of the q=1 surface
    (:math:`\hat{\rho}_{q=1}`) must be greater than a specified minimum value
    (``minimum_radius``). This prevents spurious triggers very close to the
    magnetic axis.

3.  **Critical magnetic shear:** The magnetic shear (:math:`\hat{s}`) at the
    :math:`q=1` surface must exceed a critical value (``s_critical``).

The ``simple`` redistribution model mplements a simplified redistribution
process based on conserving particle number, energy, and current within a
mixing radius.

1.  **Mixing Radius Calculation:** The mixing radius (:math:`\hat{\rho}_{mix}`)
    is calculated as ``mixing_radius_multiplier`` * :math:`\hat{\rho}_{q=1}`.

2.  **Profile Flattening:** Profiles
    (:math:`T_i`, :math:`T_e`, :math:`n_e`, :math:`j_{tot}`) within the mixing
    radius are partially flattened.

    *   Inside the q=1 surface (:math:`\hat{\rho} < \hat{\rho}_{q=1}`), a
        linear profile is created, where the value at :math:`\hat{\rho}=0` is
        set to a multiple ``flattening_factor`` of the value at
        :math:`\hat{\rho}_{q=1}`.

    *   Between the :math:`q=1` surface and the mixing radius
        (:math:`\hat{\rho}_{q=1} \le \hat{\rho} < \hat{\rho}_{mix}`),
        the profile is linearly interpolated between the flattened value at
        :math:`\hat{\rho}_{q=1}` and the original profile value at
        :math:`\hat{\rho}_{mix}`.

    The values at :math:`\hat{\rho}_{q=1}` are set by conservation laws.

3.  **Conservation:** The flattening process conserves the total number of
    particles (:math:`\int n_e dV`), total electron and ion thermal energy
    (:math:`\int \frac{3}{2} n_{e,i} T_{e,i} dV`), and total current
    (:math:`\int j_{tot} dS`) within the mixing radius.
    The values at :math:`\hat{\rho}_{q=1}` after flattening are adjusted to
    ensure these conservation laws are met. The poloidal flux profile is then
    recalculated to be consistent with the flattened current profile, and the
    pre-crash poloidal flux boundary condition.

4.  **Derived Quantities:** After redistribution, derived quantities like
    ion densities (:math:`n_i`, :math:`n_{imp}`), impurity charge states
    (:math:`Z_{imp}`), safety factor (:math:`q`), magnetic shear
    (:math:`\hat{s}`), and sources, are recalculated based on the modified
    profiles.

Simulation Step During Crash
----------------------------

When a sawtooth crash is triggered:

1.  The standard PDE time step is **skipped**.

2.  The redistribution model is applied to modify the ``core_profiles``.

3.  A short, fixed time step (user-configurable ``crash_step_duration``) is
    taken.

4.  Derived quantities are recalculated based on the redistributed profiles.

The simulation then always proceeds to the next regular PDE time step.
Subsequent sawtooth crashes are not allowed, to prevent continuous consecutive
crashes if the trigger condition is still met immediately after redistribution.

See the :ref:`sawtooth configuration details <sawtooth_config>` for a summary
of the input parameters.

The current sawtooth model is a simplified representation. Future development
plans include: Implementing more sophisticated trigger models
(e.g., based on the Porcelli model). Developing more physically detailed
redistribution models (e.g., Kadomtsev reconnection models).

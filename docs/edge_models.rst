.. include:: links.rst

.. _edge_models:

Edge Model Configuration
########################

This guide details the configuration and usage of the edge physics models in
TORAX. For a brief overview of the underlying physics, refer to the
:ref:`physics_models` section. An overview of all configurable model inputs is
provided below at :ref:`extended_lengyel_config`. We first provide some general
guidance and context on the model usage.

.. _extended_lengyel_usage:

Extended Lengyel Model Usage
============================

This model provides self-consistent boundary conditions for the core transport
solver and can calculate required impurity seeding rates to achieve target
conditions.

Computation Modes
-----------------

The model operates in two distinct modes, controlled by the ``computation_mode``
parameter. In ``forward`` mode, the target temperature is an emergent property,
resulting from the plasma conditions, geometry, and impurity mix. In ``inverse``
mode, the target temperature is constrained, and the required seeded impurity
mix to achieve this value is calculated, thus providing feedback on the core
impurity content. Each mode has its specific use-cases depending on user
preferences and specific simulation goals. Note that in both modes, the
separatrix temperature is calculated and can update the core boundary
conditions.


Forward Mode
^^^^^^^^^^^^
Set ``computation_mode`` to ``'forward'``.

*   **Goal:** Calculate the resulting target electron temperature
    (:math:`T_{e,t}`) and core boundary conditions based on the current
    plasma state and impurity mix.

*   **Key Inputs:** Uses ``fixed_impurity_concentrations`` to determine the
    impurity mix at the edge.

*   **Key Outputs:** Updates the core :math:`T_e` and :math:`T_i` boundary
    conditions at the LCFS. ``T_e_target`` is provided as an output.

Inverse Mode
^^^^^^^^^^^^
Set ``computation_mode`` to ``'inverse'``.

*   **Goal:** Determine the necessary seeded impurity concentration to achieve a
    specific target temperature (e.g., to maintain detachment and/or to avoid
    tungsten sputtering).

*   **Key Inputs:** Requires ``T_e_target`` (desired temperature in eV) and
    ``seed_impurity_weights`` (relative mix of seeded species).

*   **Key Outputs:** Updates core temperature boundary conditions AND updates
    the core impurity density profile for the seeded species to match the
    required concentration.

Impurity Handling
-----------------

Impurity Content
^^^^^^^^^^^^^^^^

*   **Fixed Impurities:** Background species (e.g., Helium ash, intrinsic
    Tungsten) defined via ``fixed_impurity_concentrations``. This can be
    provided in both forward and inverse modes. In inverse mode, the model does
    not update these values to match the target temperature constraint. The
    values are fixed.

*   **Seeded Impurities:** Species actively injected for control
    (e.g., Neon, Argon) defined via ``seed_impurity_weights``. This is only used
    in **Inverse Mode**. The values define the relative weight of the seeded
    impurities, and the model will calculate the absolute concentrations such
    that the target temperature is achieved.

Source of Truth (``impurity_sot``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For fixed impurities, you must define whether the core or the edge configuration
represents the "Source of Truth".

*   **Core ('core')**: Values set in the ``plasma_composition`` config
    defines the inventory. Edge concentrations are derived from the core LCFS
    values using the enrichment factor. This is the default.

*   **Edge ('edge')**: The edge config defines the concentration. The core
    profiles are scaled using the enrichment factor to match.

Enrichment
^^^^^^^^^^
The model accounts for divertor retention via the enrichment factor
:math:`E = c_{div}/c_{core}`.

*   **Automatic:** Set ``use_enrichment_model`` to ``True`` to calculate
    :math:`E` dynamically using the Kallenbach 2024 scaling |kallenbach2024|.
    For limited configurations, model output will be 1.0 for each species.

*   **Manual:** Set ``use_enrichment_model`` to ``False`` and provide specific
    values in the ``enrichment_factor`` dictionary.

Geometry Configuration
----------------------
The model requires specific edge geometric parameters (connection lengths,
flux expansion, etc.).

*   **FBT Geometry:** If using FBT files, TORAX attempts to extract these values
    directly from the file. If values are found in both the geometry file and
    the edge configuration, the values in the file take precedence.

*   **Other Geometries:** For CHEASE, EQDSK, or IMAS, you **must** explicitly
    provide these parameters in the ``edge`` configuration
    (e.g., ``connection_length_target``, ``diverted``).

Numerical Solver
----------------
The edge model solves a system of non-linear equations. The ``solver_mode`` can
be adjusted if convergence issues arise:

*   ``'hybrid'`` (Recommended): Uses a fixed-point iteration with default 5
    steps to warm-start a Newton-Raphson solver.

*   ``'newton_raphson'``: Standard Newton-Raphson solver.

*   ``'fixed_point'``: Simple iterative update. Robust but may be slower to
    converge.


.. _extended_lengyel_config:

Extended Lengyel Configuration Parameters
=========================================

A detailed description of all configurable parameters in the extended lengyel
model is found below:

Control Parameters
------------------

``computation_mode`` (str [default = 'forward'])
  The computation mode: ``'forward'`` or ``'inverse'``.

``solver_mode`` (str [default = 'hybrid'])
  Numerical solver strategy: ``'fixed_point'``, ``'newton_raphson'``, or
  ``'hybrid'``.

``impurity_sot`` (str [default = 'core'])
  Source of Truth (SOT) for fixed impurities.
  * ``'core'``: Core profiles define the concentration.
  * ``'edge'``: Edge config defines the concentration.

``update_temperatures`` (bool [default = True])
  If True, update core temperature boundary conditions based on edge model
  results.

``update_impurities`` (bool [default = True])
  If True, update core impurity profiles in ``inverse`` mode based on edge model
  results.

``fixed_point_iterations`` (int | None [default = None])
  Number of iterations for the fixed-point solver. If None, defaults to 25
  (fixed-point mode) or 5 (hybrid mode).

``newton_raphson_iterations`` (int [default = 30])
  Maximum number of iterations for the Newton-Raphson solver.

``newton_raphson_tol`` (float [default = 1e-5])
  Convergence tolerance for the Newton-Raphson solver.

Physical Parameters
-------------------

``T_e_target`` (**time-varying-scalar** | None [default = None])
  Desired target electron temperature [eV]. Required if
  ``computation_mode='inverse'``. Must be ``None`` in forward mode.

``seed_impurity_weights`` (dict[str, **time-varying-scalar**] | None [default = None])
  Relative weights of seeded impurity species (e.g., ``{'N': 0.95, 'Ar': 0.05}``).
  Required if ``computation_mode='inverse'``. Must be ``None`` in forward mode.

``fixed_impurity_concentrations`` (dict[str, **time-varying-scalar**] [default = {}])
  Fixed background impurity concentrations at the edge (:math:`n_z/n_e`).

``enrichment_factor`` (dict[str, **time-varying-scalar**] | None [default = None])
  Enrichment factor (:math:`c_{div}/c_{core}`) for each species. Required if
  ``use_enrichment_model=False``.

``use_enrichment_model`` (bool [default = True])
  If True, calculate enrichment factors using the Kallenbach 2024 model. In
  limited configurations, model output will be 1.0 for each species.

``enrichment_model_multiplier`` (**time-varying-scalar** [default = 1.0])
  Multiplier applied to the calculated Kallenbach 2024 enrichment factor.

``ne_tau`` (**time-varying-scalar** [default = 0.5e17])
  The non-coronal parameter (:math:`n_e \tau_{res}`) [:math:`\mathrm{s~m^{-3}}`]
  used in impurity radiation calculations.

``sheath_heat_transmission_factor`` (**time-varying-scalar** [default = 8.0])
  Sheath heat transmission coefficient (:math:`\gamma`).

``fraction_of_P_SOL_to_divertor`` (**time-varying-scalar** [default = 0.6666])
  Fraction of SOL power entering the divertor leg.

``SOL_conduction_fraction`` (**time-varying-scalar** [default = 1.0])
  Fraction of SOL power carried by conduction.

``ratio_of_molecular_to_ion_mass`` (**time-varying-scalar** [default = 2.0])
  Ratio of molecular to ion mass, used in neutral pressure calculations.

``T_wall`` (**time-varying-scalar** [default = 300.0])
  Divertor wall temperature [K].

``mach_separatrix`` (**time-varying-scalar** [default = 0.0])
  Mach number at the separatrix.

``mach_target`` (**time-varying-scalar** [default = 1.0])
  Mach number at the target (Bohm condition).

``T_i_T_e_ratio_separatrix`` (**time-varying-scalar** [default = 1.0])
  Ratio of ion to electron temperature at the separatrix.

``T_i_T_e_ratio_target`` (**time-varying-scalar** [default = 1.0])
  Ratio of ion to electron temperature at the target.

``n_e_n_i_ratio_separatrix`` (**time-varying-scalar** [default = 1.0])
  Ratio of electron to ion density at the separatrix.

``n_e_n_i_ratio_target`` (**time-varying-scalar** [default = 1.0])
  Ratio of electron to ion density at the target.

Geometry Parameters
-------------------

These must be provided if not using FBT geometry, or if using an FBT geometry
provider that does not include edge data.

``diverted`` (**time-varying-scalar** | None [default = None])
  Boolean flag indicating if the geometry is diverted.

``connection_length_target`` (**time-varying-scalar** | None [default = None])
  Parallel connection length from outboard midplane to target [m].

``connection_length_divertor`` (**time-varying-scalar** | None [default = None])
  Parallel connection length from outboard midplane to X-point [m].

``angle_of_incidence_target`` (**time-varying-scalar** | None [default = None])
  Angle between magnetic field line and divertor target [degrees].

``toroidal_flux_expansion`` (**time-varying-scalar** | None [default = None])
  Toroidal flux expansion factor, related to the ratio of target to upstream
  radii.

``ratio_bpol_omp_to_bpol_avg`` (**time-varying-scalar** | None [default = None])
  Ratio of poloidal field at outboard midplane to the separatrix average.

``divertor_broadening_factor`` (**time-varying-scalar** [default = 3.0])
  Ratio of the wetted area broadening (:math:`\lambda_{int}/\lambda_q`) in the
  divertor. Set to 1.0 in limited configurations.

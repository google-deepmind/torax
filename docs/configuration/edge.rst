.. _edge_models:
.. _extended_lengyel:

edge
####

Configures the edge physics model used to determine boundary conditions for the
core transport solver. If not provided, no edge model is run, and boundary
conditions are determined solely by the :ref:`profile_conditions <profile_conditions>`
settings.

For a brief overview of the underlying physics, refer to the
:ref:`physics_edge` section. An overview of all configurable model inputs is
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

Forward Mode: Multistart and Multiple Roots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In Forward Mode, the extended Lengyel model can exhibit multiple valid solutions
(roots) for the same input parameters (e.g., a high-temperature attached
solution and a low-temperature detached solution), particularly at lower
densities less than :math:`10^{20} \mathrm{m^{-3}}`.

To address this, the solver employs a **multistart strategy**:

*   **Parallel Execution:** The solver runs multiple instances in parallel
    starting from a grid of initial guesses (spanning low to high temperatures)
    plus the nominal guess (either default, user-provided, or from the previous
    time-step).

*   **Nominal Solution Selection:** The model selects the *closest valid root*
    to the nominal initial guess as the primary output. This ensures solution
    continuity and stability during time-dependent simulations.

*   **Multiple Roots Detection:** If multiple distinct valid roots are found,
    the ``multiple_roots_found`` flag is set to ``True``.

*   **Full Output:** The full set of found roots (from all initial guesses) is
    available in the ``roots`` child node of the model's output structure (e.g.
    ``output.edge.children['roots']``) for detailed analysis. These only
    include distinct roots, and for which the numerics converged (i.e.
    ``numerics_outcome.error == 0``).

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

Core Profile Rescaling and LCFS Boundary Requirement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When the edge model updates the core impurity inventory (either for seeded
impurities, or for fixed impurities when ``impurity_sot='edge'``), it rescales
the user-provided core impurity density profile (``n_e_ratios``) to match the
edge-determined concentration at the LCFS (reduced by the enrichment factor).

This rescaling preserves the *shape* of the user-provided profile. The scaling
factor is calculated by dividing the edge-determined concentration at the LCFS
by the user-provided profile value at the LCFS (rho_norm=1.0):

.. math::

  \text{scaling\_factor} = \frac{c_{LCFS, edge}}{c_{LCFS, core\_provided}}

Because of this division, **the user-provided core impurity profile must have a
strictly positive value at the LCFS (rho_norm=1.0)**. If the user-provided
profile value is zero at the LCFS, the scaling has no effect (the profile
remains zero everywhere), which silently breaks the edge-core coupling.

To prevent this, ``ToraxConfig`` performs a validation at configuration time.
If an edge model is active and any impurity species has a zero (or negative)
value at ``rho_norm=1.0`` in the configuration, a ``ValueError`` will be raised.
To resolve this, please set a small positive value at ``rho_norm=1.0`` (e.g.,
``1e-5`` or similar) in the core profile config; the actual boundary value will
be overwritten by the edge model during execution.

Geometry Configuration
----------------------
The model requires specific edge geometric parameters (connection lengths,
flux expansion, etc.).

*   **MEQ Geometry:** If using MEQ files, TORAX attempts to extract these values
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


.. _edge_config:

Base Edge Configuration Parameters
==================================

An edge model can be configured either as a single model (such as
``'extended_lengyel'``) or as a ``'combined'`` model containing a dictionary of
``sub_models``. In a combined model, each boundary condition is provided by
whichever sub-model has the corresponding ``update_*`` flag enabled, and at most
one sub-model may have a given ``update_*`` flag enabled across the simulation.

All edge model configurations inherit from ``EdgeModelConfig`` and support
the following common configuration parameters. Each model declares, via its
class-level ``supported_boundary_conditions``, which boundary conditions it can
compute. Each ``update_*`` flag defaults to the corresponding
``supported_boundary_conditions`` value, and cannot be set to ``True`` for a
boundary condition the model does not support.

.. torax_config:: torax._src.edge.extended_lengyel.config.ExtendedLengyelConfig
   :include-fields: model_name

.. torax_config:: torax._src.edge.base.EdgeModelConfig
   :exclude-fields: model_name


.. _extended_lengyel_config:

Extended Lengyel Configuration Parameters
=========================================

A detailed description of all configurable parameters in the extended Lengyel
model is found below:

Control Parameters
------------------

.. torax_config:: torax._src.edge.extended_lengyel.config.ExtendedLengyelConfig
   :include-fields: computation_mode, solver_mode, impurity_sot, fixed_point_iterations, fixed_point_relaxation_factor, newton_raphson_iterations, newton_raphson_tol, multistart_num_guesses, initial_guess

Physical Parameters
-------------------

.. torax_config:: torax._src.edge.extended_lengyel.config.ExtendedLengyelConfig
   :include-fields: T_e_target, seed_impurity_weights, fixed_impurity_concentrations, enrichment_factor, use_enrichment_model, enrichment_model_multiplier, n_e_tau, sheath_heat_transmission_factor, fraction_of_P_SOL_to_divertor, SOL_conduction_fraction, ratio_of_molecular_to_ion_mass, T_wall, mach_separatrix, mach_target, T_i_T_e_ratio_separatrix, T_i_T_e_ratio_target, n_e_n_i_ratio_separatrix, n_e_n_i_ratio_target

Geometry Parameters
-------------------

These must be provided if not using MEQ geometry, or if using an MEQ geometry
provider that does not include edge data. Even when using MEQ geometry, it is
recommended to include fallback values in case the MEQ provider does not succeed
in extracting the required data for specific geometries.

.. torax_config:: torax._src.edge.extended_lengyel.config.ExtendedLengyelConfig
   :include-fields: diverted, connection_length_target, connection_length_divertor, angle_of_incidence_target, toroidal_flux_expansion, ratio_bpol_omp_to_bpol_avg, divertor_broadening_factor

Initial Guess Parameters
------------------------

These parameters (configured inside the ``initial_guess`` dictionary) control
the initialization of the iterative solver. By default, the model attempts to
use the converged solution from the previous time step to warm-start the solver
(``use_previous_step_as_guess=True``), improving convergence speed and stability
in time-dependent simulations. Provided initial guess values are always used for
the first time step.

**Note**: If the previous solver had a bad outcome (Newton-Raphson did not
converge, or a physics error occurred such as negative heat flux), the previous
outputs are automatically skipped and the solver falls back to the configured
initial guess or defaults. This prevents the solver from inheriting a bad state
that can lead to cold root convergence in subsequent time steps.

.. torax_config:: torax._src.edge.extended_lengyel.config.InitialGuessConfig


.. _custom_edge_models:

Custom Edge Models
==================

TORAX provides a modular edge model interface allowing users to implement and
register custom edge physics models. For instructions on implementing and
registering custom edge models via ``torax.experimental.edge.register_edge_model``, see
:ref:`model-integration`.

.. _migration_guide:

Migrating from TORAX v1 to v2
#############################

This guide covers breaking changes and behavioral updates when upgrading from
TORAX v1 (``1.0.x``–``1.4.x``) to TORAX v2 (``2.0.0``). TORAX v2 delivers
improved numerical performance and accuracy alongside standardized naming
conventions and better Pydantic validation.

.. _migration_checklist:

Quick upgrade checklist
=======================

Most existing workflows require updates in the following areas:

* **Environment**: Upgrade to Python ``>= 3.12`` and ``jax >= 0.11.1`` (see
  :ref:`migration_dependencies`).
* **Programmatic execution**: ``torax.run_simulation(torax_config)`` now runs
  JIT-compiled by default and returns ``state_history`` directly instead of
  ``(data_tree, state_history)``. Call
  ``state_history.simulation_output_to_xr()`` to obtain the ``xarray.DataTree``
  (see :ref:`migration_execution`).
* **Strict lowercase configuration**: Unknown or deprecated v1 keys raise
  ``pydantic.ValidationError``. String enum values and ``model_name``
  identifiers now use ``lower_snake_case`` (e.g. ``'zero'``, ``'prescribed'``,
  ``'bohm_gyrobohm'``; see :ref:`migration_global_config`).
* **Timestep settings**: All timestep settings have moved from ``numerics`` to
  ``time_step_calculator``, and ``max_steps`` is required when using the
  ``'chi'`` (default) or ``'from_previous_dt'`` calculator (see
  :ref:`migration_numerics`).
* **Profile conditions**: ``profile_conditions['psi_source']`` (``'j'``,
  ``'profile_conditions'``, or ``'geometry'``) is now required, ``Ip`` is
  renamed to ``I_p``, and ``nbar`` / ``fGW`` keys are standardized to
  ``n_e_bar`` / ``f_GW`` (see :ref:`migration_profile_conditions`).
* **Impurity, transport, pedestal, and sources**:
  ``plasma_composition['impurity']`` requires an explicit dictionary with
  ``'impurity_mode'`` and ``'species'``; ``transport`` uses named
  ``core_transport_models`` dictionaries; ``pedestal`` uses nested
  ``structure_model`` (with ``*_top`` keys) and ``formation_model`` blocks; and
  active ``sources`` require explicit parameters instead of ``{}`` (see
  :ref:`migration_plasma_composition`, :ref:`migration_transport`,
  :ref:`migration_pedestal`, and :ref:`migration_sources`).
* **Outputs, plotting, and restarts**: Update post-processing scripts and custom
  plot configs for renamed ``xr.DataTree`` variables (such as ``scalars.I_p``,
  ``scalars.P_generic_heat_*``, ``scalars.P_impurity_radiation_e``,
  ``profiles.I_p_profile``, ``profiles.j_toroidal_*``, and
  ``profiles.p_ei_exchange_*``) and per-model turbulent transport outputs under
  ``auxiliary/turbulent_transport/`` (with ``'output_mode_contributions': True``
  needed on ``'qlknn'`` for ITG/TEM/ETG mode splits; see
  :ref:`migration_outputs`). Regenerate saved v1 NetCDF checkpoint files before
  using ``restart`` in v2 (see
  :ref:`restart <migration_solver_edge_neoclassical>`).

.. _migration_dependencies:

Python and dependency requirements
==================================

TORAX v2 updates minimum runtime dependencies:

* **Python**: ``>= 3.12`` (Python 3.11 support has been dropped).
* **JAX**: ``jax >= 0.11.1`` and ``jaxlib >= 0.11.1``.

See :ref:`installation` for instructions on creating a fresh virtual environment
for TORAX v2.

.. _migration_execution:

Running simulations and Python / JAX API
========================================

``torax.run_simulation`` and JIT execution
------------------------------------------

* **Return value**: ``torax.run_simulation`` now returns a single
  ``torax.StateHistory`` object rather than a ``(data_tree, state_history)``
  tuple. Convert to an ``xarray.DataTree`` outside ``jax.jit`` using
  ``state_history.simulation_output_to_xr()``. ``simulation_output_to_xr()`` no
  longer accepts a ``file_restart`` argument and stitches the prior state file
  automatically when ``torax_config.restart.stitch=True``.
* **JIT-compiled by default**: ``torax.run_simulation`` executes the
  JIT-compiled time-stepping loop by default.
  ``torax.experimental.run_simulation_jitted`` has been removed; call
  ``torax.run_simulation`` directly, or use ``torax.experimental.run_loop_jit``
  for lower-level step-function execution with runtime parameter overrides (see
  :ref:`jitted_simulation`).
* **Bounded step count** (``max_steps``): To bound the pre-allocated output
  buffer in the compiled loop, ``max_steps`` **must** be specified (via
  ``time_step_calculator['max_steps']``, ``torax.run_simulation(...,
  max_steps=...)``, or ``run_torax --max_steps``) whenever ``calculator_type``
  is ``'chi'`` (the default) or ``'from_previous_dt'``; only
  ``calculator_type='fixed'`` computes a default automatically. If
  ``max_steps`` is reached before ``t_final``, ``state_history.sim_error`` is
  set to ``torax.SimError.DID_NOT_REACH_T_FINAL`` (code ``6``).
* **In-place config updates** (``update_fields``): When mutating frozen
  ``ToraxConfig`` instances across parameter sweeps via
  ``torax_config.update_fields({...})``, update dotted field paths to match the
  v2 schema (for example, ``'time_step_calculator.dt'`` or
  ``'profile_conditions.I_p'``).

.. tab-set::

   .. tab-item:: TORAX v1

      .. code-block:: python

         import torax

         torax_config = torax.ToraxConfig.from_dict(config_dict)
         data_tree, state_history = torax.run_simulation(torax_config)
         if state_history.sim_error != torax.SimError.NO_ERROR:
           raise RuntimeError(f'Simulation failed: {state_history.sim_error}')

   .. tab-item:: TORAX v2

      .. code-block:: python

         import torax

         torax_config = torax.ToraxConfig.from_dict(config_dict)
         state_history = torax.run_simulation(torax_config)
         if state_history.sim_error != torax.SimError.NO_ERROR:
           raise RuntimeError(f'Simulation failed: {state_history.sim_error}')
         data_tree = state_history.simulation_output_to_xr()

See :ref:`running_programmatically` for full examples of running, saving, and
plotting simulations from Python.

Top-level ``torax`` and subpackage imports
------------------------------------------

Several symbols previously exposed on the top-level ``torax`` namespace or
internal modules have moved into dedicated public subpackages:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - TORAX v1 symbol
     - TORAX v2 replacement
   * - ``torax.JAX_STATIC``
     - ``torax.torax_pydantic.JAX_STATIC``
   * - ``torax.PedestalModelOutput``
     - ``torax.pedestal.PedestalStructure``
   * - ``torax.SourceProfiles``
     - ``torax.sources.CoreSources`` (``state_history.source_profiles`` is now
       ``state_history.core_sources``, with per-source ``SourceOutput`` entries
       in ``core_sources.sources``)
   * - ``torax_pydantic.BaseModelFrozen``
     - ``torax.torax_pydantic.ConfigBase``
   * - ``torax.experimental.run_simulation_jitted``
     - ``torax.run_simulation`` (or ``torax.experimental.run_loop_jit``)

TORAX v2 exposes the public subpackages ``torax.constants``,
``torax.experimental``, ``torax.math_utils``, ``torax.output``,
``torax.pedestal``, ``torax.sources``, ``torax.torax_pydantic``, and
``torax.transport``, as well as ``torax.SolverProfiles`` at the top level.

``CoreProfiles`` nested container structure
-------------------------------------------

If you inspect ``state_history.core_profiles`` or author custom physics models,
note that ``torax.CoreProfiles`` inherits from ``torax.SolverProfiles`` and
groups plasma profiles into nested sub-containers instead of flat attributes.
Density profiles ``n_e``, ``n_i``, and ``n_impurity`` are stored in SI units of
:math:`\mathrm{m^{-3}}`:

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - TORAX v1 attribute
     - TORAX v2 attribute
   * - ``core_profiles.T_i``, ``T_e``, ``n_e``, ``n_i``, ``n_impurity``
     - ``core_profiles.kinetic.T_i``, ``T_e``, ``n_e``, ``n_i``, ``n_impurity``
   * - ``core_profiles.toroidal_angular_velocity``
     - ``core_profiles.kinetic.toroidal_angular_velocity``
   * - ``core_profiles.Z_i``, ``Z_i_face``, ``Z_impurity``, ``Z_impurity_face``,
       ``A_i``, ``A_impurity``, ``A_impurity_face``, ``Z_eff``, ``Z_eff_face``
     - ``core_profiles.ions.Z_i``, ``Z_i_face``, ``Z_impurity``,
       ``Z_impurity_face``, ``A_i``, ``A_impurity``, ``A_impurity_face``,
       ``Z_eff``, ``Z_eff_face``
   * - ``core_profiles.main_ion_fractions``, ``impurity_fractions``
     - ``core_profiles.ions.main_ion_fractions``, ``impurity_fractions``
   * - ``core_profiles.charge_state_info``, ``charge_state_info_face``
     - Flattened onto ``core_profiles.ions`` as ``Z_avg``, ``Z_avg_face``,
       ``Z_impurity_per_species``, and ``Z_impurity_per_species_face``
   * - ``core_profiles.psi``, ``q_face``, ``s_face``
     - ``core_profiles.poloidal_flux.psi``, ``q_face``, ``s_face``
   * - ``core_profiles.Ip_profile_face``
     - ``core_profiles.poloidal_flux.I_p_profile_face``
   * - ``core_profiles.j_total``, ``j_total_face``
     - ``core_profiles.poloidal_flux.j_toroidal_total``,
       ``j_toroidal_total_face``
   * - ``core_profiles.vloop_lcfs`` / ``v_loop_lcfs``
     - ``core_profiles.poloidal_flux.v_loop_lcfs``
   * - ``core_profiles.sigma``, ``sigma_face``
     - ``core_profiles.conductivity.sigma``,
       ``core_profiles.conductivity.sigma_face``
   * - ``core_profiles.internal_plasma_energy``
     - Removed (use ``scalars.W_thermal_total`` in post-processed outputs)

``psidot``, ``fast_ions``, ``poloidal_velocity`` (newly added in v2), and
pressure properties (``pressure_thermal_e``, ``pressure_thermal_i``,
``pressure_thermal_total``, ``pressure_fast_i``, ``pressure_total_i``,
``pressure_total``) are direct attributes/properties on ``CoreProfiles``.
Similarly, ``state_history.core_transport``
(``torax.transport.CoreTransport``) now
groups transport coefficients into ``.total``, ``.turbulent``,
``.neoclassical``, and ``.pereverzev`` ``TransportCoeffs`` containers.

``torax.experimental.SimulationStepFn``
---------------------------------------

On ``torax.experimental.SimulationStepFn``, the ``geo_overrides`` parameter has
been renamed to ``geometry_overrides`` on ``__call__``, ``fixed_time_step``, and
``jitted_fixed_time_step``.

Custom physics model registration
---------------------------------

Custom model plugin APIs have been updated to support multi-component transport,
pedestal structure/formation separation, and stateful models. See
:ref:`model-integration` for complete v2 implementation examples:

* **Transport** (``torax.transport``):

  * ``TransportBase`` → ``ComponentTransportConfigBase``
  * ``TransportModel`` → ``ComponentTransportModel``
  * ``RuntimeParams`` → ``ComponentRuntimeParams``
  * ``call_implementation`` now accepts ``two_point_mask`` and
    ``component_state`` and returns
    ``(TransportCoeffs, ComponentTransportState)``.
  * Registered models are configured inside
    ``transport['core_transport_models']`` or
    ``transport['pedestal_transport_models']``.

* **Pedestal** (``torax.pedestal``):

  * ``BasePedestal`` → ``StructureConfigBase`` (implementing
    ``build_structure_model`` instead of ``build_pedestal_model``)
  * ``PedestalModel`` → ``PedestalStructureModel``
  * ``PedestalModelOutput`` → ``PedestalStructure`` (with ``*_top`` attributes
    ``rho_norm_top``, ``T_i_top``, ``T_e_top``, and ``n_e_top``)
  * ``RuntimeParams`` → ``StructureRuntimeParams``
  * ``register_pedestal_model`` → ``register_pedestal_structure_model``
  * ``PedestalStructureModel.__call__`` accepts ``structure_state:
    StructureState`` and returns ``(PedestalStructure, StructureState)``.

* **Sources** (``torax.sources``):

  * ``SourceModelBase`` → ``SourceConfigBase``
  * ``register_source_model_config`` → ``register_source_config``
  * ``SOURCE_NAME`` class attribute → ``SOURCE_ID``
  * ``AffectedCoreProfile`` / ``AFFECTED_CORE_PROFILES`` →
    ``PopulatedSourceOutput`` / ``POPULATED_SOURCE_OUTPUTS``
  * The ``SourceProfileFunction`` callback pattern has been replaced by
    subclassing ``Source``, ``SourceUsingCoreProfiles``, or ``FinalSource``
    (whose ``_get_model_value`` now receives ``calculated_core_sources``) and
    implementing ``_get_model_value`` returning ``(SourceOutput, SourceState)``
    (where ``SourceOutput`` takes named keyword fields ``power_density_i``,
    ``power_density_e``, ``j_parallel``, ``particle_source_e``, and optional
    ``fast_ions``).

* **Edge and neoclassical** (``torax.experimental.edge``,
  ``torax.experimental.neoclassical``): New experimental registration APIs
  (``EdgeModel``, ``EdgeModelConfig``, ``register_edge_model`` and
  ``NeoclassicalModel``, ``NeoclassicalConfigBase``,
  ``register_neoclassical_model``) are now available.

.. _migration_config:

Configuration (``ToraxConfig``) migration
=========================================

.. _migration_global_config:

Global validation and naming conventions
----------------------------------------

1. **Strict schema validation** (``extra='forbid'``): Legacy v1 compatibility
   shims have been removed across all Pydantic config models. Unrecognized,
   misspelled, or moved v1 keys now raise ``pydantic.ValidationError``
   immediately during ``ToraxConfig.from_dict(...)`` or
   ``build_torax_config_from_file(...)``.
2. **Standardized** ``lower_snake_case`` **enums and** ``model_name``
   **values**: Uppercase enum strings and hyphenated ``model_name`` strings have
   been standardized to ``lower_snake_case``:

   .. list-table::
      :header-rows: 1
      :widths: 34 33 33

      * - Config field
        - TORAX v1 value
        - TORAX v2 value
      * - ``sources.<name>.mode``
        - ``'ZERO'``, ``'MODEL_BASED'``, ``'PRESCRIBED'``
        - ``'zero'``, ``'model_based'``, ``'prescribed'``
      * - ``pedestal.mode``
        - ``'ADAPTIVE_SOURCE'``, ``'ADAPTIVE_TRANSPORT'``
        - ``'internal_boundary_condition'`` (replaces ``'ADAPTIVE_SOURCE'``),
          ``'adaptive_transport'``
      * - ``pedestal_profile_form``
        - ``'SET_AT_PED_TOP'``, ``'MTANH'``
        - ``'set_at_top'``, ``'mtanh'``
      * - ``divertor_configuration``
        - ``'HT'``, ``'VT'``
        - ``'horizontal_target'``, ``'vertical_target'``
      * - ``transport`` ``model_name``
        - ``'constant'``, ``'CGM'``, ``'bohm-gyrobohm'``, ``'tglfnn-ukaea'``
        - ``'prescribed'``, ``'cgm'``, ``'bohm_gyrobohm'``, ``'tglfnn_ukaea'``
      * - Interpolation modes (including ``(values, mode)`` tuples)
        - ``'STEP'``, ``'PIECEWISE_LINEAR'``
        - ``'step'``, ``'piecewise_linear'``

3. **Right-continuous** ``step`` **interpolation at knot boundaries**: In TORAX
   v1, step interpolation transitioned strictly *after* knot timestamps
   (:math:`t \in (t_k, t_{k+1}]` evaluated to :math:`y_k`, so :math:`t = t_k`
   returned the previous value :math:`y_{k-1}`). In TORAX v2, step interpolation
   uses standard right-continuous intervals :math:`t \in [t_k, t_{k+1})`, so
   evaluating exactly at knot time :math:`t = t_k` returns :math:`y_k`.

.. _migration_numerics:

``numerics`` and ``time_step_calculator``
-----------------------------------------

See :ref:`numerics <numerics_dataclass>` and
:ref:`time_step_calculator <time_step_calculator_config>` for the full v2
parameter reference.

Simulation time bounds, step-size limits, and solver backtracking controls have
moved from ``numerics`` to ``time_step_calculator``, and several ``numerics``
keys were renamed or removed:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - TORAX v1 (``numerics``)
     - TORAX v2
   * - ``numerics.t_initial``, ``t_final``, ``exact_t_final``, ``min_dt``
     - ``time_step_calculator.t_initial``, ``t_final``, ``exact_t_final``,
       ``min_dt``
   * - ``numerics.max_dt``
     - ``time_step_calculator.max_dt`` (valid when ``calculator_type='chi'``,
       the default, or ``'from_previous_dt'``; rejected when ``'fixed'``)
   * - ``numerics.chi_timestep_prefactor``
     - ``time_step_calculator.chi_timestep_prefactor`` (only valid when
       ``calculator_type='chi'``)
   * - ``numerics.fixed_dt``
     - ``time_step_calculator.dt`` (only valid when
       ``calculator_type='fixed'``)
   * - ``numerics.adaptive_dt``
     - ``time_step_calculator.allow_backtracking``
   * - ``numerics.dt_reduction_factor``
     - ``time_step_calculator.dt_backtracking_factor``
   * - ``numerics.resistivity_multiplier``
     - ``numerics.current_diffusion_time_reduction_factor`` (accelerates
       :math:`\partial \psi / \partial t` current diffusion only, leaving
       conductivity :math:`\sigma_\parallel`, Ohmic heating, loop voltage, and
       Ware pinch unchanged), **or**
       ``neoclassical.conductivity.conductivity_multiplier``
       (``= 1.0 / resistivity_multiplier``, scales physical conductivity
       everywhere)
   * - ``numerics.T_minimum_eV``
     - ``numerics.T_min_eV`` (default ``5.0``)
   * - ``numerics.adaptive_T_source_prefactor``, ``adaptive_n_source_prefactor``
     - Removed (obsoleted by exact Dirichlet internal boundary conditions)
   * - ``numerics.calcphibdot``
     - ``geometry.calc_Phi_b_dot``
   * - *(Unbounded step count)*
     - ``time_step_calculator.max_steps`` (required when ``calculator_type`` is
       ``'chi'`` or ``'from_previous_dt'``; computed automatically when
       ``'fixed'``)

.. tab-set::

   .. tab-item:: TORAX v1

      .. code-block:: python

         'numerics': {
             't_initial': 0.0,
             't_final': 80.0,
             'fixed_dt': 2.0,
             'adaptive_dt': True,
             'dt_reduction_factor': 3.0,
             'resistivity_multiplier': 1.0,
             'evolve_ion_heat': True,
             'evolve_electron_heat': True,
             'evolve_current': True,
             'evolve_density': True,
         },
         'time_step_calculator': {
             'calculator_type': 'fixed',
         },

   .. tab-item:: TORAX v2

      .. code-block:: python

         'numerics': {
             'current_diffusion_time_reduction_factor': 1.0,
             'evolve_ion_heat': True,
             'evolve_electron_heat': True,
             'evolve_current': True,
             'evolve_density': True,
         },
         'time_step_calculator': {
             'calculator_type': 'fixed',
             't_initial': 0.0,
             't_final': 80.0,
             'dt': 2.0,
             'allow_backtracking': True,
             'dt_backtracking_factor': 3.0,
         },

.. _migration_profile_conditions:

``profile_conditions``
----------------------

See :ref:`profile_conditions <profile_conditions>` for the full v2 parameter
reference.

.. important::
   ``profile_conditions['psi_source']`` is **required** in TORAX v2 and has no
   default value (governing :math:`\psi` at initialization and for
   :math:`t > t_{\mathrm{initial}}` when ``numerics.evolve_current=False``).
   Set it explicitly in every configuration:

   * ``'j'``: compute :math:`\psi` from the initial current profile at
     initialization (replaces ``initial_psi_mode='j'`` or ``psi=None`` with
     ``initial_psi_from_j=True``).
   * ``'geometry'``: take :math:`\psi` from the equilibrium geometry (replaces
     ``initial_psi_mode='geometry'`` or ``psi=None`` with
     ``initial_psi_from_j=False``; not available for ``circular`` geometry).
   * ``'profile_conditions'``: take :math:`\psi` from a prescribed
     ``profile_conditions['psi']`` profile (replaces
     ``initial_psi_mode='profile_conditions'`` or providing ``psi`` directly;
     ``psi`` must be ``None`` when ``psi_source != 'profile_conditions'``).

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - TORAX v1 (``profile_conditions``)
     - TORAX v2 (``profile_conditions``)
   * - ``Ip``
     - ``I_p`` (defaults to ``None``; required when
       ``I_p_source='profile_conditions'``)
   * - ``geometry.Ip_from_parameters: True / False``
     - ``I_p_source: 'profile_conditions'`` (default) or ``'geometry'``
   * - ``initial_psi_from_j`` / ``initial_psi_mode``
     - ``psi_source`` (required: ``'j'``, ``'geometry'``, or
       ``'profile_conditions'``)
   * - ``use_vloop_lcfs_boundary_condition`` /
       ``use_v_loop_lcfs_boundary_condition``
     - ``psi_bc_source``: ``'I_p'`` (default) or ``'v_loop'``
   * - ``vloop_lcfs``
     - ``v_loop_lcfs`` (defaults to ``None``; required when
       ``psi_bc_source='v_loop'`` and rejected when ``psi_bc_source='I_p'``)
   * - ``nbar``
     - ``n_e_bar``
   * - ``n_e_nbar_is_fGW``
     - ``n_e_bar_is_f_GW``
   * - ``normalize_n_e_to_nbar``
     - ``normalize_n_e``
   * - ``n_e_right_bc_is_fGW``
     - ``n_e_right_bc_is_f_GW``
   * - ``set_pedestal``
     - Removed (configure ``pedestal.formation_model`` instead)
   * - ``internal_boundary_conditions: {...}`` (v1.4)
     - Add ``'model_name': 'prescribed'``; ``internal_boundary_conditions`` is
       now a discriminated union of ``'no_ibc'`` (default), ``'prescribed'``, or
       ``'beta_poloidal_prime'``

.. tab-set::

   .. tab-item:: TORAX v1

      .. code-block:: python

         'profile_conditions': {
             'Ip': {0: 3e6, 80: 10.5e6},
             'initial_psi_from_j': False,
             'n_e': {0: {0.0: 1.5, 1.0: 1.0}},
             'nbar': 0.85,
             'n_e_nbar_is_fGW': True,
             'normalize_n_e_to_nbar': True,
             'n_e_right_bc_is_fGW': True,
             'n_e_right_bc': 0.2,
         },

   .. tab-item:: TORAX v2

      .. code-block:: python

         'profile_conditions': {
             'I_p': {0: 3e6, 80: 10.5e6},
             'I_p_source': 'profile_conditions',
             'psi_source': 'geometry',
             'n_e': {0: {0.0: 1.5, 1.0: 1.0}},
             'n_e_bar': 0.85,
             'n_e_bar_is_f_GW': True,
             'normalize_n_e': True,
             'n_e_right_bc_is_f_GW': True,
             'n_e_right_bc': 0.2,
         },

.. note::
   ``ProfileConditions`` enforces unit-magnitude sanity checks on ``I_p`` (SI
   amperes, not MA) and on ``n_e``, ``n_e_bar``, and ``n_e_right_bc`` (SI
   :math:`\mathrm{m^{-3}}` unless the corresponding ``*_is_f_GW`` flag is
   ``True``). When ``n_e_bar_is_f_GW=True`` and ``normalize_n_e=False``,
   ``n_e`` itself is interpreted in :math:`f_{\mathrm{GW}}` and cannot be left
   at its SI default (``1.2e20``).

.. _migration_plasma_composition:

``plasma_composition``
----------------------

See :ref:`plasma_composition <plasma_composition_config>` for the full v2
parameter reference.

Top-level shorthand impurity strings or species dictionaries (such as
``'impurity': 'Ne'`` or ``'impurity': {'Ne': 0.8, 'W': 0.2}``) and top-level
``Z_impurity_override`` / ``A_impurity_override`` keys have been removed. When
``impurity`` is specified, it must be a dictionary with explicit
``'impurity_mode'`` (``'fractions'``, ``'n_e_ratios'``, or
``'n_e_ratios_Z_eff'``) and ``'species'`` keys, plus optional ``'Z_override'``
and ``'A_override'`` keys. In ``'fractions'`` mode, ``'species'`` accepts either
a species-to-fraction dictionary or a single ion string such as ``'Ne'``;
omitting ``'impurity'`` defaults to ``{'impurity_mode': 'fractions', 'species':
{'Ne': 1.0}}``:

.. tab-set::

   .. tab-item:: TORAX v1

      .. code-block:: python

         'plasma_composition': {
             'main_ion': {'D': 0.5, 'T': 0.5},
             'impurity': {'Ne': 0.8, 'W': 0.2},
             'Z_eff': 1.6,
             'Z_impurity_override': 10.0,
         },

   .. tab-item:: TORAX v2

      .. code-block:: python

         'plasma_composition': {
             'main_ion': {'D': 0.5, 'T': 0.5},
             'impurity': {
                 'impurity_mode': 'fractions',
                 'species': {'Ne': 0.8, 'W': 0.2},
                 'Z_override': 10.0,
             },
             'Z_eff': 1.6,
         },

.. _migration_geometry:

``geometry``
------------

See :ref:`geometry <geometry_doc>` for the full v2 parameter reference.

* **FBT renamed to MEQ**: ``'geometry_type': 'fbt'`` has been renamed to
  ``'geometry_type': 'meq'``.
* ``calcphibdot`` **renamed**: ``calcphibdot`` is now ``calc_Phi_b_dot``.
* ``Ip_from_parameters`` **removed**: Removed from ``chease``, ``eqdsk``,
  ``meq``, and ``imas`` geometry configs. Use
  ``profile_conditions['I_p_source'] = 'geometry'`` (equivalent to
  ``Ip_from_parameters: False``) or ``'profile_conditions'`` (equivalent to
  ``Ip_from_parameters: True``).
* **EQDSK COCOS required**: ``geometry_type='eqdsk'`` requires an explicit
  integer ``cocos`` index.
* **IMAS slice loading**: ``slice_index`` and ``load_all_time_slices`` have been
  removed from ``imas`` geometry configs. All time slices are loaded
  automatically when ``slice_time`` is omitted (``None``), or a single time
  slice is loaded when ``slice_time`` is set.

.. _migration_transport:

``transport``
-------------

See :ref:`transport <transport_config>` for the full v2 parameter reference.

In TORAX v2, ``transport`` is always configured via named component dictionaries
(``core_transport_models`` and optional ``pedestal_transport_models``) rather
than a flat single-model dictionary or ``'model_name': 'combined'`` list:

* **Component model dictionaries**: Move single-model keys (e.g.
  ``{'model_name': 'qlknn', ...}``) or ``transport_models`` list entries into
  ``transport['core_transport_models'] = {'<key>': {'model_name': ..., ...}}``
  (passing ``'transport': {}`` defaults to ``{'core_transport_models':
  {'prescribed': {'model_name': 'prescribed'}}}``).
* **Inner and outer patches replaced by** ``'prescribed'`` **components**:
  ``apply_inner_patch``, ``D_e_inner``, ``V_e_inner``, ``chi_i_inner``,
  ``chi_e_inner``, ``rho_inner``, ``apply_outer_patch``, ``D_e_outer``,
  ``V_e_outer``, ``chi_i_outer``, ``chi_e_outer``, and ``rho_outer`` have been
  removed. Add ``'prescribed'`` entries to ``core_transport_models`` with
  ``rho_min`` / ``rho_max`` bounds.

  .. warning::
     Each transport component's ``merge_mode`` defaults to ``'add'``. When
     replacing ``apply_inner_patch`` or ``apply_outer_patch``, either set
     ``'merge_mode': 'overwrite'`` on the ``'prescribed'`` patch (if the
     turbulence model stays on ``[0, 1]``) or partition ``rho_min`` /
     ``rho_max`` into disjoint intervals (active domain is
     :math:`(\rho_{\min}, \rho_{\max}]`, including :math:`0` when
     ``rho_min=0``). Leaving both active on overlapping radii with the default
     ``'merge_mode': 'add'`` sums the patch onto the turbulence model instead of
     replacing it.

* **Top-level clipping and localized smoothing**: ``chi_min``, ``chi_max``,
  ``D_e_min``, ``D_e_max``, ``V_e_min``, ``V_e_max``, ``smoothing_width``, and
  ``pedestal_smoothing_width`` live only on top-level ``transport`` (no longer
  accepted inside individual component dictionaries), and ``smooth_everywhere``
  has been removed. Note that ``transport['smoothing_width']`` defaults to
  ``0.0`` even when ``'qlknn'`` is used (whereas v1 ``'qlknn'`` defaulted
  ``smoothing_width=0.1`` on ``QLKNN_7_11``), so set ``'smoothing_width': 0.1``
  explicitly to retain radial smoothing. To restrict smoothing to specific
  radial intervals, specify ``smoothing_zones`` with ``{'rho_min', 'rho_max',
  'smoothing_width'}`` dictionaries.
* **Model name renames**: ``'constant'`` → ``'prescribed'``, ``'CGM'`` →
  ``'cgm'``, ``'bohm-gyrobohm'`` → ``'bohm_gyrobohm'``, ``'tglfnn-ukaea'`` →
  ``'tglfnn_ukaea'``.
* **Opt-in QLKNN mode splits**: ``'qlknn'`` now defaults to
  ``'output_mode_contributions': False``; set ``'output_mode_contributions':
  True`` when you need separate ITG/TEM/ETG transport outputs.
* **In-memory TGLF** (``model_name='tglf'``): TGLF now executes in memory via
  ``tglf2py``; subprocess options ``tglf_exec_path``, ``output_directory``,
  ``verbose``, and ``use_legacy_torax_defaults`` have been removed, and flat
  lowercase TGLF namelist fields (``sat_rule``, ``use_bper``, ``n_ky``, etc.)
  must be passed inside ``tglf_settings`` using uppercase TGLF keys
  (``'SAT_RULE'``, ``'USE_BPER'``, etc.).

.. tab-set::

   .. tab-item:: TORAX v1

      .. code-block:: python

         'transport': {
             'model_name': 'qlknn',
             'DV_effective': True,
             'An_min': 0.05,
             'apply_inner_patch': True,
             'rho_inner': 0.25,
             'chi_i_inner': 1.5,
             'chi_e_inner': 1.5,
             'D_e_inner': 0.25,
             'V_e_inner': 0.0,
             'chi_min': 0.05,
             'smoothing_width': 0.1,
         },

   .. tab-item:: TORAX v2

      .. code-block:: python

         'transport': {
             'core_transport_models': {
                 'qlknn': {
                     'model_name': 'qlknn',
                     'DV_effective': True,
                     'An_min': 0.05,
                     'rho_min': 0.25,
                 },
                 'inner_patch': {
                     'model_name': 'prescribed',
                     'rho_max': 0.25,
                     'chi_i': 1.5,
                     'chi_e': 1.5,
                     'D_e': 0.25,
                     'V_e': 0.0,
                 },
             },
             'chi_min': 0.05,
             'smoothing_width': 0.1,
         },

.. _migration_pedestal:

``pedestal``
------------

See :ref:`pedestal <pedestal_config>` for the full v2 parameter reference.

The flat ``pedestal`` dictionary has been reorganized into modular sub-models,
and pedestal-top quantities are standardized to ``*_top`` (rather than
``*_ped`` or ``*_ped_top``):

* ``structure_model``: ``'no_pedestal'`` (default when ``'pedestal': {}``),
  ``'set_T_top_n_top'`` (replaces ``'set_T_ped_n_ped'``), or
  ``'set_p_top_n_top'`` (replaces ``'set_P_ped_n_ped'``, using lowercase
  ``p``); external models such as ``'epednn_mit'`` can be registered via
  ``torax.pedestal.register_pedestal_structure_model``. Parameter renames inside
  ``structure_model``:

  * ``T_i_ped``, ``T_e_ped`` → ``T_i_top``, ``T_e_top``
  * ``P_ped`` → ``p_top`` (with optional ``p_top_multiplier``;
    ``T_i_T_e_ratio`` is unchanged)
  * ``n_e_ped``, ``n_e_ped_is_fGW`` → ``n_e_top``, ``n_e_top_is_f_GW``
  * ``rho_norm_ped_top`` → ``rho_norm_top``
  * ``pedestal_profile_form``: ``'set_at_top'`` (default; replaces
    ``'SET_AT_PED_TOP'``) or ``'mtanh'`` (replaces ``'MTANH'``)

* ``formation_model``: ``'prescribed'`` (with ``pedestal_active``, a ``bool``
  or time-varying step schedule), ``'martin_scaling'``, or
  ``'delabie_scaling'``. Scaling-law models now own ``P_LH_hysteresis_factor``
  and ``include_dW_dt_in_P_SOL``, while ``'delabie_scaling'`` also owns
  ``divertor_configuration`` (``'horizontal_target'`` or ``'vertical_target'``,
  replacing ``'HT'`` / ``'VT'``).
* ``saturation_model``: ``'profile_value'`` (used with
  ``mode='adaptive_transport'``).
* ``mode``: ``'internal_boundary_condition'`` (default; replaces removed
  ``'ADAPTIVE_SOURCE'``) or ``'adaptive_transport'`` (replaces
  ``'ADAPTIVE_TRANSPORT'``). ``set_pedestal`` and
  ``use_formation_model_with_adaptive_source`` have been removed.

.. warning::
   Watch for three default changes when migrating ``pedestal`` configs:

   1. **Active by default when** ``structure_model`` **is set**: In TORAX v1,
      ``set_pedestal`` defaulted to ``False`` even when pedestal parameters were
      present. In TORAX v2, configuring an active ``structure_model`` (such as
      ``'set_T_top_n_top'``) in ``'internal_boundary_condition'`` mode defaults
      ``formation_model`` to ``{'model_name': 'prescribed', 'pedestal_active':
      True}``. To disable the pedestal, omit ``structure_model`` (``'pedestal':
      {}``), set ``structure_model={'model_name': 'no_pedestal'}``, or set
      ``formation_model={'model_name': 'prescribed', 'pedestal_active':
      False}``.
   2. ``mode='adaptive_transport'`` **requires** ``formation_model``: When using
      ``'adaptive_transport'``, explicitly provide ``formation_model``.
   3. ``transition_time_width`` **required for scaling-law L–H transitions in
      IBC mode**: When combining ``'martin_scaling'`` or ``'delabie_scaling'``
      with ``mode='internal_boundary_condition'``, explicitly set
      ``pedestal['transition_time_width']`` (previously defaulted to ``0.5`` s;
      now defaults to ``0.0`` and raises a validation error if omitted with a
      scaling-law formation model).

.. tab-set::

   .. tab-item:: TORAX v1

      .. code-block:: python

         'pedestal': {
             'model_name': 'set_T_ped_n_ped',
             'set_pedestal': True,
             'mode': 'ADAPTIVE_SOURCE',
             'T_i_ped': 1.0,
             'T_e_ped': 1.0,
             'n_e_ped': 0.7,
             'n_e_ped_is_fGW': True,
             'rho_norm_ped_top': 0.9,
         },

   .. tab-item:: TORAX v2

      .. code-block:: python

         'pedestal': {
             'mode': 'internal_boundary_condition',
             'structure_model': {
                 'model_name': 'set_T_top_n_top',
                 'T_i_top': 1.0,
                 'T_e_top': 1.0,
                 'n_e_top': 0.7,
                 'n_e_top_is_f_GW': True,
                 'rho_norm_top': 0.9,
             },
         },

.. _migration_sources:

``sources``
-----------

See :ref:`sources <sources_config>` for the full v2 parameter reference.

* **Omitted vs empty source dictionaries**: Omitting a source key from
  ``sources`` disables it (except ``ei_exchange``, which defaults to
  ``{'mode': 'zero'}``; note that ``generic_current`` now defaults to ``None``
  when omitted rather than ``{'mode': 'ZERO'}``). When a source dictionary *is*
  present, its ``mode`` defaults to ``'model_based'`` (lowercase ``'zero'``,
  ``'model_based'``, or ``'prescribed'``).
* **Named-channel** ``prescribed_values``: When ``mode='prescribed'``,
  ``prescribed_values`` must be provided explicitly as a dictionary keyed by
  output channel name (``'power_density_i'``, ``'power_density_e'``,
  ``'j_parallel'``, ``'particle_source_e'``, plus optional ``'fast_ions'`` on
  fast-ion sources) rather than a bare radial array or positional tuple.
* **Arbitrary physical defaults removed**: Passing an empty ``{}`` dictionary
  for parameterized sources (which previously injected arbitrary default powers,
  particle rates, or Gaussian locations) now raises ``pydantic.ValidationError``
  when ``mode='model_based'``. Either supply the required parameters below, set
  ``'mode': 'zero'``, or remove the source entry from ``sources``:

  .. list-table::
     :header-rows: 1
     :widths: 32 68

     * - Source key
       - Required fields when ``mode='model_based'``
     * - ``generic_heat``
       - ``P_total``, ``gaussian_width``, ``gaussian_location``
     * - ``generic_particle``
       - ``S_total``, ``particle_width``, ``deposition_location``
     * - ``generic_current``
       - ``gaussian_width``, ``gaussian_location``, plus
         ``fraction_of_total_current`` (when ``use_absolute_current=False``) or
         ``I_generic`` (when ``use_absolute_current=True``)
     * - ``gas_puff``
       - ``S_total``, ``puff_decay_length``
     * - ``pellet``
       - ``S_total``, ``pellet_width``, ``pellet_deposition_location``
     * - ``ecrh``
       - ``P_total``, ``gaussian_width``, and ``gaussian_location``, and/or
         ``extra_prescribed_power_density``
     * - ``icrh``
       - ``P_total`` (and ``minority_species``, default ``'He3'``, must be
         present in ``plasma_composition``)
     * - ``impurity_radiation`` (``model_name='P_in_scaled_flat_profile'``)
       - ``fraction_P_heating``

  Zero-parameter physics sources (``fusion``, ``ohmic``, ``bremsstrahlung``,
  ``cyclotron_radiation``, ``ei_exchange``, and ``impurity_radiation`` with
  default ``model_name='mavrin_fit'``) continue to accept ``{}``.

* **ICRH** ``toric_nn`` **moved to external package**: The built-in default ICRH
  model is now ``'model_name': 'scaled_profile'`` (``minority_concentration``
  has been removed, ``minority_species`` must be present in
  ``plasma_composition``, and non-zero ``heat_profile_ion`` /
  ``heat_profile_electron`` profiles must be supplied since they default to
  zero). The neural-network surrogate ``'toric_nn'`` has moved out of core
  TORAX into the external ``toric_surrogate`` package; register
  ``ToricNNIonCyclotronSourceConfig`` via
  ``torax.sources.register_source_config(...)`` before building a config with
  ``'model_name': 'toric_nn'``.

.. _migration_solver_edge_neoclassical:

``solver``, ``edge``, ``neoclassical``, ``mhd``, and ``restart``
----------------------------------------------------------------

* :ref:`solver <solver_config>`:

  * ``use_predictor_corrector`` → ``use_fixed_point``
  * ``n_corrector_steps`` → ``n_fixed_point_iterations``
  * ``convection_dirichlet_mode`` and ``convection_neumann_mode`` have been
    removed (replaced by unconditionally stable Scharfetter–Gummel convection
    discretization and ghost-cell boundary extrapolation).
  * ``log_iterations`` is only valid when ``solver_type='newton_raphson'`` (no
    longer accepted on ``'linear'`` or ``'optimizer'``).

* :ref:`edge <edge_models>` (``extended_lengyel``):

  * Lowercase enum strings: ``computation_mode`` (``'forward'``, ``'inverse'``),
    ``solver_mode`` (``'hybrid'``, ``'newton_raphson'``, ``'fixed_point'``), and
    ``impurity_sot`` (``'core'``, ``'edge'``).
  * ``ne_tau`` → ``n_e_tau``
  * ``update_temperatures`` and ``update_impurities`` have been replaced by four
    independent flags: ``update_T_e``, ``update_T_i``, ``update_n_e``, and
    ``update_n_impurities`` (on ``extended_lengyel``, ``update_T_e``,
    ``update_T_i``, and ``update_n_impurities`` default to ``True``, while
    ``update_n_e`` is ``False``).
  * Cross-validators now enforce: non-``circular`` geometry;
    ``plasma_composition['impurity']['impurity_mode'] == 'n_e_ratios'`` with
    strictly positive (``> 0``) values at ``rho_norm=1.0`` whenever
    ``update_n_impurities`` is active; exact species match between core
    ``impurity['species']`` and edge fixed/seeded impurities; and ``diverted``
    explicitly set for non-``meq`` geometries (and omitted for ``meq``).

* :ref:`neoclassical <neoclassical_config>`:

  * ``poloidal_velocity_multiplier`` (previously a top-level float on
    ``neoclassical`` in v1.4) is now configured inside a nested
    ``poloidal_velocity`` dictionary:
    ``'poloidal_velocity': {'model_name': 'kim',
    'poloidal_velocity_multiplier': 1.0}`` (or ``{'model_name': 'zeros'}``).
  * ``conductivity`` supports ``'model_name': 'sauter'`` (default) or
    ``'redl'``, along with ``conductivity_multiplier`` (for scaling physical
    parallel conductivity) and ``neoclassical_correction``.

* :ref:`mhd <mhd_config>`:

  * ``mhd.sawtooth.redistribution_model.mixing_radius_multiplier`` strictly
    enforces ``> 1.0``.

* :ref:`restart <restart_config>`:

  * TORAX v1 NetCDF (``.nc``) state files cannot be loaded directly via
    ``restart['filename']`` in TORAX v2 (loading raises ``KeyError`` on renamed
    variables such as ``profiles.I_p_profile``, ``scalars.v_loop_lcfs``, and
    ``scalars.E_aux_total``). Regenerate baseline restart checkpoints in TORAX
    v2 (or update the NetCDF variable names before restarting).
  * When ``restart['stitch'] = True``,
    ``state_history.simulation_output_to_xr()`` concatenates the prior state
    file automatically.

Complete v1 to v2 ``CONFIG`` example
------------------------------------

.. tab-set::

   .. tab-item:: TORAX v1

      .. code-block:: python

         CONFIG = {
             'plasma_composition': {
                 'main_ion': {'D': 0.5, 'T': 0.5},
                 'impurity': 'Ne',
                 'Z_eff': 1.6,
             },
             'profile_conditions': {
                 'Ip': {0: 3e6, 80: 10.5e6},
                 'initial_psi_from_j': False,
                 'n_e': {0: {0.0: 1.5, 1.0: 1.0}},
                 'nbar': 0.85,
                 'n_e_nbar_is_fGW': True,
                 'normalize_n_e_to_nbar': True,
                 'n_e_right_bc_is_fGW': True,
                 'n_e_right_bc': 0.2,
             },
             'numerics': {
                 't_final': 80.0,
                 'fixed_dt': 2.0,
                 'dt_reduction_factor': 3.0,
                 'resistivity_multiplier': 1.0,
             },
             'geometry': {
                 'model_name': 'chease',
                 'geometry_file': 'iterhybrid.mat2cols',
                 'Ip_from_parameters': True,
             },
             'sources': {
                 'generic_heat': {},
                 'fusion': {'mode': 'MODEL_BASED'},
                 'ohmic': {},
                 'ei_exchange': {},
             },
             'pedestal': {
                 'model_name': 'set_T_ped_n_ped',
                 'set_pedestal': True,
                 'mode': 'ADAPTIVE_SOURCE',
                 'T_i_ped': 1.0,
                 'T_e_ped': 1.0,
                 'n_e_ped': 0.7,
                 'n_e_ped_is_fGW': True,
                 'rho_norm_ped_top': 0.9,
             },
             'transport': {
                 'model_name': 'bohm-gyrobohm',
                 'apply_inner_patch': True,
                 'rho_inner': 0.25,
                 'chi_i_inner': 1.5,
                 'chi_e_inner': 1.5,
                 'D_e_inner': 0.25,
                 'V_e_inner': 0.0,
             },
             'solver': {
                 'solver_type': 'newton_raphson',
                 'use_predictor_corrector': True,
                 'n_corrector_steps': 5,
             },
             'time_step_calculator': {
                 'calculator_type': 'fixed',
             },
         }

   .. tab-item:: TORAX v2

      .. code-block:: python

         CONFIG = {
             'plasma_composition': {
                 'main_ion': {'D': 0.5, 'T': 0.5},
                 'impurity': {
                     'impurity_mode': 'fractions',
                     'species': 'Ne',
                 },
                 'Z_eff': 1.6,
             },
             'profile_conditions': {
                 'I_p': {0: 3e6, 80: 10.5e6},
                 'I_p_source': 'profile_conditions',
                 'psi_source': 'geometry',
                 'n_e': {0: {0.0: 1.5, 1.0: 1.0}},
                 'n_e_bar': 0.85,
                 'n_e_bar_is_f_GW': True,
                 'normalize_n_e': True,
                 'n_e_right_bc_is_f_GW': True,
                 'n_e_right_bc': 0.2,
             },
             'numerics': {
                 'current_diffusion_time_reduction_factor': 1.0,
             },
             'geometry': {
                 'model_name': 'chease',
                 'geometry_file': 'iterhybrid.mat2cols',
             },
             'sources': {
                 'generic_heat': {
                     'P_total': 120.0e6,
                     'gaussian_location': 0.0,
                     'gaussian_width': 0.25,
                 },
                 'fusion': {'mode': 'model_based'},
                 'ohmic': {},
                 'ei_exchange': {},
             },
             'pedestal': {
                 'mode': 'internal_boundary_condition',
                 'structure_model': {
                     'model_name': 'set_T_top_n_top',
                     'T_i_top': 1.0,
                     'T_e_top': 1.0,
                     'n_e_top': 0.7,
                     'n_e_top_is_f_GW': True,
                     'rho_norm_top': 0.9,
                 },
             },
             'transport': {
                 'core_transport_models': {
                     'bohm_gyrobohm': {
                         'model_name': 'bohm_gyrobohm',
                         'rho_min': 0.25,
                     },
                     'inner_patch': {
                         'model_name': 'prescribed',
                         'rho_max': 0.25,
                         'chi_i': 1.5,
                         'chi_e': 1.5,
                         'D_e': 0.25,
                         'V_e': 0.0,
                     },
                 },
             },
             'solver': {
                 'solver_type': 'newton_raphson',
                 'use_fixed_point': True,
                 'n_fixed_point_iterations': 5,
             },
             'time_step_calculator': {
                 'calculator_type': 'fixed',
                 't_final': 80.0,
                 'dt': 2.0,
                 'dt_backtracking_factor': 3.0,
             },
         }

.. _migration_outputs:

Simulation outputs (``xr.DataTree`` and NetCDF) and plotting
============================================================

Hierarchy and metadata updates
------------------------------

* **Physical units metadata**: Every physical output ``xarray.DataArray`` now
  carries units in ``.attrs['units']`` (e.g. ``'keV'``, ``'m^-3'``, ``'A'``,
  ``'W'``, ``'dimensionless'``).
* ``auxiliary`` **DataTree node**: Per-model and per-mode turbulent transport
  coefficients have moved out of ``profiles`` into
  ``data_tree['auxiliary/turbulent_transport/core/<model_key>']`` and
  ``data_tree['auxiliary/turbulent_transport/pedestal/<model_key>']``:

  * **All components**: unclipped/unsmoothed per-component ``chi_turb_i``,
    ``chi_turb_e``, ``D_turb_e``, and ``V_turb_e``.
  * **QLKNN** (when ``'output_mode_contributions': True``; defaults to
    ``False``): ``chi_itg_e``, ``chi_tem_e``, ``chi_etg_e``, ``chi_itg_i``,
    ``chi_tem_i``, ``D_itg_e``, ``D_tem_e``, ``V_itg_e``, ``V_tem_e`` (not
    emitted by full ``'qualikiz'``).
  * **Bohm–GyroBohm**: ``chi_bohm_e``, ``chi_gyrobohm_e``, ``chi_bohm_i``,
    ``chi_gyrobohm_i``.

  Total turbulent and neoclassical coefficients (``chi_turb_i``, ``chi_turb_e``,
  ``D_turb_e``, ``V_turb_e``, ``chi_neo_i``, ``chi_neo_e``, ``D_neo_e``,
  ``V_neo_e``, ``V_neo_ware_e``) remain in ``data_tree.profiles``.

Renamed, relocated, and removed output variables
------------------------------------------------

See :ref:`output` for the complete TORAX v2 output specification. Update
NetCDF / ``xarray`` post-processing scripts according to the table below:

.. list-table::
   :header-rows: 1
   :widths: 18 38 44

   * - Dataset
     - TORAX v1 variable
     - TORAX v2 variable / location
   * - ``scalars``
     - ``Ip``
     - ``I_p``
   * - ``scalars``
     - ``vloop_lcfs``
     - ``v_loop_lcfs``
   * - ``scalars``
     - ``fgw_n_e_volume_avg``, ``fgw_n_e_line_avg``
     - ``f_GW_n_e_volume_avg``, ``f_GW_n_e_line_avg``
   * - ``scalars``
     - ``I_aux_generic``
     - ``I_generic_current``
   * - ``scalars``
     - ``P_aux_generic_i``, ``P_aux_generic_e``, ``P_aux_generic_total``
     - ``P_generic_heat_i``, ``P_generic_heat_e``, ``P_generic_heat_total``
   * - ``scalars``
     - ``P_cyclotron_e``, ``P_radiation_e``
     - ``P_cyclotron_radiation_e``, ``P_impurity_radiation_e``
   * - ``scalars``
     - ``P_LH``, ``P_LH_high_density``, ``P_LH_low_density``, ``P_LH_min``,
       ``n_e_min_P_LH``
     - ``P_LH_martin``, ``P_LH_martin_high_density``,
       ``P_LH_martin_low_density``, ``P_LH_martin_min``,
       ``n_e_min_P_LH_martin`` (alongside ``*_delabie*``)
   * - ``scalars``
     - ``E_aux`` (v1.0)
     - Split into ``E_aux_total``, ``E_ohmic_e``, ``E_external_injected``, and
       ``E_external_total``
   * - ``scalars`` → ``profiles``
     - ``scalars.A_impurity`` (when radially uniform)
     - Always ``profiles.A_impurity`` on ``(time, rho_norm)``
   * - ``profiles``
     - ``Ip_profile``, ``Ip_profile_from_geo``, ``psi_from_Ip``
     - ``I_p_profile``, ``I_p_profile_from_geo``, ``psi_from_I_p``
   * - ``profiles``
     - ``j_total``, ``j_ohmic``, ``j_bootstrap``, ``j_external``, ``j_ecrh``,
       ``j_generic_current``, ``j_non_inductive``
     - ``j_toroidal_total``, ``j_toroidal_ohmic``, ``j_toroidal_bootstrap``,
       ``j_toroidal_external``, ``j_toroidal_ecrh``,
       ``j_toroidal_generic_current``, ``j_toroidal_non_inductive``
       (distinguishing toroidal :math:`j_{\mathrm{tor}}` from parallel
       ``j_parallel_*``)
   * - ``profiles``
     - ``ei_exchange``
     - ``p_ei_exchange_i`` (to ions) and ``p_ei_exchange_e``
       (``= -p_ei_exchange_i``, to electrons)
   * - ``profiles``
     - ``sigma_parallel`` on ``(time, rho_cell_norm)``
     - ``sigma_parallel`` on ``(time, rho_norm)`` (extended grid including
       boundaries)
   * - ``scalars``
     - ``drho``, ``drho_norm``
     - Removed (compute ``drho_norm = xr.DataArray(
       np.diff(data_tree.rho_face_norm), dims=['rho_cell_norm'])`` and
       ``drho = drho_norm * data_tree.scalars.rho_b``)
   * - ``numerics``
     - ``sim_status``
     - Removed (check ``data_tree.numerics.sim_error == 0`` or
       ``state_history.sim_error == torax.SimError.NO_ERROR``)
   * - ``edge``
     - ``calculated_enrichment`` dimension ``'impurity'``
     - Renamed to ``'enrichment_impurity'`` (distinct from ``'seed_impurity'``)

Disabled (``None``) sources now omit their ``p_*``, ``s_*``, and
``j_parallel_*`` arrays from ``data_tree.profiles`` rather than writing
all-zero arrays.

Custom plotting configurations (``PLOT_CONFIG``)
------------------------------------------------

If you maintain custom ``--plot_config`` files (see :ref:`plotting`), update any
``attrs`` tuples referencing old profile or scalar names:

* ``'Ip_profile'`` → ``'I_p'`` in ``PlotType.TIME_SERIES`` subplots (v1
  aliased ``'Ip_profile'`` to scalar ``Ip``), or ``'I_p_profile'`` in
  ``PlotType.SPATIAL`` subplots.
* ``'j_total'``, ``'j_ohmic'``, ``'j_bootstrap'``, ``'j_external'``,
  ``'j_ecrh'``, ``'j_generic_current'``, ``'j_non_inductive'`` →
  ``'j_toroidal_total'``, ``'j_toroidal_ohmic'``, ``'j_toroidal_bootstrap'``,
  ``'j_toroidal_external'``, ``'j_toroidal_ecrh'``,
  ``'j_toroidal_generic_current'``, ``'j_toroidal_non_inductive'``.
* ``'ei_exchange'`` → ``'p_ei_exchange_i'`` or ``'p_ei_exchange_e'``.
* ``'Ip'`` → ``'I_p'``, ``'fgw_n_e_line_avg'`` → ``'f_GW_n_e_line_avg'``,
  ``'fgw_n_e_volume_avg'`` → ``'f_GW_n_e_volume_avg'``.
* ``'I_aux_generic'`` → ``'I_generic_current'``, ``'P_aux_generic_i'`` /
  ``'P_aux_generic_e'`` / ``'P_aux_generic_total'`` → ``'P_generic_heat_i'`` /
  ``'P_generic_heat_e'`` / ``'P_generic_heat_total'``, ``'P_cyclotron_e'`` →
  ``'P_cyclotron_radiation_e'``, and ``'P_radiation_e'`` →
  ``'P_impurity_radiation_e'``.

.. _migration_physics:

Physics and numerical updates affecting results
===============================================

Even after translating a v1 configuration to v2, simulation trajectories will
not be bitwise identical to TORAX v1 due to physics and numerical accuracy
improvements:

* **Exact Dirichlet internal boundary conditions**: Pedestal top constraints
  (``mode='internal_boundary_condition'``) and prescribed internal boundary
  conditions now impose exact Dirichlet constraints directly in the discretized
  linear system rather than stiff localized penalty sources
  (``ADAPTIVE_SOURCE``). This eliminates artificial source spikes and avoids
  distorting integrated power/particle diagnostics such as
  :math:`P_{\mathrm{SOL}}` and :math:`\tau_E`.
* **Scharfetter–Gummel FVM convection and Pereverzev–Corrigan cancellation**:
  The finite-volume discretization now uses unconditionally stable exponential
  Péclet-number weighting for convection-diffusion fluxes and exact discrete
  stationary cancellation for Pereverzev–Corrigan stabilization terms across
  uniform and stretched grids.
* **Flux-surface-averaged Ohmic heating**: Ohmic power deposition now evaluates
  exact flux-surface-averaged Joule heating :math:`\langle \mathbf{j}_\Omega
  \cdot \mathbf{E} \rangle = \sigma_\parallel \langle \mathbf{E} \cdot
  \mathbf{B} \rangle^2 / \langle B^2 \rangle` (at fixed toroidal flux) rather
  than v1's approximation
  :math:`|j_{\mathrm{tor,total}} (\partial \psi / \partial t) / (2\pi R)|`.
* **Isolated current-diffusion scaling**:
  ``numerics['current_diffusion_time_reduction_factor']`` scales only the
  current-diffusion timescale in the :math:`\psi` equation without unphysically
  modifying parallel conductivity :math:`\sigma_\parallel`, Ohmic heating, loop
  voltage, or the neoclassical Ware pinch (unlike v1's
  ``resistivity_multiplier``).
* **Analytical volume-conserving sawtooth redistribution**: The sawtooth crash
  model now uses smoothstep + :math:`C^1` cubic Hermite redistribution with
  analytical conservation of volume-integrated particles, thermal energy, and
  toroidal current across shaped flux surfaces.
* **Transport, neoclassical, current-drive, and radiation fixes**: Fixes have
  been applied to the Angioni–Sauter neoclassical particle flux sign and
  trapped-particle terms, the :math:`E \times B` shear rate sign and units in
  turbulent transport models, the :math:`1/B_0` factor in ECCD parallel current
  density ``j_parallel``, and Mavrin radiation handling so impurity
  Bremsstrahlung is not subtracted when ``impurity_radiation`` has
  ``mode='zero'``.

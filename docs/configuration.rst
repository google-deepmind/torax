.. include:: links.rst

.. _configuration:

Simulation input configuration
##############################

Jump to :ref:`config_details` for an immediate overview of all configuration
parameters.

General notes
=============

TORAX's configuration system allows for fine-grained control over various
aspects of the simulation. This configuration can be represented as a nested
dictionary, where the top level keys are:

* :ref:`profile_conditions <profile_conditions>`: Configures boundary
  conditions, initial conditions, and prescribed time-dependence of temperature,
  density, and current.
* :ref:`numerics <numerics_dataclass>`: Configures simulation control such as
  equations being solved and constant numerical parameters.
* :ref:`plasma_composition <plasma_composition_config>`: Configures the
  distribution of ion species.
* :ref:`geometry <geometry_doc>`: Configures geometry setup and constructs the
  Geometry object.
* :ref:`pedestal <pedestal_config>`: Configures the pedestal for the
  simulation.
* :ref:`edge <edge_models>`: Selects and configures the edge physics model.
* :ref:`sources <sources_config>`: Selects and configures parameters of the
  various heat source, particle source, and non-inductive current models.
* :ref:`solver <solver_config>`: Selects and configures the PDE solver.
* :ref:`transport <transport_config>`: Selects and configures the transport
  model, and constructs the TransportModel object.
* :ref:`mhd <mhd_config>`: Selects and configures the MHD models. Currently
  only a sawtooth model is implemented.
* :ref:`time_step_calculator <time_step_calculator_config>`: Selects the method
  used to calculate the timestep ``dt``.
* :ref:`neoclassical <neoclassical_config>`: Selects and configures the
  neoclassical physics model.
* :ref:`restart <restart_config>`: Configures optional restart behavior of the
  simulation.

This configuration dictionary is converted internally to a Pydantic
|torax.ToraxConfig| model via ``torax.ToraxConfig.from_dict(config_dict)``.

Configuration dictionaries are typically defined in a Python file, and used via:

.. code-block:: python

    run_torax --config='path/to/config.py'

The file ``config.py`` must have the config dictionary defined as a global
variable named ``CONFIG``. See ``examples/iterhybrid_rampup.py`` in
|torax/examples/| for an example.

See :ref:`structure` for more information on TORAX simulation objects.
Further details on the internals of the configuration dictionary are found
in :ref:`config_details`.

For various definitions, see :ref:`glossary`.

Time-dependence and parameter interpolation
===========================================
Some TORAX parameters are allowed to vary over time, and are labelled with
:ref:`time-varying-scalar <time_varying_scalars>` and
:ref:`time-varying-array <time_varying_arrays>` in :ref:`config_details`
below.

.. _time_varying_scalars:

Time-varying scalars
--------------------
The following inputs are valid for **time-varying-scalar** parameters:

* A scalar (integer, float, or boolean). The parameter is then not
  time-dependent.
* A time-series ``dict`` with ``{time: value}`` pairs.
* A tuple ``(time_array, value_array)``, where ``time_array`` is a 1D array of
  times, and ``value_array`` is a 1D array of values of equal length. The 1D
  arrays can be a NumPy arrays, lists or tuples.
* A ``xarray.DataArray`` of the form
  ``xarray.DataArray(data=value_array, coords={'time': time_array})``.

Note that times do not need to be sorted in order of time. Ordering is carried
out internally.

For each evaluation of the TORAX solver (PDE solver), time-dependent variables
are interpolated at both time :math:`t` and time :math:`t+dt`.

interpolation methods
^^^^^^^^^^^^^^^^^^^^^
Currently, two interpolation modes are supported:

* ``'PIECEWISE_LINEAR'``: linear interpolation of the input time-series
  (default).
* ``'STEP'``: stepwise change in values at each time value in the time-series
  (for :math:`t \in [t_k, t_{k+1})`, the value is :math:`y_k`).

The interpolation method can be specified by giving a tuple of the form
``(inputs, interpolation_mode)``, where ``inputs`` is one of the input
specifications listed in the previous section. Specifying no interpolation mode
is equivalent to ``(inputs, 'PIECEWISE_LINEAR')``.

Examples
^^^^^^^^
1. Define a time-dependent total current :math:`I_{p,tot}` with piecewise linear
interpolation, from :math:`t=10` to :math:`t=100`. :math:`I_{p,tot}` rises from
2MA to 15MA, and then stays flat due to constant extrapolation beyond the last
time value.

.. code-block:: python

  I_p = ({10: 2.0e6, 100: 15.0e6}, 'PIECEWISE_LINEAR')

or more simply, taking advantage of the default.

.. code-block:: python

    I_p = {10: 2.0e6, 100: 15.0e6}

2. Define a time-dependent internal boundary condition for ion temperature,
``T_i_ped``, with stepwise changes, starting at :math:`1~keV` at :math:`t=2s`,
transitioning to :math:`3~keV` at :math:`t=8s`, and back down to
:math:`1~keV` at :math:`t=20s`:

.. code-block:: python

  T_i_ped= ({2: 1.0, 8: 3.0, 20: 1.0}, 'STEP')


.. _time_varying_arrays:
.. _Time-varying arrays:

Time-varying arrays
-------------------
Parameters marked as **time-varying-array** are interpolated on a grid
(:math:`t`, :math:`\hat{\rho}`). For each time point, an array of values is
defined on the spatial :math:`\hat{\rho}` grid.

**time-varying-array** parameters can be defined using either a nested
dictionary, or in the form of arrays (represented as a ``xarray.DataArray``
object or a ``tuple`` of arrays).

**Note**: :math:`\hat{\rho}` is normalized and will take values between 0 and 1.

In the case of non-evolving (prescribed) parameters for each evaluation of the
TORAX solver (PDE solver), time-dependent variables are interpolated first along
the :math:`\hat{\rho}` axis at the cell grid centers and then linearly
interpolated in time at both time :math:`t` and time :math:`t+dt`.

For :math:`t` greater than or less than the largest or smallest defined time
then the interpolation scheme will be applied from the closest time value.

Using a nested dictionary
^^^^^^^^^^^^^^^^^^^^^^^^^
This is of the form:

.. code-block:: python

  {time_1: {rho_11: value_11, rho_12: value_12, ...}, time_2: ...}

At each ``time_i``, we have a radial profile composed of  ``{rho: value}``
pairs. The ordering of the dict does not matter.


Shortcuts:

* Passing a single float value is interpreted as defining a constant profile for
  all times. For example ``T_i: 6.0`` would be equivalent to passing in
  ``T_i: {0.0: {0.0: 6.0}}``.

* Passing a single dict (instead of dict of dicts) is a shortcut for defining
  the rho profile for :math:`t=0.0`. For example,
  ``T_i: {0.0: 18.0, 0.95: 5.0, 1.0: 0.2}`` is a shortcut for
  ``T_i: {0.0: {0: 18.0, 0.95: 5.0, 1.0: 0.2}}`` where :math:`t=0.0` is
  arbitrary (due to constant extrapolation for any input :math:`t=0.0`).

Examples:

1. Define an initial profile (at :math:`t=0.0`) for :math:`T_{i}` with a
pedestal.

.. code-block:: python

  T_i = {0.0: {0.0: 15.0, 0.95: 3.0, 1.0: 1.0}}

Note: due to constant extrapolation, the ``t=0.0`` here is an arbitrary number
and could be anything.

2. Define a time-dependent :math:`T_{i}` profile initialised with a pedestal
and, if the ion equation is not being evolved by the PDE, to have a prescribed
time evolution which decays to a constant :math:`T_{i}=1` by :math:`t=80.0`.

.. code-block:: python

  T_i = {0.0: {0.0: 15.0, 0.95: 3.0, 1.0: 1.0}, 80: 1.0}


Using arrays
^^^^^^^^^^^^
This can be a tuple of arrays ``(time_array, rho_norm_array, values_array)``, or
equivalently an ``xarray.DataArray`` object of the form:

.. code-block:: python

  xarray.DataArray(
    data=values_array,
    coords={'time': time_array, 'rho_norm': rho_norm_array}
  )


All arrays can be represented as NumPy arrays or lists.

* ``time_array`` is a 1D array of times.
* ``values_array`` is a 2D array of shape ``(len(time_array), num_values)``.
* ``rho_norm_array`` either a 1D array of shape ``(num_values,)``, or a 2D array
  of shape ``(len(time_array), num_values)``.

Shortcuts:

* ``(rho_norm_array, values_array)``: constant in time profile, useful for
  defining an initial condition or a constant profile. Note that both arrays
  are now 1D arrays.


Specifying interpolation methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
By default, piecewise linear interpolation is used to interpolate values both in
time and in :math:`\hat{\rho}`. To specify a different interpolation method:

.. code-block:: python

  (time_varying_array_value, {'time_interpolation_mode': 'STEP',
  'rho_interpolation_mode': 'PIECEWISE_LINEAR'})

where ``time_varying_array_value`` is any of the above inputs
(nested dictionary, arrays, etc.).

Currently two interpolation modes are supported as for **time-varying-scalar**:

* ``'PIECEWISE_LINEAR'``: linear interpolation of the input time-series
  (default).
* ``'STEP'``: stepwise change in values at each time value in the time-series
  (for :math:`t \in [t_k, t_{k+1})`, the value is :math:`y_k`).


.. _config_details:

Detailed configuration structure
================================

Each parameter is listed alongside its expected data type and default value. Any
declared parameter in a run-specific config overrides the default value.

.. toctree::
   :maxdepth: 1

   configuration/profile_conditions
   configuration/numerics
   configuration/plasma_composition
   configuration/pedestal
   configuration/edge
   configuration/geometry
   configuration/transport
   configuration/sources
   configuration/mhd
   configuration/solver
   configuration/time_step_calculator
   configuration/neoclassical
   configuration/restart


Additional Notes
================

.. _trigger_recompilation:

What triggers a recompilation?
------------------------------

Most config options can be changed without recompiling the simulation code.

Some define the fundamental structure of the simulation and require JAX
recompilation if changed. Examples include the number of grid points or the
choice of transport model. A partial list is provided below.

* ``CONFIG['geometry']['nrho']``
* ``CONFIG['numerics']['evolve_ion_heat']``
* ``CONFIG['numerics']['evolve_electron_heat']``
* ``CONFIG['numerics']['evolve_current']``
* ``CONFIG['numerics']['evolve_density']``
* ``CONFIG['transport']['model_name']``
* ``CONFIG['solver']['solver_type']``
* ``CONFIG['time_step_calculator']['calculator_type']``
* ``CONFIG['sources']['source_name']['is_explicit']``
* ``CONFIG['sources']['source_name']['mode']``

In the future we aim to provide more transparency at the config level for which
configuration options recompilation is required.

Using IMAS input data
---------------------

Loading Geometry
^^^^^^^^^^^^^^^^

The geometry in TORAX can be constructed from IMAS equilibium IDSs. The specific
arguments to load an equilibrium IDS with TORAX are specified in
:ref:`geometry_doc`.

There are three main methods to load IMAS equilibrium:

* Using IMAS netCDF file (imas_filepath).
  This is the main method as it does not require the optional dependency to
  imas-core. The path of the file can then be provided in the config to run
  TORAX with this geometry.

* Using IMAS uri (imas_uri).
  This method does require the imas-core dependency. It loads the
  equilibrium data from the given IDS with a backend of choice.

* Providing the equilibium IDS on the fly (equilibrium_object).
  Using this method the IDS can be provided externally or pre-loaded.

Loading Profiles
^^^^^^^^^^^^^^^^

Profiles in TORAX can be read from any IMAS core_profiles or plasma_profiles IDS
saved in Data Dictionary version 4.0.0 or newer.
If the IDS is stored in an IMAS db or in a netCDF file it can be loaded using
the loader function ``load_imas_data`` from |imas_loader|.
It can then be loaded programmatically in the ``CONFIG`` by constructing a nested
dictionary with the ``profile_conditions_from_imas`` and
``plasma_composition_from_imas`` functions from |core_profiles_input_imas|. The
functions returns a dictionary whose structure fits the schema of
:ref:`profile_conditions <profile_conditions>` or
:ref:`plasma_composition <plasma_composition_config>` and can be
programmatically loaded into a ``CONFIG`` with standard dictionary manipulation.

An example on how to inject the IMAS conditions into the config can be found in
the test file ``imas_tools/input/tests/core_profiles_test.py``.
It is possible to directly input all of profile_conditions from an IDS into the
config, or to only load specific values of it. It is also possible to combine
profiles from different IDSs and their time slices can be shifted to correct
offsets between different IDSs or to match the config t_initial.


Config example
==============

An example configuration dict, corresponding to a non-rigorous demonstration
mock-up of a time-dependent ITER hybrid scenario rampup (presently with a fixed
CHEASE geometry), is shown below. The configuration file is also available in
``torax/examples/iterhybrid_rampup.py``.

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
          'initial_psi_source': 'geometry',
          # initial condition ion temperature for r=0 and r=a_minor
          'T_i': {0.0: {0.0: 6.0, 1.0: 0.1}},
          'T_i_right_bc': 0.1,  # boundary condition ion temp for r=a_minor
          # initial condition electron temperature between r=0 and r=a_minor
          'T_e': {0.0: {0.0: 6.0, 1.0: 0.1}},
          'T_e_right_bc': 0.1,  # boundary condition electron temp for r=a_minor
          'n_e_right_bc_is_f_gw': True,
          'n_e_right_bc': {0: 0.1, 80: 0.3},
          'n_e_bar_is_f_gw': True,
          'n_e_bar': 1,
          'n_e': {0: {0.0: 1.5, 1.0: 1.0}},  # Initial electron density profile
      },
      'numerics': {
          'evolve_ion_heat': True,
          'evolve_electron_heat': True,
          'evolve_current': True,
          'evolve_density': True,
      },
      'geometry': {
          'geometry_type': 'chease',
          'geometry_file': 'iterhybrid.mat2cols',
          'R_major': 6.2,
          'a_minor': 2.0,
          'B_0': 5.3,
      },
      'neoclassical': {
          'bootstrap_current': {
              'bootstrap_multiplier': 1.0,
          },
      },
      'sources': {
          'generic_current': {
              'fraction_of_total_current': 0.15,
              'gaussian_width': 0.075,
              'gaussian_location': 0.36,
          },
          'pellet': {
              'S_total': 0.0e22,
              'pellet_width': 0.1,
              'pellet_deposition_location': 0.85,
          },
          'generic_heat': {
              'gaussian_location': 0.12741589640723575,
              'gaussian_width': 0.07280908366127758,
              # total heating (with a rough assumption of radiation reduction)
              'P_total': 20.0e6,
              'electron_heat_fraction': 1.0,
          },
          'fusion': {},
          'ei_exchange': {},
      },
      'transport': {
          'model_name': 'qlknn',
          'chi_min': 0.05,
          'chi_max': 100,
          'D_e_min': 0.05,
          'D_e_max': 50,
          'V_e_min': -10,
          'V_e_max': 10,
          'smoothing_width': 0.1,
          'DV_effective': True,
          'avoid_big_negative_s': True,
          'An_min': 0.05,
          'ITG_flux_ratio_correction': 1,
      },
      'pedestal': {
          'structure_model': {
              'model_name': 'set_T_ped_n_ped',
              'T_i_ped': 1.0,
              'T_e_ped': 1.0,
              'n_e_ped': {0: 0.3, 80: 0.7},
              'rho_norm_ped_top': 0.95,
              'n_e_ped_is_f_gw': True,
          },
      },
      'solver': {
          'solver_type': 'newton_raphson',
          'use_fixed_point': True,
          'n_fixed_point_iterations': 10,
          'chi_pereverzev': 30,
          'D_pereverzev': 15,
          'use_pereverzev': True,
          'log_iterations': False,
      },
      'time_step_calculator': {
          'calculator_type': 'fixed',
          't_final': 80,
          'dt': 2,
          'dt_backtracking_factor': 3,
      },
  }

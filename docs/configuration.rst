.. _configuration:

Simulation input configuration
##############################

Jump to :ref:`config_details` for an immediate overview of all configuration parameters.

General notes
=============

TORAX's configuration system allows for fine-grained control over various aspects of the simulation.
The top layer ``config`` file is passed as input into a simulation, and contains a dictionary with the following keys:

* **plasma_composition**: Configures the distribution of ion species.
* **profile_conditions**: Configures boundary conditions, initial conditions, and prescribed time-dependence of temperature, density, and current.
* **numerics**: Configures time stepping settings and numerical parameters.
* **geometry**: Configures geometry setup and constructs the Geometry object.
* **transport**: Selects and configures the transport model, and constructs the TransportModel object.
* **sources**: Selects and configures parameters of the various heat source, particle source, and non-inductive current models.
* **solver**: Selects and configures the PDE solver.
* **time_step_calculator**: Selects the method used to calculate the timestep `dt`.

This configuration structure maps to objects instantiated and manipulated within TORAX.
See :ref:`structure` for more information on TORAX simulation objects.
Further details on the internals of the configuration data structures are found in :ref:`config_details`.

For various definitions, see :ref:`glossary`.

Time dependence and parameter interpolation
===========================================
Some TORAX parameters are allowed to temporally vary. Parameters where time dependence is enabled are
are labelled with **time-varying-scalar** and **time-varying-array** in :ref:`config_details`.

Time-varying scalars
--------------------
For fields labelled with **time-varying-scalar** time dependence is set by assigning a dict to the parameter,
instead of a single value. The dict defines a time-series with ``{time: value}`` pairs.
The keys do not need to be sorted in order of time. Ordering is carried out internally.
For each evaluation of the TORAX solver (PDE solver), time-dependent variables
are interpolated at both time :math:`t` and time :math:`t+dt`.
There are two interpolation modes:

* **PIECEWISE_LINEAR**: linear interpolation of the input time-series (default)
* **STEP**: stepwise change in values following each traversal above a time value in the time-series.

The following inputs are valid for **time-varying-scalar** parameters:

* Single integer, float, or boolean. The parameter is then not time dependent
* A time-series dict with ``{time: value}`` pairs, using the default ``interpolation_mode='PIECEWISE_LINEAR'``.
* A tuple with ``(time-series, value-series)``. The time-series is a 1D array of times, and the value-series is a 1D array of values and the dimensions of both must match.
* A ``xarray.DataArray`` with a single coordinate and a 1D value array.

Examples:

1. Define a time-dependent total current :math:`Ip_{tot}` with piecewise linear interpolation,
from :math:`t=10` to :math:`t=100`. :math:`Ip_{tot}` rises from 2 to 15, and then stays flat
due to constant extrapolation beyond the last time value.

.. code-block:: python

  I_total = ({10: 2.0, 100: 15.0}, 'PIECEWISE_LINEAR')

or more simply, taking advantage of the default.

.. code-block:: python

    I_total = {10: 2.0, 100: 15.0}

2. Define a time dependent internal boundary condition for ion temperature, ``Tiped``, with stepwise changes,
starting at :math:`1~keV`` at :math:`t=2s`, transitioning to :math:`3~keV`` at :math:`t=8s`, and back down
to :math:`1~keV` at :math:`t=20s`:

.. code-block:: python

  Tiped= ({2: 1.0, 8: 3.0, 20: 1.0}, 'STEP')

To extend configuration parameters where time-dependence is not enabled, to have time-dependence, see :ref:`developer-guides`.

Time-varying arrays
-------------------
Time-varying arrays can be defined using either primitives, an
``xarray.DataArray`` or a ``tuple`` of ``Array``.

Specifying interpolation methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
By default piecewise linear interpolation is used to interpolate values in time.
To specify a different interpolation method, use the following syntax of a tuple
with two elements. The first element in the tuple is the usual value for the
time-varying-array (as defined below), the second value is a dict with keys
``time_interpolation_mode`` and ``rho_interpolation_mode`` and values the
desired interpolation modes.

.. code-block:: python

  (time_varying_array_value, {'time_interpolation_mode': 'STEP', 'rho_interpolation_mode': 'PIECEWISE_LINEAR'})

Currently two interpolation modes are supported:

* ``'STEP'``
* ``'PIECEWISE_LINEAR'``

Using primitives
^^^^^^^^^^^^^^^^

For fields labelled with **time-varying-array** time dependence is set by assigning a dict of dicts to the parameter.

The outer dict defines a time-series with ``{time: value}`` pairs.
The ``value`` itself is interpreted as a radial profile, being made up of {rho: value} pairs.
It behaves similarly to the **time-varying-scalar** but any interpolation will happen along the
:math:`\hat{\rho}` axis and can take any of the formats defined for a **time-varying-scalar** above.

Note: :math:`\hat{\rho}` is normalized and will take values between 0 and 1.

None of the keys need to be sorted in order of time. Ordering is carried out internally.
In the case of non-evolving parameters for each evaluation of the TORAX solver (PDE solver), time-dependent variables
are interpolated first along the :math:`\hat{\rho}` axis at the cell grid centers and then linearly interpolated in time
at both time :math:`t` and time :math:`t+dt`..

For :math:`t` greater than or less than the largest or smallest defined time then the interpolation scheme
will be applied from the closest time value.

Shortcuts:

Passing a single float value is interpreted as defining a constant profile for all times.
For example ``Ti: 6.0`` would be equivalent to passing in ``Ti: {0.0: {0.0: 6.0}}``.

Passing a single dict (instead of dict of dicts) is a shortcut for defining the rho profile
for :math:`t=0.0`. For example ``Ti: {0.0: 18.0, 0.95: 5.0, 1.0: 0.2}`` is a shortcut for
``Ti: {0.0: {0: 18.0, 0.95: 5.0, 1.0: 0.2}}`` where :math:`t=0.0` is arbitrary
(due to constant extrapolation for any input :math:`t=0.0`).


Examples:

1. Define an initial profile (at :math:`t=0.0`) for :math:`T_{i}` with a pedestal.

.. code-block:: python

  Ti = {0.0: {0.0: 15.0, 0.95: 3.0, 1.0: 1.0}}

Note: due to constant extrapolation the t=0.0 here is an arbitrary number and could be anything.

2. Define a time-dependent :math:`T_{i}` profile initialised with a pedestal and, if the ion equation is not being
evolved by the PDE, to have a prescribed time evolution which decays to a
constant :math:`T_{i}=1` by :math:`t=80.0`.

.. code-block:: python

  Ti = {0.0: {0.0: 15.0, 0.95: 3.0, 1.0: 1.0}, 80: 1.0}

Using ``xarray.DataArray``
^^^^^^^^^^^^^^^^^^^^^^^^^^
If a ``xarray.DataArray`` is specified then it is expected to have a
``time`` and ``rho_norm`` coordinate. The values of the data array are the values
at each time and rho_norm.

Using ``tuple`` of ``Array``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If a ``tuple`` of ``Array`` is used, the tuple must have structure of,
``(time_array, rho_norm_array, values_array)`` or ``(rho_norm_array, values_array)``.
The latter is a useful shortcut for defining an initial condition or a constant profile.

In the case of ``(time_array, rho_norm_array, values_array)``:
``time_array`` and ``rho_norm_array`` are expected to map to 1D array values and
represent the time and rho_norm coordinates.
``values_array`` is expected to map to a 2D array with shape
``(len(time_array), len(rho_norm_array))`` and represent the values at the given
time and rho_norm.

In the case of ``(rho_norm_array, values_array)``:
``rho_norm_array`` is expected to map to a 1D array values and represent the
rho_norm coordinates.
``values_array`` is expected to map to a 1D array with shape
``len(rho_norm_array)`` and represent the values at the given rho_norm.

.. _config_details:

Detailed configuration structure
================================

Data types and default values are written in parentheses. Any declared parameter in a run-specific config, overrides the default value.

runtime_params
--------------

plasma_composition
^^^^^^^^^^^^^^^^^^

Defines the distribution of ion species.  The keys and their meanings are as follows:

``main_ion`` (str or dict = ``{'D': 0.5, 'T': 0.5}``)
  Specifies the main ion species.

  *   If a string, it represents a single ion species (e.g., ``'D'`` for deuterium, ``'T'`` for tritium, ``'H'`` for hydrogen). See below for the full list of supported ions.
  *   If a dict, it represents a mixture of ion species with given fractions. By `mixture`, we mean
      key value pairs of ion symbols and fractional concentrations, which must sum to 1 within a tolerance of 1e-6.
      The effective mass and charge of the mixture is the weighted average of the species masses and charges.
      The fractions can be time-dependent, i.e. are **time-varying-scalar**. The ion mixture API thus
      supports features such as time varying isotope ratios.

``impurity`` (str or dict = ``'Ne'``), **time-varying-scalar**
  Specifies the impurity species, following the same syntax as ``main_ion``. A single effective impurity species
  is currently supported, although multiple impurities can still be defined as a mixture.

``Zeff`` (float = 1.0), **time-varying-array**
  Plasma effective charge, defined as :math:`Z_{eff}=\sum_i Z_i^2 \hat{n}_i`, where :math:`\hat{n}_i` is
  the normalized ion density :math:`n_i/n_e`. For a given :math:`Z_{eff}` and impurity charge states,
  a consistent :math:`\hat{n}_i` is calculated, with the appropriate degree of main ion dilution.

``Zi_override`` (float, optional = None), **time-varying-scalar**
  An optional override for the main ion's charge (Z) or average charge of an ion mixture.
  If provided, this value will be used instead of the Z calculated from the ``main_ion`` specification.

``Ai_override`` (float, optional = None), **time-varying-scalar**
  An optional override for the main ion's mass (A) in amu units or average mass of an IonMixture.
  If provided, this value will be used instead of the A calculated from the ``main_ion`` specification.

``Zimp_override`` (float, optional), **time-varying-scalar**
  As ``Zi_override``, but for the impurity ion. If provided, this value will be used instead of the Z calculated
  from the ``impurity`` specification.

``Aimp_override`` (float, optional), **time-varying-scalar**
  As ``Ai_override``, but for the impurity ion. If provided, this value will be used instead of the A calculated
  from the ``impurity`` specification.

The average charge state of each ion in each mixture is determined by `Mavrin polynomials <https://doi.org/10.1080/10420150.2018.1462361>`_,
which are fitted to atomic data, and in the temperature ranges of interest in the tokamak core,
are well approximated as 1D functions of electron temperature. All ions with atomic numbers below
Carbon are assumed to be fully ionized.

Examples
--------

We remind that for all cases below, the impurity density is solely constrained by
the input ``Zeff`` value and the impurity charge state, presently assumed to be fully ionized.
Imminent development will support temperature-dependent impurity average charge states,

* Pure deuterium plasma:

  .. code-block:: python

    'plasma_composition': {
        'main_ion': 'D',
        'impurity': 'Ne',  # Neon
        'Zeff': 1.5,
    }

* 50-50 DT ion mixture:

  .. code-block:: python

    'plasma_composition': {
        'main_ion': {'D': 0.5, 'T': 0.5},
        'impurity': 'Be',  # Beryllium
        'Zeff': 1.8,
    }

* Time-varying DT ion mixture:

  .. code-block:: python

    'plasma_composition': {
      'main_ion': {
        'D': {0.0: 0.1, 5.0: 0.9},  # D fraction from 0.1 to 0.9
        'T': {0.0: 0.9, 5.0: 0.1},  # T fraction from 0.9 to 0.1
      },
      'impurity': 'W',  # Tungsten
      'Zeff': 2.0,
    }

Allowed ion symbols
-------------------

The following ion symbols are recognized for ``main_ion`` and ``impurity`` input fields.

  *   H  (Hydrogen)
  *   D  (Deuterium)
  *   T  (Tritium)
  *   He3 (Helium-3)
  *   He4 (Helium-4)
  *   Li (Lithium)
  *   Be (Beryllium)
  *   C (Carbon)
  *   N (Nitrogen)
  *   O (Oxygen)
  *   Ne (Neon)
  *   Ar (Argon)
  *   Kr (Krypton)
  *   Xe (Xenon)
  *   W (Tungsten)

Profile conditions
^^^^^^^^^^^^^^^^^^

Configures boundary conditions, initial conditions, and prescribed time-dependence of temperature, density, and current.

``I_total`` (float = 15.0), **time-varying-scalar**
  Total plasma current in MA. Boundary condition for the :math:`\psi` equation.

``T_i_right_bc`` (float | None [default]), **time-varying-scalar**
  Ion temperature boundary condition at :math:`\hat{\rho}=1` in units of keV.
  If not provided or set to `None` then the boundary condition is taken from the
  :math:`\hat{\rho}=1` value derived from the provided `Ti` profile.

``T_e_right_bc`` (float | None [default]), **time-varying-scalar**
  Electron temperature boundary condition at :math:`\hat{\rho}=1`, in units of keV.
  If not provided or set to `None` then the boundary condition is taken from the
  :math:`\hat{\rho}=1` value derived from the provided `Te` profile.

``Ti`` (dict = {0: {0: 15.0, 1: 1.0}}), **time-varying-array**
  Initial and (if not time evolving) prescribed :math:`\hat{\rho}` ion temperature, in units of keV.

  Note: For a given time ``t``, ``Ti[t]`` is used to define interpolation along :math:`\hat{\rho}` at cell centers.
  If `T_i_right_bc=None`, the boundary condition at :math:`\hat{\rho}=1`
  is taken from the :math:`\hat{\rho}=1` value derived from the provided `Ti` profile.
  Note that if the `Ti` profile does not contain a :math:`\hat{\rho}=1` point
  for all provided times, an error will be raised.

``Te`` (dict = {0: {0: 15.0, 1: 1.0}}), **time-varying-array**
  Initial and (if not time evolving) prescribed :math:`\hat{\rho}` electron temperature, in units of keV.

  Note: For a given time ``t``, ``Te[t]`` is used to define interpolation along :math:`\hat{\rho}` at cell centers.
  If `T_e_right_bc=None`, the boundary condition at :math:`\hat{\rho}=1`
  is taken from the :math:`\hat{\rho}=1` value derived from the provided `Te` profile.
  Note that if the `Te` profile does not contain a :math:`\hat{\rho}=1` point,
  for all provided times, an error will be raised.

``psi`` (dict | None [default]), **time-varying-array**
  Initial poloidal flux. If not provided the initial psi will be calculated from either the geometry
  or the "nu formula".


``n_e`` (dict = {0: {0: 1.5, 1: 1.0}}), **time-varying-array**
  Electron density profile.

  If ``dens_eq==True`` (see :ref:`numerics_dataclass`), then time dependent ``n_e`` is ignored, and only the initial value is used.

  If ``n_e_bound_right=None``, the boundary condition at :math:`\hat{\rho}=1`
  is taken from the :math:`\hat{\rho}=1` value derived from the provided ``n_e`` profile.
  Note that if the ``n_e`` profile does not contain a :math:`\hat{\rho}=1` point
  for all provided times, an error will be raised.

``normalize_to_nbar`` (bool = True)
  If True, then the electron density profile is normalized to have the desired line averaged density
  :math:`\bar{n}`.

``nbar`` (float = 0.5), **time-varying-scalar**
  Line averaged density. In units of reference density ``nref`` (see :ref:`numerics_dataclass`) if ``n_e_is_fGW==False``.
  In units of Greenwald fraction :math:`n_{GW}` if ``n_e_is_fGW==True``. :math:`n_{GW}=I_p/(\pi a^2)` in units of :math:`10^{20} m^{-3}`, where :math:`a`
  is the tokamak minor radius in meters, and :math:`I_p` is the plasma current in MA.

``n_e_is_fGW`` (bool = True)
  Toggles units of ``nbar``.

``n_e_bound_right`` (float = 0.5), **time-varying-scalar**
  Density boundary condition at :math:`\hat{\rho}=1`. In units of ``nref`` if ``n_e_bound_right_is_fGW==False``.
  In units of Greenwald fraction :math:`n_{GW}` if ``n_e_bound_right_is_fGW==True``.
  If not provided or set to `None` then the boundary condition is taken from the
  :math:`\hat{\rho}=1` value derived from the provided `n_e` profile.

``n_e_bound_right_is_fGW`` (bool = False)
  Toggles units of ``n_e_bound_right``.

``nu`` (float = 3.0)
  Peaking coefficient of initial current profile: :math:`j = j_0(1 - \hat{\rho}^2)^\nu`. :math:`j_0` is calculated
  to be consistent with a desired total current. Only used if ``initial_psi_from_j==True``, otherwise the ``psi`` profile from the geometry file is used.

``initial_j_is_total_current`` (bool = False)
  Toggles the interpretation of :math:`j` above. If true, then :math:`j` is the total current.
  If false, then :math:`j` is Ohmic current, with :math:`I_{ohm} = I_{tot} - I_{ni}`, where :math:`I_{ni}` is the total non-inductive current
  calculated upon initialization.

``initial_psi_from_j`` (bool = False)
  Toggles if the initial ``psi`` (:math:`\psi`) calculation is based on the "nu" current formula, or from the ``psi``
  available in the numerical geometry file. This setting is ignored for the ad-hoc circular geometry option, which has no numerical geometry, and thus the
  initial ``psi`` is always calculated from the "nu" current formula.

.. _numerics_dataclass:

numerics
^^^^^^^^

Configures simulation control such as time settings and timestep calculation, equations being solved, constant numerical variables.

``t_initial`` (float = 0.0)
  Simulation start time, in units of seconds.

``t_final`` (float = 5.0)
  Simulation end time, in units of seconds.

``exact_t_final`` (bool = False)
  If True, ensures that the simulation end time is exactly ``t_final``, by adapting the final ``dt`` to match.

``maxdt`` (float = 1e-1)
  Maximum timesteps allowed in the simulation. This is only used with the ``chi_time_step_calculator`` time_step_calculator.

``mindt`` (float = 1e-8)
  Minimum timestep allowed in simulation.

``dtmult`` (float = 9.0)
  Prefactor in front of ``chi_timestep_calculator`` base timestep :math:`dt_{base}=\frac{dx^2}{2\chi}` (see :ref:`time_step_calculator`).
  In most use-cases with implicit solution methods, ``dtmult`` can be increased further above the conservative default.

``fixed_dt`` (float = 1e-2)
  Timestep used for ``fixed_time_step_calculator`` (see :ref:`time_step_calculator`).

``ion_heat_eq`` (bool = True)
  Solve the ion heat equation in the time-dependent PDE.

``el_heat_eq`` (bool = True)
  Solve the electron heat equation in the time-dependent PDE.

``current_eq`` (bool = False)
  Solve the current diffusion equation (evolving :math:`\psi`) in the time-dependent PDE.

``dens_eq`` (bool = False)
  Solve the electron density equation in the time-dependent PDE.

``resistivity_mult`` (float = 1.0)
  1/multiplication factor for :math:`\sigma` (conductivity) to reduce the current
  diffusion timescale to be closer to the energy confinement timescale, for testing purposes.

``largeValue_T`` (float = 1e10)
  Prefactor for adaptive source term for setting temperature internal boundary conditions.

``largeValue_n`` (float = 1e8)
  Prefactor for adaptive source term for setting density internal boundary conditions.

``nref`` (float = 1e20)
  Reference density value for normalizations.

output_dir
^^^^^^^^^^

``output_dir`` (str)
  Optional string containing the file directory where the simulation outputs
  will be saved. If not provided, this will default to
  ``'/tmp/torax_results_<YYYYMMDD_HHMMSS>/'``

.. _time_step_calculator:

pedestal
--------
In TORAX we aim to support different models for computing the pedestal width,
and electron density, ion temperature and electron temperature at the pedestal
top. These models will only be used if the ``set_pedestal`` flag is set to True.

The model can be configured by setting the ``pedestal_model`` key in the
``pedestal`` section of the configuration. If this field is not set, then
the default model is ``no_profile``.

``set_pedestal`` (bool = False), **time-varying-scalar**
  If True use the configured pedestal model to set internal boundary conditions. Do not set internal boundary conditions if False.
  Internal boundary conditions are set using an adaptive localized source term. While a common use-case is to mock up a pedestal, this feature
  can also be used for L-mode modeling with a desired internal boundary condition below :math:`\hat{\rho}=1`.

The following models are currently supported:

``no_profile``
^^^^^^^^^^^^^
No pedestal profile is set. This is the default option and the equivalent of
setting ``set_pedestal`` to False.

set_tped_nped
^^^^^^^^^^^^^
Directly specify the pedestal width, electron density, ion temperature and
electron temperature.

``neped`` (float = 0.7) **time-varying-scalar**
  Electron density at the pedestal top.
  In units of reference density if ``neped_is_fGW==False``. In units of
  Greenwald fraction if ``neped_is_fGW==True``.

``neped_is_fGW`` (bool = False) **time-varying-scalar**
  Toggles units of ``neped``.

``Tiped`` (float = 5.0) **time-varying-scalar**
  Ion temperature at the pedestal top in units of keV.

``Teped`` (float = 5.0) **time-varying-scalar**
  Electron temperature at the pedestal top in units of keV.

``rho_norm_ped_top`` (float = 0.91) **time-varying-scalar**
  Location of pedestal top, in units of :math:`\hat{\rho}`.

set_pped_tpedratio_nped
^^^^^^^^^^^^^^^^^^^^^^^
Set the pedestal width, electron density and ion temperature by providing the
total pressure at the pedestal and the ratio of ion to electron temperature.

``Pped`` (float = 10.0) **time-varying-scalar**
  The plasma pressure at the pedestal in units of :math:`[Pa]`.

``neped`` (float = 0.7) **time-varying-scalar**
  Electron density at the pedestal top.
  In units of reference density if ``neped_is_fGW==False``. In units of Greenwald fraction if ``neped_is_fGW==True``.

``neped_is_fGW`` (bool = False) **time-varying-scalar**
  Toggles units of ``neped``.

``ion_electron_temperature_ratio`` **time-varying-scalar**
  Ratio of the ion and electron temperature at the pedestal.

``rho_norm_ped_top`` (float = 0.91) **time-varying-scalar**
  Location of pedestal top, in units of :math:`\hat{\rho}`.

geometry
--------

``geometry_type`` (str)
  Geometry model used. A string must be provided from the following options.

* ``'circular'``
    An ad-hoc circular geometry model. Includes elongation corrections.
    Not recommended for use apart from for testing purposes.

* ``'chease'``
    Loads a CHEASE geometry file.

* ``'fbt'``
    Loads FBT geometry files.

* ``'eqdsk'``
    Loads a EQDSK geometry file, and carries out the appropriate flux-surface-averages of the 2D poloidal flux.
    Use of EQDSK geometry comes with the following caveat:
    The TORAX EQDSK converter has only been tested against CHEASE-generated EQDSK which is COCOS=2.
    The converter is not guaranteed to work as expected with arbitrary EQDSK input, so please verify carefully.
    Future work will be done to correctly handle EQDSK inputs provided with a specific COCOS value.

Geometry dicts for all geometry types can contain the following additional keys.

``n_rho`` (int = 25)
  Number of radial grid points

``hires_fac`` (int = 4)
  Only used when the initial condition ``psi`` is from plasma current. Sets up a higher resolution mesh
  with ``nrho_hires = nrho * hi_res_fac``, used for ``j`` to ``psi`` conversions.

Geometry dicts for all non-circular geometry types can contain the following additional keys.

``geometry_file`` (str = 'ITER_hybrid_citrin_equil_cheasedata.mat2cols')
  Required for all geometry types except ``'circular'``. Sets the geometry file loaded.
  The geometry directory is set with the ``TORAX_GEOMETRY_DIR`` environment variable. If none is set, then the default is ``torax/data/third_party/geo``.

``geometry_dir`` (str = None)
  Optionally overrides the``TORAX_GEOMETRY_DIR`` environment variable.

``Ip_from_parameters`` (bool = True)
  Toggles whether total plasma current is read from the configuration file, or from the geometry file.
  If True, then the :math:`\psi` calculated from the geometry file is scaled to match the desired :math:`I_p`.

Geometry dicts for analytical circular geometry require the following additional keys.

``Rmaj`` (float = 6.2)
  Major radius (R) in meters.

``Rmin`` (float = 2.0)
  Minor radius (a) in meters.

``B0`` (float = 5.3)
  Vacuum toroidal magnetic field on axis [T].

``kappa`` (float = 1.72)
  Sets the plasma elongation used for volume, area and q-profile corrections.

Geometry dicts for CHEASE geometry require the following additional keys for denormalization.

``Rmaj`` (float = 6.2)
  Major radius (R) in meters.

``Rmin`` (float = 2.0)
  Minor radius (a) in meters.

``B0`` (float = 5.3)
  Vacuum toroidal magnetic field on axis [T].

Geometry dicts for FBT geometry require the following additional keys.

``LY_object`` (dict[str, np.ndarray] | str)
  Sets a single-slice FBT LY geometry file to be loaded, or alternatively a dict
  directly containing a single time slice of LY data.

``LY_bundle_object`` (dict[str, np.ndarray] | str)
  Sets the FBT LY bundle file to be loaded, corresponding to multiple time-slices,
  or alternatively a dict directly containing all time-slices of LY data.

``LY_to_torax_times`` (ndarray = None)
  Sets the TORAX simulation times corresponding to the individual slices in the
  FBT LY bundle file. If not provided, then the times are taken from the LY_bundle_file
  itself. The length of the array must match the number of slices in the bundle.

``L_object`` (dict[str, np.ndarray] | str)
  Sets the FBT L geometry file loaded, or alternatively a dict directly containing
  the L data.

Geometry dicts for EQDSK geometry can contain the following additional keys.
It is only recommended to change the default values if issues arise.

``n_surfaces`` (int = 100)
  Number of surfaces for which flux surface averages are calculated.

``last_surface_factor`` (float = 0.99)
  Multiplication factor of the boundary poloidal flux, used for the contour
  defining geometry terms at the LCFS on the TORAX grid. Needed to avoid
  divergent integrations in diverted geometries.

For setting up time-dependent geometry, a subset of varying geometry parameters
and input files can be defined in a ``geometry_configs`` dict, which is a
time-series of {time: {configs}} pairs. For example, a time-dependent geometry
input with 3 time-slices of single-time-slice FBT geometries can be set up as:

.. code-block:: python

  'geometry': {
      'geometry_type': 'fbt',
      'Ip_from_parameters': True,
      'geometry_configs': {
          20.0: {
              'LY_file': 'LY_early_rampup.mat',
              'L_file': 'L_early_rampup.mat',
          },
          50.0: {
              'LY_file': 'LY_mid_rampup.mat',
              'L_file': 'L_mid_rampup.mat',
          },
          100.0: {
              'LY_file': 'LY_endof_rampup.mat',
              'L_file': 'L_endof_rampup.mat',
          },
      },
  },

Alternatively, for FBT data specifically, TORAX supports loading a bundle of LY
files packaged within a single ``.mat`` file using LIUQE meqlpack. This eliminates
the need to specify multiple individual LY files in the ``geometry_configs`` parameter.

To use this feature, set ``LY_bundle_file`` to the corresponding ``.mat`` file containing
the LY bundle. Optionally set ``LY_to_torax_times`` as a NumPy array corresponding to times
of the individual LY slices within the bundle. If not provided, then the times are taken
from the bundle file itself.

Note that ``LY_bundle_file`` cannot coexist with ``LY_file`` or ``geometry_configs`` in the
same configuration, and will raise an error if so.

All file loading and geometry processing is done upon simulation initialization.
The geometry inputs into the TORAX PDE coefficients are then time-interpolated
on-the-fly onto the TORAX time slices where the PDE calculations are done.

transport
---------

Select and configure various transport models. The dictionary consists of keys
common to all transport models, and additional keys pertaining to a specific
transport model.

``transport_model`` (str = 'constant')
  Select the transport model according to the following options:

* ``'constant'``
  Constant transport coefficients
* ``'CGM'``
  Critical Gradient Model
* ``'bohm-gyrobohm'``
  Bohm-GyroBohm model.
* ``'qlknn'``
  A QuaLiKiz Neural Network surrogate model, the default is `QLKNN_7_11 <https://github.com/google-deepmind/fusion_surrogates>`_.
* ``'qualikiz'``
  The `QuaLiKiz <https://gitlab.com/qualikiz-group/QuaLiKiz>`_ quasilinear gyrokinetic transport model.

``chimin`` (float = 0.05)
  Lower allowed bound for heat conductivities :math:`\chi`, in units of :math:`m^2/s`.

``chimax`` (float = 100.0)
  Upper allowed bound for heat conductivities :math:`\chi`, in units of :math:`m^2/s`.

``Demin`` (float = 0.05)
  Lower allowed bound for particle conductivity :math:`D`, in units of :math:`m^2/s`.

``Demax`` (float = 100.0)
  Upper allowed bound for particle conductivity :math:`D`, in units of :math:`m^2/s`.

``Vemin`` (float = -50.0)
  Lower allowed bound for particle convection :math:`V`, in units of :math:`m^2/s`.

``Vemax`` (float = 50.0)
  Upper allowed bound for particle convection :math:`V`, in units of :math:`m^2/s`.

``apply_inner_patch`` (bool = False), **time-varying-scalar**
  If True, set a patch for inner core transport coefficients below `rho_inner`.
  Typically used as an ad-hoc measure for MHD (e.g. sawteeth) or EM (e.g. KBM) transport in the inner-core.

``De_inner``  (float = 0.2), **time-varying-scalar**
  Particle diffusivity value for inner transport patch.

``Ve_inner``  (float = 0.0), **time-varying-scalar**
  Particle convection value for inner transport patch.

``chii_inner``  (float = 1.0), **time-varying-scalar**
  Ion heat conduction value for inner transport patch.

``chie_inner`` (float = 1.0), **time-varying-scalar**
  Electron heat conduction value for inner transport patch.

``rho_inner`` (float = 0.3)
  :math:`\hat{\rho}` below which inner patch is applied.

``apply_outer_patch`` (bool = False), **time-varying-scalar**
  If True, set a patch for outer core transport coefficients above ``rho_outer``.
  Useful for the L-mode near-edge region where models like QLKNN10D are not applicable. Only used if ``set_pedestal==False``.

``De_outer``  (float = 0.2), **time-varying-scalar**
  Particle diffusivity value for outer transport patch.

``Ve_outer``  (float = 0.0), **time-varying-scalar**
  Particle convection value for outer transport patch.

``chii_outer``  (float = 1.0), **time-varying-scalar**
  Ion heat conduction value for outer transport patch.

``chie_outer`` (float = 1.0), **time-varying-scalar**
  Electron heat conduction value for outer transport patch.

``rho_outer`` (float = 0.9)
  :math:`\hat{\rho}` above which outer patch is applied.

``smoothing_sigma`` (float = 0.0)
  Width of HWHM Gaussian smoothing kernel operating on transport model outputs.
  If using the ``QLKNN_7_11`` transport model, the default is set to 0.1

constant
^^^^^^^^

Runtime parameters for the constant chi transport model.

``chii_const`` (float = 1.0), **time-varying-scalar**
  Ion heat conductivity. In units of :math:`m^2/s`.

``chie_const`` (float = 1.0), **time-varying-scalar**
  Electron heat conductivity. In units of :math:`m^2/s`.

``De_const`` (float = 1.0), **time-varying-scalar**
  Electron particle diffusion. In units of :math:`m^2/s`.

``Ve_const`` (float = -0.33), **time-varying-scalar**
  Electron particle convection. In units of :math:`m^2/s`.

CGM
^^^

Runtime parameters for the Critical Gradient Model (CGM).

``alpha`` (float = 2.0)
  Exponent of chi power law: :math:`\chi \propto (R/L_{Ti} - R/L_{Ti_crit})^\alpha`.

``chistiff`` (float = 2.0)
  Stiffness parameter.

``chiei_ratio`` (float = 2.0), **time-varying-scalar**
  Ratio of ion to electron heat conductivity. ITG turbulence has values above 1.

``chi_D_ratio`` (float = 5.0), **time-varying-scalar**
  Ratio of ion heat conductivity to electron particle diffusion.

``VR_D_ratio`` (float = 0.0), **time-varying-scalar**
  Ratio of major radius * electron particle convection to electron particle diffusion.
  Sets the electron particle convection in the model. Negative values will set a peaked
  electron density profile in the absence of sources.

Bohm-GyroBohm
^^^^^^^^^^^^^

Runtime parameters for the Bohm-GyroBohm model.

``chi_e_bohm_coeff`` (float = 8e-5), **time-varying-scalar**
  Prefactor for Bohm term for electron heat conductivity.

``chi_e_gyrobohm_coeff`` (float = 5e-6), **time-varying-scalar**
  Prefactor for GyroBohm term for electron heat conductivity.

``chi_i_bohm_coeff`` (float = 8e-5), **time-varying-scalar**
  Prefactor for Bohm term for ion heat conductivity.

``chi_i_gyrobohm_coeff`` (float = 5e-6), **time-varying-scalar**
  Prefactor for GyroBohm term for ion heat conductivity.

``chi_e_bohm_multiplier`` (float = 1.0), **time-varying-scalar**
  Multiplier for Bohm term for electron heat conductivity. Intended for
  user-friendly default modification.

``chi_e_gyrobohm_multiplier`` (float = 1.0), **time-varying-scalar**
  Multiplier for GyroBohm term for electron heat conductivity. Intended for
  user-friendly default modification.

``chi_i_bohm_multiplier`` (float = 1.0), **time-varying-scalar**
  Multiplier for Bohm term for ion heat conductivity. Intended for
  user-friendly default modification.

``chi_i_gyrobohm_multiplier`` (float = 1.0), **time-varying-scalar**
  Multiplier for GyroBohm term for ion heat conductivity. Intended for
  user-friendly default modification.

``d_face_c1`` (float = 1.0), **time-varying-scalar**
  Constant for the electron diffusivity weighting factor.

``d_face_c2`` (float = 0.3), **time-varying-scalar**
  Constant for the electron diffusivity weighting factor.

qlknn
^^^^^

Runtime parameters for the QLKNN model. These parameters determine which model
to load, as well as model parameters. To determine which model to load,
TORAX uses the following logic:

* If ``model_path`` is provided, then we load the model from this path.
* Otherwise, if ``model_name`` is provided, we load that model from registered
  models in the ``fusion_surrogates`` library.
* If ``model_name`` is not set either, we load the default QLKNN model from
  ``fusion_surrogates`` (currently ``QLKNN_7_11``).

It is recommended to not set ``model_name``,  or
``model_path`` to use the default QLKNN model.

``model_path`` (str = '')
  Path to the model. Takes precedence over ``model_name``.

``model_name`` (str = '')
  Name of the model. Used to select a model from the ``fusion_surrogates`` library.

``coll_mult`` (float = 1.0)
  Collisionality multiplier.
  If using ``QLKNN10D``, the default is 0.25. It is a proxy for the upgraded
  collision operator in QuaLiKiz, in place since ``QLKNN10D`` was developed.

``include_ITG`` (bool = True)
  If True, include ITG modes in the total fluxes.

``include_TEM`` (bool = True)
  If True, include TEM modes in the total fluxes.

``include_ETG`` (bool = True)
  If True, include ETG modes in the total electron heat flux.

``ITG_flux_ratio_correction`` (float = 1.0)
  Increase the electron heat flux in ITG modes by this factor.
  If using ``QLKNN10D``, the default is 2.0. It is a proxy for the impact of the
  upgraded QuaLiKiz collision operator, in place since ``QLKNN10D`` was developed.

``DVeff`` (bool = False)
  If True, use either :math:`D_{eff}` or :math:`V_{eff}` for particle transport. See :ref:`physics_models` for more details.

``An_min`` (float = 0.05)
  :math:`|R/L_{n_e}|` value below which :math:`V_{eff}` is used instead of :math:`D_{eff}`, if ``DVeff==True``.

``avoid_big_negative_s`` (bool = True)
  If True, modify input magnetic shear such that :math:`\hat{s} - \alpha_{MHD} > -0.2` always,
  to compensate for the lack of slab ITG modes in QuaLiKiz.

``smag_alpha_correction`` (bool = True)
  If True, reduce input magnetic shear by :math:`0.5*\alpha_{MHD}` to capture the main impact of
  :math:`\alpha_{MHD}`, which was not itself part of the ``QLKNN`` training set.

``q_sawtooth_proxy`` (bool = True)
  To avoid un-physical transport barriers, modify the input q-profile and magnetic shear for zones where
  :math:`q < 1`, as a proxy for sawteeth. Where :math:`q<1`, then the :math:`q` and :math:`\hat{s}` ``QLKNN`` inputs are clipped to
  :math:`q=1` and :math:`\hat{s}=0.1`.

qualikiz
^^^^^^^^

Runtime parameters for the QuaLiKiz model.

``maxruns`` (int = 2)
  Frequency of full QuaLiKiz contour solutions. For maxruns>1, every maxruns-th
  call will use the full contour integral solution. Other runs will use the previous
  solution as the initial guess for the Newton solver, which is significantly faster.

``numprocs`` (int = 8)
  Number of MPI processes to use for QuaLiKiz.

``coll_mult`` (float = 1.0)
  Collisionality multiplier for sensitivity analysis.

``DVeff`` (bool = False)
  If True, use either :math:`D_{eff}` or :math:`V_{eff}` for particle transport. See :ref:`physics_models` for more details.

``An_min`` (float = 0.05)
  :math:`|R/L_{n_e}|` value below which :math:`V_{eff}` is used instead of :math:`D_{eff}`, if ``DVeff==True``.

``avoid_big_negative_s`` (bool = True)
  If True, modify input magnetic shear such that :math:`\hat{s} - \alpha_{MHD} > -0.2` always,
  to compensate for the lack of slab ITG modes in QuaLiKiz.

``q_sawtooth_proxy`` (bool = True)
  To avoid un-physical transport barriers, modify the input q-profile and magnetic shear for zones where
  :math:`q < 1`, as a proxy for sawteeth. Where :math:`q<1`, then the :math:`q` and :math:`\hat{s}` QuaLiKiz inputs are clipped to
  :math:`q=1` and :math:`\hat{s}=0.1`.

sources
-------

dict with nested dicts containing the runtime parameters of all TORAX heat, particle, and current sources. The following runtime parameters
are common to all sources, with defaults depending on the specific source. See :ref:`physics_models` For details on the source physics models.

Any source which is not explicitly included in the sources dict, is set to zero. To include a source with default
options, the source dict should contain an empty dict. For example, for setting ``ei_exchange``, with default options,
as the only active source in ``sources``, set:

.. code-block:: python

    'sources': {
        'ei_exchange': {},
    }

The configurable runtime parameters of each source are as follows:

``mode`` (str)
  Defines how the source values are computed. Currently the options are:

* ``'ZERO'``
    Source is set to zero.

* ``'MODEL'``
    Source values come from a model in code. Specific model selection where more
    than one model is available can be done by specifying a ``model_func``.
    This is documented in the individual source sections.

* ``'PRESCRIBED'``
    Source values are arbitrarily prescribed by the user. The value is set by
    ``prescribed_values``, and  should be a tuple of values. Each value can
    contain the same data structures as :ref:`Time-varying arrays`. Note that
    these values are treated completely independently of each other so for
    sources with multiple time dimensions, the prescribed values should each
    contain all the information they need.
    For sources which affect multiple core profiles, look at the source's
    ``affected_core_profiles`` property to see the order in which the
    prescribed values should be provided.

For example, to set 'fusion_power' to zero, e.g. for testing or sensitivity purposes, set:

.. code-block:: python

    'sources': {
        'fusion': {'mode': 'ZERO'},
    }

To set 'j_ext' to a prescribed value based on a tuple of numpy arrays, e.g. as defined or loaded from a file in the
preamble to the CONFIG dict within config module, set:

.. code-block:: python

    'sources': {
        'generic_current': {
            'mode': 'PRESCRIBED',
            'prescribed_values': ((times, rhon, current_profiles),),
        },

where the example ``times`` is a 1D numpy array of times, ``rhon`` is a 1D numpy array of normalized toroidal flux
coordinates, and ``current_profiles`` is a 2D numpy array of the current profile at each time. These names are arbitrary,
and can be set to anything convenient.


``is_explicit`` (bool)
  Defines whether the source is to be considered explicit or implicit. Explicit sources are calculated based on the simulation state at the
  beginning of a time step, or do not have any dependance on state. Implicit sources depend on updated states as the iterative solvers evolve the state through the
  course of a time step. If a source model is complex but evolves over slow timescales compared to the state, it may be beneficial to set it as explicit.


generic_heat
^^^^^^^^^^^^

A utility source module that allows for a time dependent Gaussian ion and electron heat source.

``mode`` (str = 'model')

``rsource`` (float = 0.0), **time-varying-scalar**
  Gaussian center of source profile in units of :math:`\hat{\rho}`.

``w`` (float = 0.25), **time-varying-scalar**
  Gaussian width of source profile in units of :math:`\hat{\rho}`.

``Ptot`` (float = 120e6), **time-varying-scalar**
  Total source power in W. High default based on total ITER power including alphas

``el_heat_fraction`` (float = 0.66666), **time-varying-scalar**
  Electron heating fraction.

ei_exchange
^^^^^^^^^^^

Ion-electron heat exchange.

``mode`` (str = 'model')

``Qei_mult`` (float = 1.0)
  Multiplication factor for ion-electron heat exchange term for testing purposes.

ohmic
^^^^^

Ohmic power.

``mode`` (str = 'model')

fusion
^^^^^^

Fusion power assuming a 50-50 D-T ion distribution.

``mode`` (str = 'model')

gas_puff
^^^^^^^^

Exponential based gas puff source. No first-principle-based model is yet implemented in TORAX.

``mode`` (str = 'model')

``puff_decay_length`` (float = 0.05), **time-varying-scalar**
  Gas puff decay length from edge in units of :math:`\hat{\rho}`.

``S_puff_tot`` (float = 1e22), **time-varying-scalar**
  Total number of particle source in units of particles/s.

pellet
^^^^^^

Time dependent Gaussian pellet source. No first-principle-based model is yet implemented in TORAX.

``mode`` (str = 'model')

``pellet_deposition_location`` (float = 0.85), **time-varying-scalar**
  Gaussian center of source profile in units of :math:`\hat{\rho}`.

``pellet_width`` (float = 0.1), **time-varying-scalar**
  Gaussian width of source profile in units of :math:`\hat{\rho}`.

``S_pellet_tot`` (float = 2e22), **time-varying-scalar**
  Total particle source in units of particles/s

generic_particle
^^^^^^^^^^^^^^^^

Time dependent Gaussian particle source. No first-principle-based model is yet implemented in TORAX.

``mode`` (str = 'model')

``deposition_location`` (float = 0.0), **time-varying-scalar**
  Gaussian center of source profile in units of :math:`\hat{\rho}`.

``particle_width`` (float = 0.25), **time-varying-scalar**
  Gaussian width of source profile in units of :math:`\hat{\rho}`.

``S_tot`` (float = 1e22), **time-varying-scalar**
  Total particle source.

j_bootstrap
^^^^^^^^^^^

Bootstrap current calculated with the Sauter model.

``mode`` (str = 'model')

``bootstrap_mult`` (float = 1.0)
  Multiplication factor for bootstrap current for testing purposes.

generic_current
^^^^^^^^^^^^^^^

Generic external current profile, parameterized as a Gaussian.

``mode`` (str = 'model')

``rext`` (float = 0.4), **time-varying-scalar**
  Gaussian center of current profile in units of :math:`\hat{\rho}`.

``wext`` (float = 0.05), **time-varying-scalar**
  Gaussian width of current profile in units of :math:`\hat{\rho}`.

``Iext`` (float = 3.0), **time-varying-scalar**
  Total current in MA. Only used if ``use_absolute_current==True``.

``fext`` (float = 0.2), **time-varying-scalar**
  Sets total ``j_ext`` to be a fraction ``fext`` of the total plasma current.
  Only used if ``use_absolute_current==False``.

``use_absolute_current`` (bool = False)
  Toggles relative vs absolute external current setting.

bremsstrahlung
^^^^^^^^^^^^^^

Bremsstrahlung model from Wesson, with an optional correction for relativistic effects from Stott PPCF 2005.

``mode`` (str = 'model')

``use_relativistic_correction`` (bool = False)

impurity_radiation
^^^^^^^^^^^^^^^^^^

Various models for impurity radiation. Runtime params for each available model are listed separately

``mode`` (str = 'model')

``model_func`` (str = 'impurity_radiation_mavrin_fit')

The following models are available:

* ``'impurity_radiation_mavrin_fit'``
    Polynomial fits to ADAS data from `Mavrin, 2018. <https://doi.org/10.1080/10420150.2018.1462361>`_

    ``radiation_multiplier`` (float = 1.0). Multiplication factor for radiation term for testing sensitivities.

* ``'radially_constant_fraction_of_Pin'``
    Sets impurity radiation to be a constant fraction of the total external input power.

    ``fraction_of_total_power_density`` (float = 1.0). Fraction of total external input power to use for impurity radiation.

cyclotron_radiation
^^^^^^^^^^^^^^^^^^^

Cyclotron radiation model from Albajar NF 2001 with a deposition profile from Artaud NF 2018.

``mode`` (str = 'model')

``wall_reflection_coeff`` (float = 0.9)
  Machine-dependent dimensionless parameter corresponding to the fraction of
  cyclotron radiation reflected off the wall and reabsorbed by the plasma.

``beta_min`` (float = 0.5)

``beta_max`` (float = 8.0)

``beta_grid_size`` (int = 32)
  beta in this context is a variable in the temperature profile parameterization used
  in the Albajar model. The parameter is fit with simple grid search performed over
  the range ``[beta_min, beta_max]``, with ``beta_grid_size`` uniformly spaced steps.

ecrh
^^^^
Electron-cyclotron heating and current drive, based on the local efficiency model in `Lin-Liu et al., 2003 <https://doi.org/10.1063/1.1610472>`_.
Given an EC power density profile and efficiency profile, the model produces the corresponding EC-driven current density profile.
The user has three options:

1. Provide an entire EC power density profile manually (via ``manual_ec_power_density``).
2. Provide the parameters of a Gaussian EC deposition (via ``gaussian_ec_power_density_width``, ``gaussian_ec_power_density_location``, and ``gaussian_ec_total_power``).
3. Any combination of the above.

By default, both the manual and Gaussian profiles are zero. The manual and Gaussian profiles are summed together to produce the final EC deposition profile.

    ``mode`` (str = 'model')

    ``manual_ec_power_density`` **time-varying-array**
        EC power density deposition profile, in units of :math:`W/m^3`.

    ``gaussian_ec_power_density_width`` **time-varying-scalar**
        Width of Gaussian EC power density deposition profile.

    ``gaussian_ec_power_density_location`` **time-varying-scalar**
        Location of Gaussian EC power density deposition profile on the normalized rho grid.

    ``gaussian_ec_total_power`` **time-varying-scalar**
        Integral of the Gaussian EC power density profile, setting the total power.

    ``cd_efficiency`` **time-varying-scalar**
        Dimensionless local efficiency profile for conversion of EC power to current.

icrh
^^^^
Ion cyclotron heating using a surrogate model of the TORIC ICRH spectrum
solver simulation https://meetings.aps.org/Meeting/DPP24/Session/NP12.106.
This source is currently SPARC specific.

Weights and configuration for the surrogate model are needed to use this source.
By default these are expected to be found under
``'~/toric_surrogate/TORIC_MLP_v1/toricnn.json'``. To use a different file path
an alternative path can be provided using the ``TORIC_NN_MODEL_PATH``
environment variable which should point to a compatible JSON file.

``mode`` (str = 'model')

``wall_inner`` (float = 1.24)
  Inner radial location of first wall at plasma midplane level [m].

``wall_outer`` (float = 2.43)
  Outer radial location of first wall at plasma midplane level [m].

``frequency`` (float = 120e6) **time-varying-scalar**
  ICRF wave frequency in Hz.

``minority_concentration`` (float = 3.0) **time-varying-scalar**
  Helium-3 minority concentration relative to the electron density in %.

``Ptot`` (float = 10e6), **time-varying-scalar**
  Total injected source power in W.

See :ref:`physics_models` for more detail.

solver
-------

Select and configure the ``Solver`` object, which evolves the PDE system by one timestep. See :ref:`solver_details` for further details.
The dictionary consists of keys common to all solvers. Additional fields for
parameters pertaining to a specific solver are defined in the relevant section below.

``solver_type`` (str = 'linear')
  Selected PDE solver algorithm. The current options are:

* ``'linear'``
    Linear solver where PDE coefficients are set at fixed values of the state. An approximation of the nonlinear solution is optionally
    carried out with a predictor-corrector method, i.e. fixed point iteration of the PDE coefficients.

* ``'newton_raphson'``
    Nonlinear solver using the Newton-Raphson iterative algorithm, with backtracking line search, and timestep backtracking,
    for increased robustness.

* ``'optimizer'``
    Nonlinear solver using the jaxopt library.

``theta_implicit`` (float = 1.0)
  theta value in the theta method of time discretization. 0 = explicit, 1 = fully implicit, 0.5 = Crank-Nicolson.

``adaptive_dt`` (bool = True)
  If true, then turns on dt backtracking, where dt is iteratively reduced by ``dt_reduction_factor`` in a new attempt step
  if the solver does not converge. Only relevant for nonlinear solvers.

``dt_reduction_factor`` (float = 3.0)
  dt reduction factor if the solver does not converge following a call, and ``adaptive_dt=True``. Only relevant
  for nonlinear solvers.

``use_predictor_corrector`` (bool = True)
  Enables use_predictor_corrector iterations with the linear solver.

``n_corrector_steps`` (int = 1)
  Number of corrector steps for the predictor-corrector linear solver. 0 means a pure linear solve with no corrector steps.

``use_pereverzev`` (bool = False)
  Use Pereverzev-Corrigan terms in the heat and particle flux when using the linear solver.
  Critical for stable calculation of stiff transport, at the cost of introducing non-physical lag during transient. Also used for
  the ``linear_step`` initial guess mode in the nonlinear solvers.

``chi_pereverzev`` (float = 20.0)
  Large heat conductivity used for the Pereverzev-Corrigan term.

``D_pereverzev`` (float = 10.0)
  Large particle diffusion used for the Pereverzev-Corrigan term.

linear
^^^^^^

Runtime parameters relevant for the ``LinearThetaMethod``, e.g. ``use_predictor_corrector``, are not defined in the child class but in the parent
``Stepper`` class and hence in the upper layer of the ``solver`` config dict. Since the nonlinear solvers also have the option of using
a linear solver for calculating an initial guess, it is more appropriate for these shared linear runtime parameters to be defined in the
parent ``Stepper`` class.

newton_raphson
^^^^^^^^^^^^^^

.. _log_iterations:

``log_iterations`` (bool = False)
  If True, logs information about the internal state of the Newton-Raphson
  solver. For the first iteration, this contains the initial residual value and
  time-step size. For subsequent iterations, this contains the iteration step
  number, the current value of the residual, and the current value of ``tau``,
  which is the relative reduction in Newton step size compared to the original
  Newton step size. If the solver does not converge, then these inner iterations
  will restart at a smaller timestep size if ``adaptive_dt=True`` in the
  ``solver`` config dict.

``initial_guess_mode`` (str = 'linear_step')
  Sets the approach taken for the initial guess into the Newton-Raphson solver for the first iteration.
  Two options are available:

* ``x_old``
    Use the state at the beginning of the timestep.

* ``linear_step``
    Use the linear solver to obtain an initial guess to warm-start the nonlinear solver.
    If used, is recommended to do so with the use_predictor_corrector solver and
    several corrector steps. It is also strongly recommended to
    use_pereverzev=True if a stiff transport model like qlknn is used.

``tol`` (float = 1e-5)
  PDE residual magnitude tolerance for successfully exiting the iterative solver.

``coarse_tol`` (float = 1e-2)
  If the solver hits an exit criterion due to small steps or many iterations,
  but the residual is still below ``coarse_tol``, then the step is allowed to successfully pass, and a warning is passed to the user.

``maxiter`` (int = 30)
  Maximum number of allowed Newton iterations. If the number of iterations surpasses ``maxiter``, then the solver will
  exit in an unconverged state.   The step will still be accepted if ``residual < coarse_tol``, otherwise dt backtracking will take place if enabled.

``delta_reduction_factor`` (float = 0.5)
  Reduction of Newton iteration step size in the backtracking line search. If in a given iteration,
  the new state is unphysical (e.g. negative temperatures) or the residual increases in magnitude, then a smaller step will be iteratively taken
  until the above conditions are met.

``tau_min`` (float = 0.01)
  tau is the relative reduction in step size: delta/delta_original, following backtracking line search,
  where delta_original is the step in state :math:`x` that minimizes the linearized PDE system. If following some iterations,
  ``tau`` :math:`<` ``tau_min``, , then the solver will exit in an unconverged state. The step will still be accepted if ``residual < coarse_tol``,
  otherwise dt backtracking will take place if enabled.

optimizer
^^^^^^^^^

``initial_guess_mode`` (str = 'linear_step')
  Sets the approach taken for the initial guess into the Newton-Raphson solver for the first iteration.
  Two options are available:

* ``x_old``
    Use the state at the beginning of the timestep.

* ``linear_step``
    Use the linear solver to obtain an initial guess to warm-start the nonlinear solver.
    If used, is recommended to do so with the use_predictor_corrector solver and
    several corrector steps. It is also strongly recommended to
    use_pereverzev=True if a stiff transport model like qlknn is used.

``tol`` (float = 1e-12)
  PDE loss magnitude tolerance for successfully exiting the iterative solver.
  Note: the default tolerance here is smaller than the default tolerance for
  the Newton-Raphson solver because it's a tolerance on the loss (square of the
  residual).

``maxiter`` (int = 100)
  Maximum number of allowed optimizer iterations.

time_step_calculator
--------------------

``time_step_calculator_type`` (str = 'chi')
  The name of the ``time_step_calculator``, a method which calculates ``dt`` at every timestep.
  Two methods are currently available:

* ``'fixed'``
    ``dt`` is equal to ``fixed_dt`` defined in :ref:`numerics_dataclass`. If the Newton-Raphson solver is being used
    and ``adaptive_dt==True``, then in practice some steps may have lower ``dt`` if the solver needed to backtrack.

* ``'chi'``
    adaptive dt method, where ``dt`` is a multiple of a base dt inspired by the explicit stability limit for parabolic PDEs:
    :math:`dt_{base}=\frac{dx^2}{2\chi}`, where :math:`dx` is the grid resolution and :math:`\chi=max(\chi_i, \chi_e)`. ``dt=dtmult * dt_base``, where
    ``dtmult`` is defined in :ref:`numerics_dataclass`, and can be significantly larger than unity for implicit solvers.

Scaling the timestep to be :math:`\propto \chi` helps protect against traversing through fast transients, if there is a desire for them to be fully resolved.


Additional Notes
================

.. _dynamic_vs_static:

Dynamic vs. Static Parameters
-----------------------------

Dynamic parameters: These can be changed without recompiling the simulation code. Examples include time-dependent parameters like heating power or external current.

Static parameters: These define the fundamental structure of the simulation and require JAX recompilation if changed.
Examples include the number of grid points or the choice of transport model. A partial list is provided below.

* ``runtime_params['geometry']['nrho']``
* ``runtime_params['numerics']['ion_heat_eq']``
* ``runtime_params['numerics']['el_heat_eq']``
* ``runtime_params['numerics']['current_eq']``
* ``runtime_params['numerics']['dens_eq']``
* ``transport['transport_model']``
* ``solver['solver_type']``
* ``time_step_calculator['time_step_calculator_type']``
* ``sources['source_name']['is_explicit']``
* ``sources['source_name']['mode']``

Examples
========

An example configuration dict, corresponding to a non-rigorous demonstration mock-up of a time-dependent ITER
hybrid scenario rampup (presently with a fixed CHEASE geometry), is shown below.
The configuration file is also available in ``torax/examples/iterhybrid_rampup.py``.

.. code-block:: python

  CONFIG = {
      'plasma_composition': {
          'main_ion': {'D': 0.5, 'T': 0.5},
          'impurity': 'Ne',
          'Zeff': 1.6,
      },
      'profile_conditions': {
          'I_total': {0: 3, 80: 10.5},
          # initial condition ion temperature for r=0 and r=Rmin
          'Ti': {0.0: {0.0: 6.0, 1.0: 0.1}},
          'T_i_right_bc': 0.1,  # boundary condition ion temp for r=Rmin
          # initial condition electron temperature between r=0 and r=Rmin
          'Te': {0.0: {0.0: 6.0, 1.0: 0.1}},
          'T_e_right_bc': 0.1,  # boundary condition electron temp for r=Rmin
          'n_e_bound_right_is_fGW': True,
          'n_e_bound_right': {0: 0.1, 80: 0.3},
          'n_e_is_fGW': True,
          'nbar': 1,
          'n_e': {0: {0.0: 1.5, 1.0: 1.0}},  # Initial electron density profile
          'Tiped': 1.0,
          'Teped': 1.0,
          'neped_is_fGW': True,
          'neped': {0: 0.3, 80: 0.7},
          'Ped_top': 0.9,
      },
      'numerics': {
          't_final': 80,
          'fixed_dt': 2,
          'ion_heat_eq': True,
          'el_heat_eq': True,
          'current_eq': True,
          'dens_eq': True,
          'dt_reduction_factor': 3,
          'largeValue_T': 1.0e10,
          'largeValue_n': 1.0e8,
      },
      'geometry': {
          'geometry_type': 'chease',
          'geometry_file': 'ITER_hybrid_citrin_equil_cheasedata.mat2cols',
          'Ip_from_parameters': True,
          'Rmaj': 6.2,
          'Rmin': 2.0,
          'B0': 5.3,
      },
      'sources': {
          'j_bootstrap': {},
          'generic_current': {
              'fext': 0.15,
              'wext': 0.075,
              'rext': 0.36,
          },
          'pellet': {
              'S_pellet_tot': 0.0e22,
              'pellet_width': 0.1,
              'pellet_deposition_location': 0.85,
          },
          'generic_heat': {
              'rsource': 0.12741589640723575,
              'w': 0.07280908366127758,
              # total heating (with a rough assumption of radiation reduction)
              'Ptot': 20.0e6,
              'el_heat_fraction': 1.0,
          },
          'fusion': {},
          'ei_exchange': {},
      },
      'transport': {
          'transport_model': 'qlknn',
          'apply_inner_patch': True,
          'De_inner': 0.25,
          'Ve_inner': 0.0,
          'chii_inner': 1.5,
          'chie_inner': 1.5,
          'rho_inner': 0.3,
          'apply_outer_patch': True,
          'De_outer': 0.1,
          'Ve_outer': 0.0,
          'chii_outer': 2.0,
          'chie_outer': 2.0,
          'rho_outer': 0.9,
          'chimin': 0.05,
          'chimax': 100,
          'Demin': 0.05,
          'Demax': 50,
          'Vemin': -10,
          'Vemax': 10,
          'smoothing_sigma': 0.1,
          'qlknn_params': {
              'DVeff': True,
              'avoid_big_negative_s': True,
              'An_min': 0.05,
              'ITG_flux_ratio_correction': 1,
          },
      },
      'pedestal': {
          'pedestal_model': 'set_tped_nped',
          'set_pedestal': True,
          'Tiped': 1.0,
          'Teped': 1.0,
          'rho_norm_ped_top': 0.95,
      },
      'solver': {
          'solver_type': 'newton_raphson',
          'use_predictor_corrector': True,
          'n_corrector_steps': 10,
          'chi_pereverzev': 30,
          'D_pereverzev': 15,
          'use_pereverzev': True,
          'log_iterations': False,
      },
      'time_step_calculator': {
          'calculator_type': 'fixed',
      },
  }


Restarting a simulation
=======================
In order to restart a simulation a field can be added to the config.

For example following a simulation in which a state file is saved to
``/path/to/torax_state_file.nc``, if we want to start a new simulation from the
state of the previous one at ``t=10`` we could add the following to our config:

.. code-block:: python

  {
      'filename': '/path/to/torax_state_file.nc',
      'time': 10,
      'do_restart': True,  # Toggle to enable/disable a restart.
      # Whether or not to pre"stitch" the contents of the loaded state file up
      # to `time` with the output state file from this simulation.
      'stitch': True,
  }

The subsequence simulation will then recreate the state from ``t=10`` in the
previous simulation and then run the simulation from that point in time. For
all subsequent steps the dynamic runtime parameters will be constructed using
the given runtime parameter configuration (from ``t=10`` onwards).

If the requested restart time is not exactly available in the state file, the
simulation will restart from the closest available time. A warning will be
logged in this case.

We envisage this feature being useful for example to:

* restart a(n expensive) simulation that was healthy up till a certain time and
  then failed. After discovering the issue for breakage you could then restart
  the sim from the last healthy point.

* do uncertainty quantification by sweeping lots of configs following running
  a simulation up to a certain point in time. After running the initial
  simulation you could then modify and sweep the runtime parameter config in
  order to do some uncertainty quantification.

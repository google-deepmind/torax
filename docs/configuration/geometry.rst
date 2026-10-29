.. _geometry_doc:

geometry
########

``geometry_type`` (str)
  Geometry model used. A string must be provided from the following options.

* ``'circular'``
    An ad-hoc circular geometry model. Includes elongation corrections.
    Not recommended for use apart from for testing purposes.

* ``'chease'``
    Loads a CHEASE geometry file.

* ``'fbt'``
    Loads FBT geometry files.
    NOTE: ``<B^2>`` and ``<1/B^2>`` are not currently supported by FBT geometry;
    TORAX approximates these using analytical expressions for circular
    geometry. This might cause inaccuracies in neoclassical transport.

* ``'eqdsk'``
    Loads a EQDSK geometry file, and carries out the appropriate
    flux-surface-averages of the 2D poloidal flux. Use of EQDSK geometry comes
    with the following caveat: The TORAX EQDSK converter has only been tested
    against CHEASE-generated EQDSK which is COCOS=2. The converter is not
    guaranteed to work as expected with arbitrary EQDSK input, so please verify
    carefully. Future work will be done to correctly handle EQDSK inputs
    provided with a specific COCOS value.

* ``'imas'``
    Loads an IMAS netCDF file containing an equilibrium Interface Data Structure
    (IDS) or directly the equilibrium IDS on the fly. It handles IDSs in Data
    Dictionary version 4.0.0.

* ``'tokamaker'``
    Takes flux surface averages computed by
    `TokaMaker <https://github.com/openfusiontoolkit/OpenFUSIONToolkit>`_.
    TokaMaker traces flux surfaces on the finite element mesh used for the
    equilibrium solve, then passes arrays of flux surface averaged quantities
    to TORAX. TORAX does not depend on TokaMaker (or Open FUSION Toolkit),
    the geometry information is passed as plain arrays, which can be done
    without saving and loading files.

Geometry dicts for all geometry types can contain the following additional keys.

``calc_Phi_b_dot`` (bool [default = True])
  Toggles whether to calculate time-dependent toroidal flux derivatives
  (:math:`\dot{\Phi}_b`) in the geometry dataclasses. Used in ``calc_coeffs`` for moving
  mesh / time-dependent geometry terms.

``n_rho`` (int | None [default = 25])
  Number of radial grid cells. Creates a uniform grid with cell width
  :math:`d\hat{\rho} = 1/n\_rho`. Must be at least 4.
  Either ``n_rho`` or ``face_centers`` must be specified.

``face_centers`` (array | None [default = None])
  Explicit array of face center coordinates in normalized :math:`\hat{\rho}`
  (ranging from 0 to 1) for defining non-uniform radial grids. For a grid with
  N cells, there should be N+1 face centers. The array must start at 0.0, end
  at 1.0, be strictly increasing, and have at least 5 elements (4 cells).
  This enables finer resolution in regions of interest, such as
  near the plasma edge.

  Example of a non-uniform grid with finer resolution near the edge:

  .. code-block:: python

    import numpy as np

    # Create non-uniform grid: coarse in core, fine near edge
    core_faces = np.linspace(0, 0.8, 9)  # 8 cells from rho=0 to rho=0.8
    edge_faces = np.linspace(0.8, 1.0, 9)[1:]  # 8 cells from rho=0.8 to rho=1.0
    face_centers = np.concatenate([core_faces, edge_faces])

    'geometry': {
        'geometry_type': 'chease',
        'face_centers': face_centers,
        # ... other geometry parameters
    }

``hires_factor`` (int [default = 4])
  Only used when the initial condition ``psi`` is from plasma current. Sets up a
  higher resolution mesh with ``nrho_hires = nrho * hi_res_fac``, used for
  ``j`` to ``psi`` conversions.

``trapped_fraction_source`` (str [default = 'sauter'])
  Selects how the effective trapped particle fraction, used by the
  neoclassical ``bootstrap_current``, ``conductivity``, and ``transport``
  models, is computed. Computed once at geometry construction time. Options are:

  * ``'sauter'`` (default)
    Uses the analytical approximation from
    `O. Sauter, Fusion Eng. Des. 112, 633 (2016) <https://doi.org/10.1016/j.fusengdes.2016.04.033>`_.
    Supported by every geometry source.

  * ``'file'``
    Reads the value precomputed by the input equilibrium code directly from
    the geometry file. Only supported for CHEASE and IMAS geometries.

  * ``'exact'``
    Computes the exact bounce-averaged integral directly from the traced 2D
    equilibrium, using the formulation of
    `Y. R. Lin-Liu and R. L. Miller, Phys. Plasmas 2, 1666 (1995) <https://doi.org/10.1063/1.871315>`_.
    Only supported for EQDSK and IMAS geometries.


Geometry dicts for file-based geometry types (such as CHEASE and EQDSK) can
contain the following additional keys.

``geometry_file`` (str) See below for information on defaults
  Required for CHEASE and EQDSK geometry. Sets the geometry file loaded.
  The default is set to ``iterhybrid.mat2cols`` for
  CHEASE geometry and ``iterhybrid_cocos02.eqdsk`` for EQDSK geometry.

``geometry_directory`` (str | None [default = None])
  Optionally set the geometry directory. This should be set to an absolute path.
  If not set, then the default is ``torax/data/third_party/geo``

Geometry dicts for all non-circular geometry types can contain the following
additional key.

``I_p_from_parameters`` (bool [default = True])
  Toggles whether total plasma current is read from the configuration file, or
  from the geometry file or equilibrium object. If ``True``, then the
  :math:`\psi` calculated from the geometry file or equilibrium object is scaled
  to match the desired :math:`I_p`.

Geometry dicts for analytical circular geometry require the following additional
keys.

``R_major`` (float [default = 6.2])
  Major radius, defined as the midplane average of the LCFS radius,
  :math:`R_\mathrm{major} = (R_\mathrm{max} + R_\mathrm{min}) / 2`, in meters.

``a_minor`` (float [default = 2.0])
  Minor radius, defined as half the horizontal width of the LCFS,
  :math:`a_\mathrm{minor} = (R_\mathrm{max} - R_\mathrm{min}) / 2`, in meters.

``B_0`` (float [default = 5.3])
  Vacuum toroidal magnetic field at ``R_major``, in :math:`T`.

``elongation_LCFS`` (float [default = 1.72])
  Sets the plasma elongation used for volume, area and q-profile corrections.

Geometry dicts for CHEASE geometry require the following additional keys for
denormalization.

``R_major`` (float [default = 6.2])
  Major radius, defined as the midplane average of the LCFS radius,
  :math:`R_\mathrm{major} = (R_\mathrm{max} + R_\mathrm{min}) / 2`, in meters.

``a_minor`` (float [default = 2.0])
  Minor radius, defined as half the horizontal width of the LCFS,
  :math:`a_\mathrm{minor} = (R_\mathrm{max} - R_\mathrm{min}) / 2`, in meters.

``B_0`` (float [default = 5.3])
  Vacuum toroidal magnetic field at ``R_major``, in :math:`T`.

Geometry dicts for FBT geometry require the following additional keys.

``divertor_domain`` (str [default = 'lower_null'])
  Selects the divertor domain used to extract edge geometry parameters used
  by the :ref:`extended_lengyel` edge model.
  Either ``'lower_null'`` or ``'upper_null'``.

``LY_object`` (dict[str, np.ndarray | float] | str | None [default = None])
  Sets a single-slice FBT LY geometry file to be loaded, or alternatively a dict
  directly containing a single time slice of LY data.
  **Note:** FBT files can optionally contain edge geometry parameters
  (e.g., ``Lpar_target``) used by the :ref:`extended_lengyel` edge model.

``LY_bundle_object`` (dict[str, np.ndarray | float] | str | None
  [default = None]) Sets the FBT LY bundle file to be loaded, corresponding to
  multiple time-slices, or alternatively a dict directly containing all
  time-slices of LY data.

``LY_to_torax_times`` (ndarray | None [default = None])
  Sets the TORAX simulation times corresponding to the individual slices in the
  FBT LY bundle file. If not provided, then the times are taken from the
  LY_bundle_file itself. The length of the array must match the number of slices
  in the bundle.

``L_object`` (dict[str, np.ndarray | float] | str | None [default = None])
  Sets the FBT L geometry file loaded, or alternatively a dict directly
  containing the L data.

Geometry dicts for EQDSK geometry can contain the following additional keys.
It is only recommended to change the default values if issues arise.

``n_surfaces`` (int [default = 100])
  Number of surfaces for which flux surface averages are calculated.

``last_surface_factor`` (float [default = 0.99])
  Multiplication factor of the boundary poloidal flux, used for the contour
  defining geometry terms at the LCFS on the TORAX grid. Needed to avoid
  divergent integrations in diverted geometries.

``target_psi_grid_resolution`` (int | None [default = 128])
  Desired minimum number of grid points along the R and Z dimensions of the 2D
  psi grid before generating flux surface contours. Contours are linear between
  grid points, so for coarse EQDSK grids the flux surfaces near the separatrix
  can be inaccurate (e.g. underestimating the plasma current) or fail to close.
  Refining the grid allows ``last_surface_factor`` to be closer to 1. If set,
  the 2D (R, Z) grid is refined (using bicubic spline interpolation of psi)
  by an integer factor calculated dynamically as:

  .. code-block:: python

    ceil(target_psi_grid_resolution / min(nx, nz))

  refining both dimensions uniformly. If None, no refinement is performed.
  Must be >= 1 if provided.


Geometry dicts for IMAS geometry require one and only one of the following
additional keys.

``imas_filepath`` (str)
  Sets the path of the IMAS netCDF file containing the geometry data in an
  equilibrium IDS to be loaded.

``imas_uri`` (str)
  Sets the path of the IMAS data entry containing the geometry data in an
  equilibrium IDS to be loaded.

``equilibrium_object`` (imas.ids_toplevel.IDSToplevel)
  An equilibrium IDS object that can be inserted directly.

Geometry dicts for IMAS geometry also support the following optional keys.

``slice_time`` (float | None [default = None])
  Controls which time slice(s) are loaded from the IMAS IDS. If ``None``, all
  time slices are loaded and the TORAX time-dependent geometry is interpolated
  between them. If a float is provided, only the single time slice
  closest to the given time (in seconds) is loaded.

``explicit_convert`` (bool [default = True])
  Whether to explicitly convert the IDS to the current Data Dictionary (DD)
  version before loading. Explicit conversion is required when converting
  between major DD versions. See the
  `IMAS-Python multi-DD documentation
  <https://imas-python.readthedocs.io/en/latest/multi-dd.html#conversion-of-idss-between-dd-versions>`_
  for details.

**Example — all time slices from a single IMAS file (default):**

.. code-block:: python

  'geometry': {
      'geometry_type': 'imas',
      'I_p_from_parameters': True,
      'imas_filepath': '/path/to/equilibrium.nc',
  }

**Example — single time slice:**

.. code-block:: python

  'geometry': {
      'geometry_type': 'imas',
      'I_p_from_parameters': True,
      'imas_filepath': '/path/to/equilibrium.nc',
      'slice_time': 40.0,
  }

Geometry dicts for TokaMaker geometry require the following additional keys.

``fsa_profiles`` (dict)
  Dictionary containing flux surface averaged profiles matching the format
  returned by ``OpenFUSIONToolkit.TokaMaker.TokaMaker.get_fsa()`` (see the
  `OpenFUSIONToolkit TokaMaker documentation
  <https://openfusiontoolkit.github.io/OpenFUSIONToolkit/doc_tokamaker_main.html>`_).

For setting up time-dependent geometry, a subset of varying geometry parameters
and input files can be defined in a ``geometry_configs`` dict, which is a
time-series of {time: {configs}} pairs. For example, a time-dependent geometry
input with 3 time-slices of single-time-slice FBT geometries can be set up as:

.. code-block:: python

  'geometry': {
      'geometry_type': 'fbt',
      'I_p_from_parameters': True,
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
files packaged within a single ``.mat`` file using LIUQE meqlpack. This
eliminates the need to specify multiple individual LY files in the
``geometry_configs`` parameter.

To use this feature, set ``LY_bundle_object`` to the corresponding ``.mat`` file
containing the LY bundle. Optionally set ``LY_to_torax_times`` as a NumPy array
corresponding to times of the individual LY slices within the bundle. If not
provided, then the times are taken from the bundle file itself.

Note that ``LY_bundle_object`` cannot coexist with ``LY_file`` or
``geometry_configs`` in the same configuration, and will raise an error if so.

All file loading and geometry processing is done upon simulation initialization.
The geometry inputs into the TORAX PDE coefficients are then time-interpolated
on-the-fly onto the TORAX time slices where the PDE calculations are done.

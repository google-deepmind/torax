.. _geometry_doc:

geometry
########

.. torax_config:: torax._src.geometry.config.GeometryConfig

.. torax_config:: torax._src.geometry.base.GeometryConfigBase

circular
========

.. torax_config:: torax._src.geometry.circular_geometry.CircularConfig
   :include-class-doc:
   :own-fields-only:
   :exclude-fields: geometry_type

chease
======

.. torax_config:: torax._src.geometry.chease.CheaseConfig
   :include-class-doc:
   :own-fields-only:
   :exclude-fields: geometry_type

meq
===

.. torax_config:: torax._src.geometry.meq.MEQConfig
   :include-class-doc:
   :own-fields-only:
   :exclude-fields: geometry_type

eqdsk
=====

.. torax_config:: torax._src.geometry.eqdsk.EQDSKConfig
   :include-class-doc:
   :own-fields-only:
   :exclude-fields: geometry_type

imas
====

.. torax_config:: torax._src.geometry.imas.IMASConfig
   :include-class-doc:
   :own-fields-only:
   :exclude-fields: geometry_type

**Example — all time slices from a single IMAS file (default):**

.. code-block:: python

  'geometry': {
      'geometry_type': 'imas',
      'imas_filepath': '/path/to/equilibrium.nc',
  }

**Example — single time slice:**

.. code-block:: python

  'geometry': {
      'geometry_type': 'imas',
      'imas_filepath': '/path/to/equilibrium.nc',
      'slice_time': 40.0,
  }

tokamaker
=========

.. torax_config:: torax._src.geometry.tokamaker.TokaMakerConfig
   :include-class-doc:
   :own-fields-only:
   :exclude-fields: geometry_type

Time-dependent geometry
=======================

For setting up time-dependent geometry, a subset of varying geometry parameters
and input files can be defined in a ``geometry_configs`` dict, which is a
time-series of ``{time: {configs}}`` pairs. For example, a time-dependent
geometry input with 3 time-slices of single-time-slice MEQ geometries can be set
up as:

.. code-block:: python

  'geometry': {
      'geometry_type': 'meq',
      'geometry_configs': {
          20.0: {
              'LY_object': 'LY_early_rampup.mat',
              'L_object': 'L_early_rampup.mat',
          },
          50.0: {
              'LY_object': 'LY_mid_rampup.mat',
              'L_object': 'L_mid_rampup.mat',
          },
          100.0: {
              'LY_object': 'LY_endof_rampup.mat',
              'L_object': 'L_endof_rampup.mat',
          },
      },
  },

Alternatively, for MEQ data specifically, TORAX supports loading a bundle of LY
files packaged within a single ``.mat`` file using LIUQE ``meqlpack``. This
eliminates the need to specify multiple individual LY files in the
``geometry_configs`` parameter.

To use this feature, set ``LY_bundle_object`` to the corresponding ``.mat`` file
containing the LY bundle. Optionally set ``LY_to_torax_times`` as a NumPy array
corresponding to times of the individual LY slices within the bundle. If not
provided, then the times are taken from the bundle file itself.

Note that ``LY_bundle_object`` cannot coexist with ``LY_object`` or
``geometry_configs`` in the same configuration, and will raise an error if so.

All file loading and geometry processing is done upon simulation initialization.
The geometry inputs into the TORAX PDE coefficients are then time-interpolated
on-the-fly onto the TORAX time slices where the PDE calculations are done.

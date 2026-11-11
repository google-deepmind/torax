.. _transport_config:

transport
#########

In TORAX, turbulent transport models for the core and pedestal are combined
under a single top-level model container: the ``TransportModel`` (configured
under the ``transport`` key). Note that neoclassical transport is configured
separately under the :ref:`neoclassical <neoclassical_config>` section. See
:ref:`physics_transport` for details on the underlying transport physics models.

The transport model calculates turbulent transport coefficients by evaluating a
dictionary of named component models for the core (``core_transport_models``)
and pedestal (``pedestal_transport_models``). Each component model has a
user-chosen name as its key and is active within its defined radial domain,
which can be overlapping or non-overlapping. Top-level post-processing (min/max
clipping and Gaussian smoothing) is performed on the summed coefficients from
all component models across the core and pedestal. These combined coefficients
are then added to the neoclassical transport coefficients to form the total
transport coefficients used in the transport PDEs.

Top-level Transport Parameters
==============================

.. torax_config:: torax._src.transport_model.config.TransportConfig
   :include-fields: core_transport_models, pedestal_transport_models, chi_min, chi_max, D_e_min, D_e_max, V_e_min, V_e_max, smoothing_width, pedestal_smoothing_width, smoothing_zones

smoothing_zones
---------------

.. torax_config:: torax._src.transport_model.config.SmoothingZoneConfig

Combining Logic and Merge Modes
===============================

The combination logic is controlled by the ``merge_mode`` of each component
model inside ``core_transport_models`` and ``pedestal_transport_models``:

*   **ADD (default)**: The model's coefficients are added to the accumulated
    total in its active region. However, it does not contribute to regions
    and channels covered by an OVERWRITE model.
*   **OVERWRITE**: The model is the sole contributor to the transport
    coefficients in its active region, for enabled transport channels. It
    **locks** this region for the specific transport channels it provides,
    preventing other models from modifying them.

You can selectively enable or disable specific transport channels (e.g.,
``chi_i``, ``D_e``) for each model using flags like ``disable_chi_i``.
Disabled channels are "transparent" in OVERWRITE mode; they do not overwrite
existing values and do not set a lock. An example of when this could be useful
is when using one model for heat transport across the entire radial range, but
then using another model specifically for particle transport within a
restricted range, e.g. towards the LCFS.

Examples:

.. code-block:: python

  ...
  'transport': {
      'core_transport_models': {
          'inner': {
              'model_name': 'prescribed',
              'chi_i': 1.0,
              'rho_max': 0.3,
          },
          'outer': {
              'model_name': 'prescribed',
              'chi_i': 2.0,
              'rho_min': 0.2,
          },
      },
      'pedestal_transport_models': {
          'pedestal': {
              'model_name': 'prescribed',
              'chi_i': 0.5,
          },
      },
  },
  'pedestal': {
      'structure_model': {
          'model_name': 'set_T_top_n_top',
          'rho_norm_top': 0.9,
          'n_e_top': 0.8,
          'n_e_top_is_f_GW': True,
      },
  },
  ...

This would produce a ``chi_i`` profile that looks like the following.

.. image:: /images/combined_transport_example.png
  :width: 400
  :alt: A stepwise prescribed chi_i profile

Note that in the region :math:`[0, 0.2]`, only the first component is active,
so ``chi_i = 1.0``. In :math:`(0.2, 0.3]` the first two components are both
active, leading to a combined value of ``chi_i = 3.0``. In :math:`(0.3, 0.9]`,
only the second model is active (``chi_i = 2.0``), and in :math:`(0.9, 1.0]`
only the pedestal transport model is active (``chi_i = 0.5``).

The code for generating the plot above is found in
``docs/scripts/combined_transport_example.py``.

The next example shows how to apply a physics-based model (QLKNN) in the core,
but enforcing specific transport coefficients in the edge region using a
Prescribed model with ``OVERWRITE`` mode, effectively overriding the core model
in that region. This is useful e.g. for L-mode modelling.

.. code-block:: python

  'transport': {
      'core_transport_models': {
          # Base model: QLKNN applied everywhere (default ADD)
          'qlknn': {
              'model_name': 'qlknn',
              'rho_max': 1.0,
          },
          # Edge overwrite: Sets D_e and V_e in the edge, ignoring QLKNN there.
          # Keeps chi_i/chi_e from QLKNN (because they are disabled here).
          'edge_particle_transport_patch': {
              'model_name': 'prescribed',
              'rho_min': 0.9,
              'D_e': 0.5,
              'V_e': -1.0,
              'merge_mode': 'overwrite',
              'disable_chi_i': True,
              'disable_chi_e': True,
          },
      },
  },


Component Transport Models
==========================

Component models are specified inside ``core_transport_models`` and
``pedestal_transport_models``.

.. torax_config:: torax._src.transport_model.config.ComponentTransportModelConfig

   Transport model identifier selecting the component implementation.

.. torax_config:: torax._src.transport_model.config_base.ComponentTransportConfigBase


prescribed
==========

Prescribed (constant or space/time-varying) transport coefficients. This model
can be used to implement constant coefficients (e.g., ``chi_i = 1.0`` for all
:math:`\hat{\rho}`), as well as time-varying transport profiles of arbitrary
form using the **time-varying-array** syntax.

.. torax_config:: torax._src.transport_model.config.PrescribedTransportModelConfig
   :own-fields-only:
   :exclude-fields: model_name


cgm
===

Critical Gradient Model (CGM).

.. torax_config:: torax._src.transport_model.config.CriticalGradientTransportModelConfig
   :own-fields-only:
   :exclude-fields: model_name


bohm_gyrobohm
=============

Bohm-GyroBohm turbulent transport model.

.. torax_config:: torax._src.transport_model.config.BohmGyroBohmTransportModelConfig
   :own-fields-only:
   :exclude-fields: model_name


qlknn
=====

.. torax_config:: torax._src.transport_model.config.QLKNNTransportModelConfig
   :include-class-doc:
   :exclude-fields: model_name, rho_min, rho_max, disable_chi_i, disable_chi_e, disable_D_e, disable_V_e, fast_ion_stabilization, fast_ion_stabilization_model, fast_ion_stabilization_multiplier, merge_mode


tglfnn_ukaea
============

A TGLF Neural Network surrogate model developed by UKAEA, reimplemented in JAX
in |fusion_surrogates|. If you use this model, please cite |zanisi2025|.

.. torax_config:: torax._src.transport_model.config.TGLFNNukaeaTransportModelConfig
   :exclude-fields: model_name, rho_min, rho_max, disable_chi_i, disable_chi_e, disable_D_e, disable_V_e, fast_ion_stabilization, fast_ion_stabilization_model, fast_ion_stabilization_multiplier, merge_mode


qualikiz
========

The |qualikiz| quasilinear gyrokinetic transport model.

.. torax_config:: torax._src.transport_model.qualikiz_transport_model.QualikizTransportModelConfig
   :exclude-fields: model_name, rho_min, rho_max, disable_chi_i, disable_chi_e, disable_D_e, disable_V_e, fast_ion_stabilization, fast_ion_stabilization_model, fast_ion_stabilization_multiplier, merge_mode


tglf
====

The TGLF quasilinear turbulent transport model. If you want to use TORAX with
TGLF, see :ref:`tglf_installation` for installation instructions.

.. torax_config:: torax._src.transport_model.tglf.tglf_transport_model.TGLFTransportModelConfig
   :exclude-fields: model_name, rho_min, rho_max, disable_chi_i, disable_chi_e, disable_D_e, disable_V_e, fast_ion_stabilization, fast_ion_stabilization_model, fast_ion_stabilization_multiplier, merge_mode

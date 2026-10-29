.. _transport_config:

transport
#########

In TORAX, turbulent transport models for the core and pedestal are combined
under a single top-level model container: the ``TransportModel`` (configured
under the ``transport`` key). Note that neoclassical transport is configured
separately under the ``neoclassical`` section.

The transport model calculates turbulent transport coefficients by
evaluating a dictionary of named component models for the core
(``core_transport_models``) and pedestal (``pedestal_transport_models``). Each
component model has a user-chosen name as its key and is active within its
defined radial domain, which can be overlapping or non-overlapping. Top-level
post-processing (min/max clipping and Gaussian smoothing) is performed on the
summed coefficients from all component models across the core and pedestal.
These combined coefficients are then added to the neoclassical transport
coefficients to form the total transport coefficients used in the transport
PDEs.

Top-level Transport Parameters
==============================

``core_transport_models`` (dict[str, dict] [default = {}])
  A dictionary mapping user-given names to config dictionaries for the component
  models for turbulent transport in the core. Each entry must specify
  ``'model_name'`` (e.g. ``'prescribed'``, ``'qlknn'``, ``'cgm'``,
  ``'bohm_gyrobohm'``, ``'tglf'``, ``'tglfnn_ukaea'``, ``'qualikiz'``) along
  with model-specific parameters. For each component model, ``rho_min`` and
  ``rho_max`` define its active radial domain. If a pedestal is active and
  configured with an internal boundary condition, the core ``rho_max`` is
  bounded by ``rho_norm_ped_top``, where ``rho_norm_ped_top`` is set in the
  ``pedestal`` config.

``pedestal_transport_models`` (dict[str, dict] [default = {}])
  A dictionary mapping user-given names to config dictionaries for the component
  models for turbulent transport in the pedestal. The pedestal transport models
  are active only for radii above ``rho_norm_ped_top``. Setting ``rho_min`` or
  ``rho_max`` is not supported for pedestal transport models, and an error is
  raised if they deviate from default values (0.0 and 1.0).

``chi_min`` (float [default = 0.05])
  Lower allowed bound for heat conductivities :math:`\chi` across all models,
  in units of :math:`m^2/s`.

``chi_max`` (float [default = 100.0])
  Upper allowed bound for heat conductivities :math:`\chi` across all models,
  in units of :math:`m^2/s`.

``D_e_min`` (float [default = 0.05])
  Lower allowed bound for particle diffusivity :math:`D` across all models,
  in units of :math:`m^2/s`.

``D_e_max`` (float [default = 100.0])
  Upper allowed bound for particle diffusivity :math:`D` across all models,
  in units of :math:`m^2/s`.

``V_e_min`` (float [default = -50.0])
  Lower allowed bound for particle convection :math:`V` across all models,
  in units of :math:`m^2/s`.

``V_e_max`` (float [default = 50.0])
  Upper allowed bound for particle convection :math:`V` across all models,
  in units of :math:`m^2/s`.

``smoothing_width`` (float [default = 0.0])
  Width of HWHM Gaussian smoothing kernel operating on combined transport model
  outputs. Note that if a QLKNN surrogate model (such as ``qlknn`` / ``QLKNN_7_11``)
  is used and ``smoothing_width`` is set to ``0.0``, a warning is logged
  recommending non-zero smoothing to avoid sharp numerical artifacts.

``pedestal_smoothing_width`` (float [default = 0.0])
  Width of HWHM Gaussian smoothing kernel operating on pedestal transport model
  outputs. If set to ``0.0``, no smoothing is applied to pedestal models.

``smoothing_zones`` (list[dict] [default = []])
  Optional list of configuration dictionaries specifying radial zones with
  distinct Gaussian smoothing widths. Each zone dictionary specifies
  ``rho_min``, ``rho_max``, and ``smoothing_width``. When ``smoothing_zones`` is
  provided, the smoothing kernel width :math:`\sigma(\hat{\rho})` varies
  spatially across the radial grid: inside each defined zone
  :math:`(\text{rho\_min}, \text{rho\_max}]`, the
  zone's specific ``smoothing_width`` is used, while outside all defined zones,
  the smoothing width is set to ``0.0`` (no smoothing). The top-level
  ``smoothing_width`` parameter is used only when ``smoothing_zones`` is empty.

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
then using another model specifically for particle transport within
a restricted range, e.g. towards the LCFS.

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
          'model_name': 'set_T_ped_n_ped',
          'rho_norm_ped_top': 0.9,
          'n_e_ped': 0.8,
          'n_e_ped_is_fGW': True,
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
docs/scripts/combined_transport_example.py.

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
``pedestal_transport_models``. The following runtime parameters are common to
all component models:

``model_name`` (str)
  Select the component transport model according to the following options:

* ``'prescribed'``
   Prescribed (constant or space/time-varying) transport coefficients.
* ``'cgm'``
   Critical Gradient Model.
* ``'bohm_gyrobohm'``
   Bohm-GyroBohm model.
* ``'qlknn'``
   A QuaLiKiz Neural Network surrogate model, the default is
   `QLKNN_7_11 <https://github.com/google-deepmind/fusion_surrogates>`_.
* ``'tglfnn_ukaea'``
   A TGLF Neural Network surrogate model developed by UKAEA,
   `reimplemented in JAX <https://github.com/google-deepmind/fusion_surrogates>`_.
* ``'qualikiz'``
   The `QuaLiKiz <https://gitlab.com/qualikiz-group/QuaLiKiz>`_ quasilinear
   gyrokinetic transport model.
* ``'tglf'``
   The TGLF quasilinear turbulent transport model.

``rho_min`` (**time-varying-scalar**  [default = 0.0])
  :math:`\hat{\rho}` above which the transport model is applied. For
  ``rho_min > 0``, the model will be active in the range
  ``rho_min < rho <= rho_max``. For ``rho_min == 0``, it will be active in the
  range ``rho_min <= rho <= rho_max``.
  Note that ``rho_min`` and ``rho_max`` must have the same interpolation mode
  to simplify the validation test ``rho_min < rho_max`` at all times.

``rho_max`` (**time-varying-scalar**  [default = 1.0])
  :math:`\hat{\rho}` below which the transport model is applied. See comment
  about ``rho_min`` for more detail.
  Note that ``rho_min`` and ``rho_max`` must have the same interpolation mode
  to simplify the validation test ``rho_min < rho_max`` at all times.

``merge_mode`` (str [default = 'add'])
  Combination mode for this component: ``'add'`` or ``'overwrite'``.

``disable_chi_i``, ``disable_chi_e``, ``disable_D_e``, ``disable_V_e``
(bool [default = False])
  If ``True``, disables contribution from this channel to the overall transport
  coefficients.

``fast_ion_stabilization`` (**time-varying-scalar** [default = False])
  If ``True``, apply a fast ion stabilization correction to the :math:`R/L_{Ti}`
  input of quasilinear transport models (QLKNN, TGLFNN, QuaLiKiz). The fast ion
  stabilization model predicts an ITG threshold upshift factor based on fast ion
  parameters (:math:`\hat{s}`, :math:`q`, :math:`n_{fi}/n_e`,
  :math:`T_{fi}/T_e`, :math:`R/L_{T_{fi}}`). The input :math:`R/L_{Ti}` is
  divided by this factor, reducing the effective gradient seen by the transport
  model. For multiple fast ion species, the product of per-species factors is
  used. Requires ``numerics.enable_fast_ions`` to be ``True``.

``fast_ion_stabilization_multiplier`` (float [default = 2.0])
  Multiplier for the fast ion stabilization factor. The raw model prediction is
  multiplied by this value before being applied to :math:`R/L_{Ti}`. The default
  of 2.0 accounts for the fact that true gyrokinetic simulations predict
  stronger stabilization than the QuaLiKiz-based surrogate model.

``fast_ion_stabilization_model`` (str [default = ''])
  Name or path of the fast ion stabilization model. If empty (default), the
  default model for each species is used. If a registered model name (e.g.
  ``'fast_ion_H_v1'``), it is loaded from the model registry. Otherwise treated
  as a file path to a ``.fistab`` model file.

prescribed
==========

Runtime parameters for the prescribed transport model. This
model can be used to implement constant coefficients (e.g. ``chi_i`` = 1.0 for
all rho), as well as time-varying transport profiles of arbitrary form (such as
an exponential decay) using the time-varying-array syntax.

``chi_i`` (**time-varying-array** [default = 1.0])
  Ion heat conductivity. In units of :math:`m^2/s`.

``chi_e`` (**time-varying-array** [default = 1.0])
  Electron heat conductivity. In units of :math:`m^2/s`.

``D_e`` (**time-varying-array** [default = 1.0])
  Electron particle diffusion. In units of :math:`m^2/s`.

``V_e`` (**time-varying-array** [default = -0.33])
  Electron particle convection. In units of :math:`m^2/s`.

cgm
===

Runtime parameters for the Critical Gradient Model (CGM).

``alpha`` (float [default = 2.0])
  Exponent of chi power law:
  :math:`\chi \propto (R/L_{Ti} - R/L_{Ti_crit})^\alpha`.

``chi_stiff`` (float [default = 2.0])
  Stiffness parameter.

``chi_e_i_ratio`` (**time-varying-scalar** [default = 2.0])
  Ratio of ion to electron heat conductivity. ITG turbulence has values above 1.

``chi_D_ratio`` (**time-varying-scalar** [default = 5.0])
  Ratio of ion heat conductivity to electron particle diffusion.

``VR_D_ratio`` (**time-varying-scalar** [default = 0.0])
  Ratio of major radius :math:`\times` electron particle convection to electron
  particle diffusion. Sets the electron particle convection in the model.
  Negative values will set a peaked electron density profile in the absence of
  sources.

bohm_gyrobohm
=============

Runtime parameters for the Bohm-GyroBohm model.

``chi_e_bohm_coeff`` (**time-varying-scalar** [default = 8e-5])
  Prefactor for Bohm term for electron heat conductivity.

``chi_e_gyrobohm_coeff`` (**time-varying-scalar** [default = 5e-6])
  Prefactor for GyroBohm term for electron heat conductivity.

``chi_i_bohm_coeff`` (**time-varying-scalar** [default = 8e-5])
  Prefactor for Bohm term for ion heat conductivity.

``chi_i_gyrobohm_coeff`` (**time-varying-scalar** [default = 5e-6])
  Prefactor for GyroBohm term for ion heat conductivity.

``chi_e_bohm_multiplier`` (**time-varying-scalar** [default = 1.0])
  Multiplier for Bohm term for electron heat conductivity. Intended for
  user-friendly default modification.

``chi_e_gyrobohm_multiplier`` (**time-varying-scalar** [default = 1.0])
  Multiplier for GyroBohm term for electron heat conductivity. Intended for
  user-friendly default modification.

``chi_i_bohm_multiplier`` (**time-varying-scalar** [default = 1.0])
  Multiplier for Bohm term for ion heat conductivity. Intended for
  user-friendly default modification.

``chi_i_gyrobohm_multiplier`` (**time-varying-scalar** [default = 1.0])
  Multiplier for GyroBohm term for ion heat conductivity. Intended for
  user-friendly default modification.

``D_face_c1`` (**time-varying-scalar** [default = 1.0])
  Constant for the electron diffusivity weighting factor.

``D_face_c2`` (**time-varying-scalar** [default = 0.3])
  Constant for the electron diffusivity weighting factor.

``V_face_coeff`` (**time-varying-scalar** [default = -0.1])
  Proportionality factor between convectivity and diffusivity.

qlknn
=====

Runtime parameters for the QLKNN model. These parameters determine which model
to load, as well as model parameters. To determine which model to load,
TORAX uses the following logic:

* If ``model_path`` is provided, then we load the model from this path.
* Otherwise, if ``qlknn_model_name`` is provided, we load that model from
  registered models in the ``fusion_surrogates`` library.
* If ``qlknn_model_name`` is not set either, we load the default QLKNN model
  from ``fusion_surrogates`` (currently ``QLKNN_7_11``).

It is recommended to not set ``qlknn_model_name``,  or
``model_path`` to use the default QLKNN model.

``model_path`` (str [default = ''])
  Path to the model. Takes precedence over ``qlknn_model_name``.

``qlknn_model_name`` (str [default = ''])
  Name of the model. Used to select a model from the ``fusion_surrogates``
  library.

``include_ITG`` (bool [default = True])
  If ``True``, include ITG modes in the total fluxes.

``include_TEM`` (bool [default = True])
  If ``True``, include TEM modes in the total fluxes.

``include_ETG`` (bool [default = True])
  If ``True``, include ETG modes in the total electron heat flux.

``ITG_flux_ratio_correction`` (float [default = 1.0])
  Increase the electron heat flux in ITG modes by this factor.
  If using ``QLKNN10D``, the default is 2.0. It is a proxy for the impact of the
  upgraded QuaLiKiz collision operator, in place since ``QLKNN10D`` was
  developed.

``ETG_correction_factor`` (float [default = 1.0/3.0])
  Correction factor for ETG electron heat flux.
  https://gitlab.com/qualikiz-group/QuaLiKiz/-/commit/5bcd3161c1b08e0272ab3c9412fec7f9345a2eef

``clip_inputs`` (bool [default = False])
  Whether to clip inputs within desired margin of the QLKNN training set
  boundaries.

``clip_margin`` (float [default = 0.95])
  Margin to clip inputs within desired margin of the QLKNN training set
  boundaries.

``collisionality_multiplier`` (float [default = 1.0])
  Collisionality multiplier.
  If using ``QLKNN10D``, the default is 0.25. It is a proxy for the upgraded
  collision operator in QuaLiKiz, in place since ``QLKNN10D`` was developed.

``max_normalized_collisionality`` (float [default = inf])
  Maximum normalized collisionality (nu_star) passed to the model. For
  QuaLiKiz-based models, nu_star is the ion-electron collision frequency
  normalized to the bounce frequency
  (see `QuaLiKiz docs <https://gitlab.com/qualikiz-group/QuaLiKiz/-/wikis/QuaLiKiz/Input%20and%20output%20variables#debug/>`_).
  Acts as a ceiling to mitigate unreliable transport predictions at high
  collisionality. The ceiling is applied after the
  ``collisionality_multiplier``. Set to a finite value (e.g. 1.0) to cap
  collisionality in the collisional regime.

``avoid_big_negative_s`` (bool [default = True])
  If ``True``, modify input magnetic shear such that
  :math:`\hat{s} - \alpha_{MHD} > -0.2` always, to compensate for the lack of
  slab ITG modes in QuaLiKiz.

``smag_alpha_correction`` (bool [default = True])
  If ``True``, reduce input magnetic shear by :math:`0.5*\alpha_{MHD}` to
  capture the main impact of :math:`\alpha_{MHD}`, which was not itself part of
  the ``QLKNN`` training set.

``q_sawtooth_proxy`` (bool [default = True])
  To avoid un-physical transport barriers, modify the input q-profile and
  magnetic shear for zones where :math:`q < 1`, as a proxy for sawteeth.
  Where :math:`q<1`, then the :math:`q` and :math:`\hat{s}` ``QLKNN`` inputs are
  clipped to :math:`q=1` and :math:`\hat{s}=0.1`.

``DV_effective`` (bool [default = False])
  If ``True``, use either :math:`D_{eff}` or :math:`V_{eff}` for particle
  transport. See :ref:`physics_models` for more details.

``An_min`` (float [default = 0.05])
  :math:`|R/L_{ne}|` value below which :math:`V_{eff}` is used instead of
  :math:`D_{eff}`, if ``DV_effective==True``.

``DV_effective_smooth_width`` (float [default = 0.01])
  Width in dimensionless GyroBohm-normalized particle flux units
  (:math:`\Gamma_e / \Gamma_{GB}`) over which down-gradient transport
  transitions smoothly from :math:`V_{eff}` to :math:`D_{eff}`. If ``0.0``, uses
  a sharp step-function transition. Note that QuaLiKiz normalizes using major
  radius :math:`R_{major}` rather than minor radius :math:`a`, so the default
  (:math:`0.01`) corresponds roughly to the TGLF default (:math:`0.001`) in SI
  units.

``rotation_multiplier`` (float [default = 1.0])
  Multiplier for :math:`v_{E\times B}` in the rotation correction factor.

``rotation_mode`` (str [default = 'off'])
  Defines how the rotation correction is applied. Options are:

  * ``off``: No rotation correction is applied.

  * ``half_radius``: The rotation correction is only applied to the outer
    half of the radius (:math:`\hat{\rho} > 0.5`).

  * ``full_radius``: The rotation correction is applied everywhere.

``shear_suppression_alpha`` (float [default = 1.0])
  Alpha parameter for the Waltz rule shear suppression applied to
  poloidal/pressure contributions.

``output_mode_contributions`` (bool [default = False])
  If ``True``, output individual ITG/TEM/ETG contributions to transport
  coefficients in addition to the total values.

tglfnn_ukaea
============

Runtime parameters for the TGLFNN-UKAEA model. If you use this model, please cite
`L. Zanisi et al. (2025) <https://conferences.iaea.org/event/392/contributions/36059/>`_.

``machine`` (str [default = 'multimachine'])
  Machine type for TGLFNN-UKAEA. Options are 'multimachine', and 'step'.
  The 'multimachine' model is trained on a hypercube covering a wide variety
  of conventional tokamaks. The 'step' model is trained specifically on the
  design space of UKAEA's Spherical Tokamak for Energy Production (STEP).
  See `ukaea/tglfnn-ukaea <https://github.com/ukaea/tglfnn-ukaea>`_ for more
  details. Note: currently, only the 'multimachine' model is publicly available.

``DV_effective`` (bool [default = False])
  If ``True``, use either :math:`D_{eff}` or :math:`V_{eff}` for particle
  transport. See :ref:`physics_models` for more details.

``An_min`` (float [default = 0.05])
  :math:`|R/L_{ne}|` value below which :math:`V_{eff}` is used instead of
  :math:`D_{eff}`, if ``DV_effective==True``.

``DV_effective_smooth_width`` (float [default = 0.001])
  Width in dimensionless GyroBohm-normalized particle flux units
  (:math:`\Gamma_e / \Gamma_{GB}`) over which down-gradient transport
  transitions smoothly from :math:`V_{eff}` to :math:`D_{eff}`. If ``0.0``, uses
  a sharp step-function transition. Note that TGLF normalizes using minor
  radius :math:`a` rather than :math:`R_{major}`, so the default
  (:math:`0.001`) corresponds roughly to the QuaLiKiz default (:math:`0.01`) in
  SI units.

``rotation_multiplier`` (float [default = 1.0])
  Multiplier for :math:`v_{E\times B}^{\text{shear}}`.

``use_rotation`` (bool [default = False])
  If ``True``, use the rotation term :math:`v_{E\times B}^{\text{shear}}` in the model.

``collisionality_multiplier`` (float [default = 1.0])
  Collisionality multiplier for sensitivity analysis.

``max_normalized_collisionality`` (float [default = inf])
  Maximum dimensionless electron-electron collision frequency (XNUE in TGLF
  notation) passed to the model. See the `TGLF docs
  <https://gacode.io/tglf/tglf_list.html>`_ . Acts as a ceiling to mitigate
  unreliable transport predictions at high collisionality. The ceiling is
  applied after the ``collisionality_multiplier``.

qualikiz
========

Runtime parameters for the QuaLiKiz model.

``n_max_runs`` (int [default = 2])
  Frequency of full QuaLiKiz contour solutions. For n_max_runs>1, every
  n_max_runs-th call will use the full contour integral solution. Other runs
  will use the previous solution as the initial guess for the Newton solver,
  which is significantly faster.

``n_processes`` (int [default = 8])
  Number of MPI processes to use for QuaLiKiz.

``collisionality_multiplier`` (float [default = 1.0])
  Collisionality multiplier for sensitivity analysis.

``max_normalized_collisionality`` (float [default = inf])
  Maximum normalized collisionality (nu_star) passed to the model. For
  QuaLiKiz-based models, nu_star is the ion-electron collision frequency
  normalized to the bounce frequency
  (see `QuaLiKiz docs <https://gitlab.com/qualikiz-group/QuaLiKiz/-/wikis/QuaLiKiz/Input%20and%20output%20variables#debug/>`_).
  Acts as a ceiling to mitigate unreliable transport predictions at high
  collisionality. The ceiling is applied after the
  ``collisionality_multiplier``. Set to a finite value (e.g. 1.0) to cap
  collisionality in the collisional regime.

``avoid_big_negative_s`` (bool [default = True])
  If ``True``, modify input magnetic shear such that
  :math:`\hat{s} - \alpha_{MHD} > -0.2` always, to compensate for the lack of
  slab ITG modes in QuaLiKiz.

``smag_alpha_correction`` (bool [default = True])
  If ``True``, reduce input magnetic shear by :math:`0.5*\alpha_{MHD}` to
  capture the main impact of :math:`\alpha_{MHD}`, which was not itself part of
  the ``QLKNN`` training set.

``q_sawtooth_proxy`` (bool [default = True])
  To avoid un-physical transport barriers, modify the input q-profile and
  magnetic shear for zones where :math:`q < 1`, as a proxy for sawteeth. Where
  :math:`q<1`, then the :math:`q` and :math:`\hat{s}` QuaLiKiz inputs are
  clipped to :math:`q=1` and :math:`\hat{s}=0.1`.

``DV_effective`` (bool [default = False])
  If ``True``, use either :math:`D_{eff}` or :math:`V_{eff}` for particle
  transport. See :ref:`physics_models` for more details.

``An_min`` (float [default = 0.05])
  :math:`|R/L_{ne}|` value below which :math:`V_{eff}` is used instead of
  :math:`D_{eff}`, if ``DV_effective==True``.

``DV_effective_smooth_width`` (float [default = 0.01])
  Width in dimensionless GyroBohm-normalized particle flux units
  (:math:`\Gamma_e / \Gamma_{GB}`) over which down-gradient transport
  transitions smoothly from :math:`V_{eff}` to :math:`D_{eff}`. If ``0.0``, uses
  a sharp step-function transition. Note that QuaLiKiz normalizes using major
  radius :math:`R_{major}` rather than minor radius :math:`a`, so the default
  (:math:`0.01`) corresponds roughly to the TGLF default (:math:`0.001`) in SI
  units.


tglf
====

Runtime parameters for the TGLF model. If you want to use TORAX with TGLF, see
:ref:`tglf_installation` for installation instructions.

``n_processes`` (int [default = 8])
  Number of parallel TGLF calculations to run across radial faces.

``n_cores_per_process`` (int [default = 2])
  Number of cores to use for each TGLF process.

``use_rotation`` (bool [default = False])
  Toggles the use of the rotation shear model.

``rotation_multiplier`` (float [default = 1.0])
  Multiplier for the input rotation shear.

``DV_effective`` (bool [default = False])
  If ``True``, use either :math:`D_{eff}` or :math:`V_{eff}` for particle
  transport. See :ref:`physics_models` for more details.

``An_min`` (float [default = 0.05])
  :math:`|R/L_{ne}|` value below which :math:`V_{eff}` is used instead of
  :math:`D_{eff}`, if ``DV_effective==True``.

``DV_effective_smooth_width`` (float [default = 0.001])
  Width in dimensionless GyroBohm-normalized particle flux units
  (:math:`\Gamma_e / \Gamma_{GB}`) over which down-gradient transport
  transitions smoothly from :math:`V_{eff}` to :math:`D_{eff}`. If ``0.0``, uses
  a sharp step-function transition. Note that TGLF normalizes using minor
  radius :math:`a` rather than :math:`R_{major}`, so the default
  (:math:`0.001`) corresponds roughly to the QuaLiKiz default (:math:`0.01`) in
  SI units.

``collisionality_multiplier`` (float [default = 1.0])
  Collisionality multiplier.

``max_normalized_collisionality`` (float [default = inf])
  Maximum dimensionless electron-electron collision frequency (XNUE in TGLF
  notation) passed to the model. See the `TGLF docs
  <https://gacode.io/tglf/tglf_list.html>`_ . Acts as a ceiling to mitigate
  unreliable transport predictions at high collisionality. The ceiling is
  applied after the ``collisionality_multiplier``.

``tglf_settings`` (dict [default = {}])
  Dictionary of TGLF namelist parameters to override default physics settings.

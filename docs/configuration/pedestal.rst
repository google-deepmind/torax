.. _pedestal_config:

pedestal
########

The ``pedestal`` dictionary configures the pedestal model, which is composed of
three sub-models:

* ``structure_model``: computes the H-mode target pedestal-top location
  (``rho_norm_top``) and pedestal-top electron density, ion temperature,
  and electron temperature.
* ``formation_model``: determines when L-H and H-L transitions occur (and in
  ``'adaptive_transport'`` mode, the transport decrease multiplier).
* ``saturation_model``: determines how transport scales in the pedestal region
  relative to the H-mode target values in ``'adaptive_transport'`` mode.

.. torax_config:: torax._src.pedestal_model.config.PedestalConfig

structure_model
===============

.. torax_config:: torax._src.pedestal_model.config.NoPedestalConfig
   :include-fields: model_name

.. torax_config:: torax._src.pedestal_model.config.StructureConfigBase

no_pedestal
-----------

Disables the pedestal across the entire ``PedestalModel`` regardless of
``formation_model`` or ``saturation_model``.

set_T_top_n_top
---------------

.. torax_config:: torax._src.pedestal_model.config.SetTtopNtopConfig
   :include-class-doc:
   :own-fields-only:
   :exclude-fields: model_name

set_p_top_n_top
---------------

.. torax_config:: torax._src.pedestal_model.config.SetPtopTtopRatioNtopConfig
   :include-class-doc:
   :own-fields-only:
   :exclude-fields: model_name

formation_model
===============

.. torax_config:: torax._src.pedestal_model.config.PrescribedFormationConfig
   :include-fields: model_name

.. torax_config:: torax._src.pedestal_model.config.FormationConfigBase

prescribed
----------

.. torax_config:: torax._src.pedestal_model.config.PrescribedFormationConfig
   :include-class-doc:
   :own-fields-only:
   :exclude-fields: model_name

martin_scaling
--------------

.. torax_config:: torax._src.pedestal_model.config.MartinScalingFormationConfig
   :include-class-doc:
   :exclude-fields: model_name, base_multiplier, initial_confinement_mode

delabie_scaling
---------------

.. torax_config:: torax._src.pedestal_model.config.DelabieScalingFormationConfig
   :include-class-doc:
   :exclude-fields: model_name, base_multiplier, initial_confinement_mode

saturation_model
================

.. torax_config:: torax._src.pedestal_model.config.ProfileValueSaturationConfig
   :include-class-doc:

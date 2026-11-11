.. _neoclassical_config:

neoclassical
############

Configures the neoclassical physics model.

.. torax_config:: torax._src.neoclassical.config.AnalyticalNeoclassicalConfig

bootstrap_current
=================

.. torax_config:: torax._src.neoclassical.bootstrap_current.sauter.SauterModelConfig
   :include-fields: model_name, bootstrap_multiplier

conductivity
============

.. torax_config:: torax._src.neoclassical.conductivity.sauter.SauterModelConfig

transport
=========

.. torax_config:: torax._src.neoclassical.transport.angioni_sauter.AngioniSauterModelConfig
   :include-fields: model_name

.. torax_config:: torax._src.neoclassical.transport.base.NeoclassicalTransportModelConfig

zeros
-----

Sets neoclassical transport coefficients to zero.

angioni_sauter
--------------

Computes neoclassical transport coefficients according to the Angioni-Sauter
|angioni2000| model.

.. torax_config:: torax._src.neoclassical.transport.angioni_sauter.AngioniSauterModelConfig
   :own-fields-only:
   :exclude-fields: model_name

poloidal_velocity
=================

.. torax_config:: torax._src.neoclassical.poloidal_velocity.kim.KimModelConfig
   :include-fields: model_name, poloidal_velocity_multiplier

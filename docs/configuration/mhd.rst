.. _mhd_config:

mhd
###

Configuration for MHD models. Currently only the sawtooth model is implemented.
If the ``mhd`` key or the nested ``sawtooth`` key is absent or set to ``None``,
the sawtooth model will be disabled.

.. torax_config:: torax._src.mhd.config.MHDConfig

.. _sawtooth_config:

sawtooth
========

.. torax_config:: torax._src.mhd.sawtooth.config.SawtoothConfig

trigger_model
-------------

.. torax_config:: torax._src.mhd.sawtooth.simple_trigger.SimpleTriggerConfig

redistribution_model
--------------------

.. torax_config:: torax._src.mhd.sawtooth.simple_redistribution.SimpleRedistributionConfig

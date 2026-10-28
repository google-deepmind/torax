.. _mhd_config:

mhd
###

Configuration for MHD models. Currently only the sawtooth model is implemented.
If the mhd key or the nested sawtooth key is absent or set to None, the sawtooth
model will be disabled.

.. _sawtooth_config:

sawtooth
========

``model_name`` (str [default = 'simple'])
  Currently only 'simple' is supported.

``simple`` trigger model parameters:

* ``s_critical`` (**time-varying-scalar** [default = 0.1])
  The critical magnetic shear value at the q=1 surface. A crash is triggered
  only if the shear exceeds this value.

* ``minimum_radius`` (**time-varying-scalar** [default = 0.05])
  The minimum normalized radius (:math:`\hat{\rho}`) of the q=1 surface required
  to trigger a crash.

``model_name`` (str [default = 'simple'])
  Currently only 'simple' is supported.

``simple`` redistribution model parameters:

*   ``flattening_factor`` (**time-varying-scalar** [default = 1.01]):
    Factor determining the degree of flattening inside the q=1 surface.

*   ``mixing_radius_multiplier`` (**time-varying-scalar** [default = 1.1]):
    Multiplier applied to :math:`\hat{\rho}_{q=1}` to determine the mixing
    radius :math:`\hat{\rho}_{mix}`.

``crash_step_duration`` (float [default = 1e-5]):
  Duration of a sawtooth crash step in seconds. This is how much the solver time
  will be bumped forward during a crash. It is important that this is is small
  enough to satisfy the assumption of timescale separation between the sawtooth
  crash and the transport timescale, otherwise issues may arise regarding
  conservation properties.

.. _restart_config:

restart
#######

If this optional entry is supplied in the config, TORAX will be run starting
from the selected state in a previous simulation. This can be used to resume a
completed simulation.

If the requested restart time is not exactly available in the state file, the
simulation will restart from the closest available time. A warning will be
logged in this case.

.. torax_config:: torax._src.torax_pydantic.file_restart.FileRestartConfig

For example, following a simulation in which a state file is saved to
``/path/to/torax_state_file.nc``, a new simulation can be started from the old
state at ``t=10`` by adding the following to the config:

.. code-block:: python

  'restart': {
      'filename': '/path/to/torax_state_file.nc',
      'time': 10,
      'do_restart': True,
      'stitch': True,
  }

The resulting simulation will recreate the state from ``t=10`` in the
previous simulation, and then run forwards from that point in time. For
all subsequent steps the runtime parameters will be constructed using
the runtime parameter configuration provided in the new config (from ``t=10``
onwards).

Some potential use cases for this feature include:

* Restarting a simulation that was healthy up to a certain time and then failed.
  After fixing the cause of the failure, a new simulation can be run from the
  last healthy point.

* Performing uncertainty quantification by sweeping a set of configs starting
  from the same initial plasma state. After performing an initial simulation,
  the restart option could be used to run a set of configs with modified
  parameters to evaluate the impact of their variation.

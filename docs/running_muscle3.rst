.. _running_muscle3:

Running simulations with Muscle3
################################

For general MUSCLE3 workflow instructions, see the `MUSCLE3 documentation <https://muscle3.readthedocs.io/en/latest/>`_.

This page shows the specifications for the MUSCLE3 actor running the Torax core transport simulator.

Available Operational Modes
---------------------------

- ***Torax actor***: Default.

.. code-block:: bash

  implementations:
    torax:
      executable: python
      args: "-u -m torax._src.orchestration.run_muscle3"

Available Settings
------------------

* Mandatory

  - ***python_config_module***: (string) configuration module for torax

* Optional

  - ***output_all_timeslices***: (string) IMAS Data Dictionary version number to which data will be converted. Defaults to original dd_version of the data.

Available Ports
---------------

The Torax actor currently only has equilibrium IDS input and output.

* Optional

  - ***equilibrium_f_init (F_INIT)***: equilibrium IDS as initial input.
  - ***equilibrium_o_i (O_I)***: equilibrium IDS as inner loop output.
  - ***equilibrium_s (S)***: equilibrium IDS as inner loop input.
  - ***equilibrium_o_f (O_F)***: equilibrium IDS as final output.

General
-------
The torax actor can be used with IMAS DDv4.

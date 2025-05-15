.. _quickstart:

Quickstart to Running and Plotting
##################################

This page describes a minimal demonstration of running and plotting a TORAX run
using the pre-built ``run_torax`` script. See :ref:`running` for more details.

It is convenient to set up a Python virtual environment for running TORAX, as
described in :ref:`installation`.

Running a basic example
=======================

The following command will run TORAX using the configuration file
``examples/basic_config.py``.

.. code-block:: console

  run_torax --config='examples/basic_config.py'

Simulation progress is shown by a progress bar in the terminal, displaying the
current simulation time, and the percentage of the total simulation time
completed.

More involved examples in ``torax/examples`` include non-rigorous mockups of the
ITER hybrid scenario:

* ``iterhybrid_predictor_corrector.py``: flattop phase with the linear stepper
  using predictor-corrector iterations.

* ``iterhybrid_rampup.py``: time-dependent ramppup phase with the nonlinear
  Newton-Raphson stepper.

To run one of these, run for example:

.. code-block:: console

  run_torax --config='examples/iterhybrid_rampup.py'

The configuration files can be inspected on
`GitHub <https://github.com/google-deepmind/torax/tree/main/torax/examples>`_ or
by cloning the repository.

Post-simulation
---------------

Once complete, the time history of a simulation state and derived quantities
is written to a timestamped file of the format
``state_history_%Y%m%d_%H%M%S.nc``.

The output directory is user configurable, with a default
``/tmp/torax_results``. The ``output_dir`` flag overrides the default output
directory, e.g.

.. code-block:: console

  run_torax --config='examples/basic_config.py' --output_dir=/path/to/output/dir

To take advantage of the in-memory (non-persistent) cache, the process does not
end upon simulation termination. Instead, various options are provided in a
menu. See :ref:`running` for more details. Most pertinent for this minimum
demonstration is simulation plotting, selected with the ``pr`` user command,
followed by ``0`` to plot the last run.

Simulation plotting script
--------------------------

Beyond plotting via the ``run_torax`` script post-simulation, a standalone
plotting script is also available.

To plot the output of a single simulation, run the following command:

.. code-block:: console

  plot_torax --outfile <full_path_to_simulation_output>

Replace ``<full_path_to_simulation_output>`` with the full path to your
simulation's output file. Optionally, specify a custom plot configuration using
``--plot_config``, with the path for the plotting configuration module.
If no ``--plot_config`` is specified, the default configuration at
``plotting/configs/default_plot_config.py`` is used. See :ref:`plotting` for
more details.

A slider allows to scroll through the output of all simulation timesteps.

Cleaning up
-----------

If in one, you can exit the Python virtual env by deactivating it:

.. code-block:: console

  deactivate

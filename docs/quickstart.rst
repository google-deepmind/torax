.. _quickstart:

Quickstart to Running and Plotting
##################################
Running an example
==================
It is convenient to set up a Python virtual environment for running TORAX, as described in :ref:`installation`.

The following command will run TORAX using the configuration file ``torax/examples/basic_config.py``.
TORAX configuration files overwrite the defaults in ``config.py``. See :ref:`configuration` for details
of all input configuration fields.

.. code-block:: console

  run_torax --config='torax.examples.basic_config'

Simulation progress is shown by a progress bar in the terminal, displaying the current
simulation time, and the percentage of the total simulation time completed.

More involved examples in ``torax/examples`` include non-rigorous mockups of the ITER hybrid scenario:

* ``iterhybrid_predictor_corrector.py``: flattop phase with the linear stepper using predictor-corrector iterations.

* ``iterhybrid_rampup.py``: time-dependent ramppup phase with the nonlinear Newton-Raphson stepper.

Additional configuration is provided through flags which append the above run command, and environment variables.

Set environment variables
-------------------------
All environment variables can be set in shell configuration scripts, e.g. ``.bashrc``, or by shell prompt commands.

TORAX_QLKNN_MODEL_PATH
^^^^^^^^^^^^^^^^^^^^^^^
Path to the QuaLiKiz-neural-network parameters. The path specified here
will be ignored if the ``model_path`` field in the ``qlknn_params`` section of
the run config file is set.

.. code-block:: console

  export TORAX_QLKNN_MODEL_PATH="<myqlknnmodelpath>"

TORAX_GEOMETRY_DIR
^^^^^^^^^^^^^^^^^^
Path to the geometry file directory. This prefixes the path and filename provided in the ``geometry_file``
geometry constructor argument in the run config file. If not set, ``TORAX_GEOMETRY_DIR`` defaults to the
relative path ``torax/data/third_party/geo``.

.. code-block:: console

  export TORAX_GEOMETRY_DIR="<mygeodir>"

TORAX_ERRORS_ENABLED
^^^^^^^^^^^^^^^^^^^^
If true, error checking is enabled in internal routines. Used for debugging.
Default is false since it is incompatible with the persistent compilation cache.

.. code-block:: console

  export TORAX_ERRORS_ENABLED=<True/False>

TORAX_COMPILATION_ENABLED
^^^^^^^^^^^^^^^^^^^^^^^^^
If false, JAX does not compile internal TORAX functions. Used for debugging. Default is true.

.. code-block:: console

  export TORAX_COMPILATION_ENABLED=<True/False>

Set flags
---------

.. _log_progress_quickstart:

log_progress
^^^^^^^^^^^^
Increased output verbosity. Logs, for each timestep (dt), the current simulation
time, dt, and number of outer stepper iterations carried out during the step.

.. code-block:: console

  run_torax
  --config='torax.examples.basic_config' \
   --log_progress

plot_progress
^^^^^^^^^^^^^
Live plotting of simulation state and derived quantities as the simulation progresses.

.. code-block:: console

  run_torax \
   --config='torax.examples.basic_config' \
   --plot_progress

For a combination of the above:

.. code-block:: console

  run_torax \
  --config='torax.examples.basic_config' \
  --log_progress --plot_progress

reference_run
^^^^^^^^^^^^^
Provide a reference run to compare against in post-simulation plotting.

.. code-block:: console

  run_torax \
  --config='torax.examples.basic_config' \
  --reference_run=<path_to_reference_run>

output_dir
^^^^^^^^^^
Override the default output directory. If not provided, it will be set to
``output_dir`` defined in the config. If that is not defined, will default to
``'/tmp/torax_results_<YYYYMMDD_HHMMSS>/'``.

.. code-block:: console

  run_torax \
  --config='torax.examples.basic_config' \
  --output_dir=<output_dir>

plot_config
^^^^^^^^^^^
Sets the plotting configuration used for the post-simulation plotting options.
This flag should point to a python module path containing a `PLOT_CONFIG` variable
which is an instance of `torax.plotting.plotruns_lib.FigureProperties`.
By default, `torax.plotting.configs.default_plot_config` is used.
See :ref:`plotting` for further details and examples. An example using a non-default
plot config is shown below.

.. code-block:: console

  run_torax \
  --config='torax.examples.basic_config' \
  --plot_config=torax.plotting.configs.simple_plot_config

Post-simulation
---------------

Once complete, the time history of a simulation state and derived quantities is
written to ``state_history.nc``. For convenience, the output path is written to stdout.

To take advantage of the in-memory (non-persistent) cache, the process does not end upon
simulation termination. Instead, various options are provied to the user:

* Modify the config
* Rerun the simulation
* Toggle the ``log_progress`` or ``plot_progress`` flags
* Plot the output of the last simulation (against another) (see :ref:`running`)
* Quit

When modifying the config and then rerunning the simulation, most config modifications will not
trigger recompilation. However, modifications to the following elements will trigger a recompilation:

* Grid resolution
* Evolved variables (equations being solved)
* Changing internal functions used, e.g. transport model, sources, or time_step_calculator

Simulation plotting
-------------------

To plot the output of a single simulation, run the following command from the TORAX
root directory:

.. code-block:: console

  plot_torax --outfile <full_path_to_simulation_output> \
   --plot_config <module_path_to_plot_config>

Replace <full_path_to_simulation_output> with the full path to your simulation's
output file. Optionally, specify a custom plot configuration module using
``--plot_config``, with the module path for the plotting configuration module.
If no ``--plot_config`` is specified, the default configuration at
``torax.plotting.configs.default_plot_config`` is used.

A slider allows to scroll through the output of all simulation timesteps.

To plot the output of two simulations on top of each other, run the following command:

.. code-block:: console

  plot_torax --outfile <full_path_to_simulation_output1> \
   <full_path_to_simulation_output2> --plot_config <module_path_to_plot_config>


Cleaning up
-----------

If in one, you can get out of the Python virtual env by deactivating it:

.. code-block:: console

  deactivate

Quickstart to Running and Plotting
##################################
Running an example
==================
It is convenient to set up a Python virtual environment for running TORAX, as described in :ref:`installation`.

The following command will run TORAX using the configuration file ``tests/test_data/default_config.py``.
TORAX configuration files overwrite the defaults in ``config.py``. See :ref:`configuration` for details
of all input configuration fields.

.. code-block:: console

  python3 run_simulation_main.py \
     --python_config='torax.tests.test_data.default_config' --log_progress

Additional configuration is provided through flags which append the above run command, and environment variables.

Set environment variables
-------------------------
All environment variables can be set in shell configuration scripts, e.g. ``.bashrc``, or by shell prompt commands.

TORAX_QLKNN_MODEL_PATH
^^^^^^^^^^^^^^^^^^^^^^^
Path to the QuaLiKiz-neural-network parameters.

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
log_progress
^^^^^^^^^^^^
Output simulation time, dt, and number of stepper iterations carried out at each timestep.
For nonlinear solvers, the stepper iterations can be more than 1 due to dt backtracking.

.. code-block:: console

  python3 run_simulation_main.py \
   --python_config='torax.tests.test_data.default_config' \
   --log_progress

plot_progress
^^^^^^^^^^^^^
Live plotting of simulation state and derived quantities as the simulation progresses.

.. code-block:: console

  python3 run_simulation_main.py \
   --python_config='torax.tests.test_data.default_config' \
   --plot_progress

For a combination of the above:

.. code-block:: console

  python3 run_simulation_main.py \
  --python_config='torax.tests.test_data.default_config' \
  --log_progress --plot_progress

Post-simulation
---------------

Once complete, the time history of a simulation state and derived quantities is
written to ``state_history.nc``. For convenience, the output path is written to stdout.

To take advantage of the in-memory (non-persistent) cache, the process does not end upon
simulation termination. Instead, various options are provied to the user:

* Modify the config
* Rerun the simulation
* Toggle the ``log_progress`` or ``plot_progress`` flags
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

  python3 plotting/plotruns.py --outfile <full_path_to_simulation_output>

Alternatively, ``plotting/plotruns.py`` can be replaced by the relative path and the
command run from anywhere. The command will plot the following outputs:

* Ion and electron heat conductivity
* Ion and electron temperature
* Electron density
* Total, Ohmic, bootstrap, and external current
* q-profile
* Magnetic shear

A slider allows to scroll through the output of all simulation timesteps.

To plot the output of two simulations on top of each other, run the following command:

.. code-block:: console

  python3 plotting/plotruns.py --outfile <full_path_to_simulation_output1> \\
   <full_path_to_simulation_output2>


Cleaning up
-----------

If in one, you can get out of the Python virtual env by deactivating it:

.. code-block:: console

  deactivate

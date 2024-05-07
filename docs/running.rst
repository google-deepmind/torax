.. running:

Running simulations
###################

It is convenient to set up a Python virtual environment for running TORAX, as described in :ref:`installation`.

The following command will run TORAX using the configuration file ``torax/examples/basic_config.py``.
TORAX configuration files overwrite the defaults in ``config.py``. See :ref:`configuration` for details
of all input configuration fields.

.. code-block:: console

  python3 run_simulation_main.py \
     --config='torax.examples.basic_config' --log_progress

More involved examples in ``torax/examples`` include non-rigorous mockups of the ITER hybrid scenario:

* ``iterhybrid_predictor_corrector.py``: flattop phase with the linear stepper using predictor-corrector iterations.

* ``iterhybrid_rampup.py``: time-dependent ramppup phase with the nonlinear Newton-Raphson stepper.

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
   --config='torax.examples.basic_config' \
   --log_progress

plot_progress
^^^^^^^^^^^^^
Live plotting of simulation state and derived quantities as the simulation progresses.

.. code-block:: console

  python3 run_simulation_main.py \
   --config='torax.examples.basic_config' \
   --plot_progress

For a combination of the above:

.. code-block:: console

  python3 run_simulation_main.py \
  --config='torax.examples.basic_config' \
  --log_progress --plot_progress

Post-simulation
---------------

Once complete, the time history of a simulation state and derived quantities is
written to ``state_history.nc``. The output path is written to stdout. The ``output_dir``
is user-configurable (see :ref:`configuration`). The default is ``'/tmp/torax_results_<YYYYMMDD_HHMMSS>/'``.

To take advantage of the in-memory (non-persistent) cache, the process does not end upon
simulation termination. Instead, the user is presented with the following menu.

  | r: RUN SIMULATION
  | cc: change config for the same sim object (may recompile)
  | cs: change config and build new sim object (will recompile)
  | tlp: toggle --log_progress
  | tpp: toggle --plot_progress
  | tlo: toggle --log_output
  | q: quit

* **cc** will load a new config file, which optionally can be the same config file previously loaded, including any changes that the user has implemented in the interim. If in the new config file, the only different config variables compared to the previous run are `dynamic` variables (see :ref:`dynamic_vs_static`), then the new simulation can be run without recompilation. Static config variables which will trigger recompilation include variables related to:

  * Grid resolution
  * Evolved variables (equations being solved)
  * Changing internal functions used, e.g. transport model, sources, or time_step_calculator

* **cs** will load a new config file, and rebuild the internal Sim object, definitely leading to recompilation when running a new simulation.
* **r** will launch a new run, with a new config if **cs** or **cc** was chosen previously.
* **tlp** toggles the ``--log_progress`` flag for the next run.
* **tpp** toggles the ``--plot_progress`` flag for the next run.
* **tlo** toggles the ``--log_output`` flag for the next run, used for debugging purposes.
* **q** quits the process.

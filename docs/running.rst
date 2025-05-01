.. _running:

Running simulations
###################

It is convenient to set up a Python virtual environment for running TORAX, as described in :ref:`installation`.

The following command will run TORAX using the configuration file ``torax/examples/basic_config.py``.
TORAX configuration files overwrite the defaults in ``config.py``. See :ref:`configuration` for details
of all input configuration fields.

.. code-block:: console

  run_torax --config='torax.examples.basic_config'

Simulation progress is shown by a progress bar in the terminal, displaying the current
simulation time, and the percentage of the total simulation time completed.

Increased logging verbosity is set by the  :ref:`log progress<log_progress_running>` flag and the
:ref:`log_iterations<log_iterations>` variable in the ``solver`` section of the config (for the Newton-Raphson solver).

More involved examples in ``torax/examples`` include non-rigorous mockups of the ITER hybrid scenario:

* ``iterhybrid_predictor_corrector.py``: flattop phase with the linear solver using predictor-corrector iterations.

* ``iterhybrid_rampup.py``: time-dependent ramppup phase with the nonlinear Newton-Raphson solver.

Additional configuration is provided through flags which append the above run command, and environment variables.

Set environment variables
-------------------------
All environment variables can be set in shell configuration scripts, e.g. ``.bashrc``, or by shell prompt commands.

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

.. _log_progress_running:

log_progress
^^^^^^^^^^^^
Log progress for each timestep (dt) the current simulation time, dt, and number of
outer solver iterations carried out during the step. For the Newton-Raphson solver,
the outer solver iterations can be more than 1 due to dt backtracking
(enabled by ``adaptive_dt=True`` in the ``solver`` config dict) when the solver
does not converge within a set number of inner solver iterations.

.. code-block:: console

  run_torax \
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
  | pr: plot previous run(s) or against reference if provided
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
* **pr** provides three options. Plot the last run (0), the last two runs (1), the last run against a reference run (2).
* **q** quits the process.

.. _running:

It is convenient to set up a Python virtual environment for running TORAX, as
described in :ref:`installation`.

Running simulations
###################

TORAX comes with several pre-packaged example configuration files. These can be
inspected on
`GitHub <https://github.com/google-deepmind/torax/tree/main/torax/examples>`_ or
by cloning the repository. See :ref:`configuration` for details of all input
configuration fields.

The following command will run TORAX using the prepackaged configuration file
`basic_config.py <https://github.com/google-deepmind/torax/tree/main/torax/examples/basic_config.py>`_.
When inspecting the configuration file, note that empty dicts as configuration
values or missing variables signify that default values are used. See
:ref:`configuration` for the full list of configuration variables and their
default values.

.. code-block:: console

  run_torax --config='examples/basic_config.py'

Simulation progress is shown by a progress bar in the terminal, displaying the
current simulation time, and the percentage of the total simulation time
completed. Once complete, the time history of a simulation state and
derived quantities is written to a timestamped file of the format
``state_history_%Y%m%d_%H%M%S.nc``, in the output directory specified by
the ``output_dir`` flag (see :ref:`torax_flags`), or the default
``'/tmp/torax_results'`` directory if not specified.

More involved examples in ``torax/examples`` include non-rigorous mockups of the
ITER hybrid scenario:

* ``iterhybrid_predictor_corrector.py``: flattop phase with the linear stepper
  using predictor-corrector iterations.

* ``iterhybrid_rampup.py``: time-dependent rampup phase with the nonlinear
  Newton-Raphson stepper.

To run one of these, run for example:

.. code-block:: console

  run_torax --config='examples/iterhybrid_rampup.py'

To run your own configuration, set the ``--config`` flag to point to the
relative or absolute path of your configuration file. It is required that the
configuration (``.py``) file contains a valid TORAX config dict named
``CONFIG``. An invalid dict will raise informative errors by the underlying
Pydantic library which uses the ``CONFIG`` dict to construct a
`torax.ToraxConfig` object used under the hood.

.. code-block:: console

  run_torax --config=</path/to/your/config.py>

Additional configuration is provided through flags which append the above run
command, and environment variables.

Environment variables
---------------------
All environment variables can be set in shell configuration scripts, e.g.
``.bashrc``, or by shell prompt commands.

``TORAX_ERRORS_ENABLED`` (default: `False`) - If `True`, error checking is enabled
in internal routines. Used for debugging. Default is `False` since it is
incompatible with the persistent compilation cache.

.. code-block:: console

  export TORAX_ERRORS_ENABLED=<True/False>

``TORAX_COMPILATION_ENABLED`` (default: `True`) - If `False`, JAX does not compile
internal TORAX functions. Used for debugging.

.. code-block:: console

  export TORAX_COMPILATION_ENABLED=<True/False>

.. _torax_flags:

run_torax flags
---------------

``log_progress`` (default: `False`) - Logs for each timestep (dt) the current
simulation time, dt, and number of outer solver iterations carried out during
the step. For the Newton-Raphson solver, the outer solver iterations can be more
than 1 due to dt backtracking (enabled by ``adaptive_dt=True`` in the
``numerics`` config dict) when the solver does not converge within a set number
of inner solver iterations.

.. code-block:: console

  run_torax --config='torax.examples.basic_config' --log_progress

``reference_run`` (default: `None`) - Absolute path or relative path
  (relative to the current directory) to a reference run to compare against in
  post-simulation plotting.

.. code-block:: console

  run_torax --config='torax.examples.basic_config' \
  --reference_run=<path/to/reference_run/myoutput.nc>

``output_dir`` (default: `'/tmp/torax_results'`) - Absolute path or relative
  path (relative to the current directory) to a directory where the output files
  will be written in the format ``state_history_%Y%m%d_%H%M%S.nc``.

.. code-block:: console

  run_torax --config='torax.examples.basic_config' \
  --output_dir=</path/to/output_dir>

``plot_config`` (default: `plotting/configs/default_plot_config.py`) -
Sets the plotting configuration used for the post-simulation plotting options.
This flag should give the path to a Python file containing a `PLOT_CONFIG`
variable which is an instance of `torax.plotting.plotruns_lib.FigureProperties`.
By default, `plotting/configs/default_plot_config.py` is used.
See :ref:`plotting` for further details and examples. An example using a
non-default plot config is shown below.

.. code-block:: console

  run_torax --config='torax.examples.basic_config' \
  --plot_config=plotting/configs/simple_plot_config.py

``log_output`` (default: `False`) - Logs a subset of the initial and final
state of the simulation, including: ion and electron temperature, electron
density, safety factor and magnetic shear. Used for debugging.

.. code-block:: console

  run_torax \
  --config='torax.examples.basic_config' \
  --output_dir=</path/to/output_dir>

Any number of the above flags can be combined.

Post-simulation menu
--------------------

To take advantage of the in-memory (non-persistent) cache, the process does not
end upon simulation termination. Instead, the user is presented with the
following menu.

  | **r**: RUN SIMULATION
  | **mc**: modify the existing config and reload it
  | **cc**: provide a new config file to load
  | **tlp**: toggle --log_progress
  | **tlo**: toggle --log_output
  | **pr**: plot previous run(s) or against reference if provided
  | **q**: quit

* **mc** allows for reloading the existing config file, which can be modified
  in the interim.

* **cc** allows for loading a new config file. The user will be prompted to
  provide a path to a new config file. Optionally the same config file
  previously used can be reloaded, including any changes that the user has
  implemented in the interim.

For both the **mc** and **cc** options, if in the new config file, the only
different config variables compared to the previous run are `dynamic` variables
(see :ref:`dynamic_vs_static`), then the new simulation can be run without
recompilation. `Static` config variables which will trigger recompilation
include variables related to:

  * Grid resolution
  * Evolved variables (equations being solved)
  * Changing internal functions used, e.g. transport model, sources,
    time_step_calculator, pedestal model, solver, etc.

* **r** will launch a new run, include with config changes if **cc** or **mc**
  was chosen previously and changes applied.

* **tlp** toggles the ``--log_progress`` flag for the next run.

* **tlo** toggles the ``--log_output`` flag for the next run, used for debugging
  purposes.

* **pr** provides three options. Plot the last run (0), the last two runs (1),
  the last run against a reference run (2).

* **q** quits the process.

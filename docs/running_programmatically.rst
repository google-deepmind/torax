.. _running_programmatically:

Running simulations programmatically
####################################

This short guide describes how to integrate Torax into your codebase and allows
you to run it multiple times efficiently.

First, we need a ``torax.ToraxConfig`` object representing the simulation config.
In this example, we will use the ``torax/examples/iterhybrid_rampup.py`` config:

.. code-block:: python

  import torax

  torax_config = torax.build_torax_config_from_file('examples/iterhybrid_rampup.py')

If you already have a ``config_dict`` dictionary in Python, you could
instead use ``torax_config = torax.ToraxConfig.from_dict(config_dict)``.

We can then run the simulation:

.. code-block:: python

  # returns the output XArray DataTree and a torax.StateHistory object.
  data_tree, state_history = torax.run_simulation(torax_config)

  # Check that the simulation completed successfully.
  if state_history.sim_error != torax.SimError.NO_ERROR:
    raise ValueError(
        f'TORAX failed to run the simulation with error: {state_history.sim_error}.'
    )

  # Example below shows how to access the fusion gain at time=2 seconds.
  Q_fusion_t2 = data_tree.scalars.Q_fusion.sel(time=2, method='nearest')

Plotting from an in-memory simulation
######################################

If you have already run a simulation and have a ``data_tree`` in memory, you can
plot it directly without saving to a file first using
``torax.plot_run_from_data_tree``:

.. code-block:: python

  plot_config = torax.import_module('plotting/configs/default_plot_config.py')['PLOT_CONFIG']

  # Plot directly from the in-memory data_tree returned by run_simulation.
  fig = torax.plot_run_from_data_tree(plot_config, {"TORAX": data_tree})

To compare multiple in-memory runs, pass a dictionary mapping labels to
DataTrees. The labels appear in the plot legends:

.. code-block:: python

  fig = torax.plot_run_from_data_tree(
      plot_config,
      {
          "TORAX": torax_data_tree,
          "JINTRAC": jintrac_data_tree,
          "EXPERIMENT": experimental_data_tree,
      },
  )

If you have saved the output to a ``.nc`` file and want to plot from disk,
use ``torax.plot_run`` instead:

.. code-block:: python

  fig = torax.plot_run(plot_config, {"TORAX": PATH_TO_LOCAL_NC_FILE})

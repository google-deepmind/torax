.. _running_programmatically:

Running simulations programmatically
####################################

This short guide describes how to integrate Torax into you codebase and allows you
to run it multiple times efficiently.

First, we need a ``torax.ToraxConfig`` object representing the simulation config.
In this example, we will use the ``torax/examples/iterhybrid_rampup.py`` config:

.. code-block:: python

  import torax

  torax_config = torax.build_torax_config_from_file('examples/iterhybrid_rampup.py')

If you already have a ``config_dict`` dictionary in Python, you could
instead use ``torax_config = torax.ToraxConfig.from_dict(config_dict)``.

We can then run the simulation:

.. code-block:: python

  # returns a torax.output.StateHistory object
  results = torax.run_simulation(torax_config)

  # Check that the simulation completed successfully.
  if results.sim_error != torax.SimError.NO_ERROR:
    raise ValueError(
        f'TORAX failed to run the simulation with error: {results.sim_error}.'
    )

  # Example data: access the fusion gain time series
  Q_fusion = results.post_processed_outputs.Q_fusion

  # Optionally save the results as an xarray datatree and use xarray methods
  # to access the data. Example below shows how to access the fusion gain
  # at time=2 seconds.
  data_tree = results.simulation_output_to_xr()
  Q_fusion_t2 = data_tree.post_processed_outputs.Q_fusion.sel(time=2, method='nearest')

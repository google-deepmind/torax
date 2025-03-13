.. _running_programmatically:

Running simulations programmatically
####################################

This guide describes how to integrate Torax into you codebase and allows you
to run it multiple times efficiently.

First we need to load up the config and make sure it is configured correctly.
In this example below, we load the iterhybrid_rampup example from the torax
package itself.

NOTE: This is an optional step, if you are working in a notebook or have already
loaded the config in another way you can skip this step. The config dict
just needs to be a dictionary that matches the description in the
configuration section.

.. code-block:: python

  import torax

  config_module = torax.import_module('torax.examples.iterhybrid_rampup')

  if not hasattr(config_module, 'CONFIG'):
    raise ValueError(
        f'Config module {config_module.__file__} should define a '
        'CONFIG dictionary.'
    )
  torax_config = config_module.CONFIG

The torax config is now a python dictionary and any fields can be optimially
overridden at this point to customize the running.

To run the simulation we need to create a Sim object.

.. code-block:: python

  torax_sim = torax.build_sim_from_config(torax_config)

And then we can run the simulation using

.. code-block:: python

  # returns a torax.output.ToraxSimOutputs object
  results = torax_sim.run()

  # Check that the simulation completed successfully.
  if results.sim_error != torax.SimError.NO_ERROR:
    raise ValueError(
        f'TORAX failed to run the simulation with error: {results.sim_error}.'
    )

  # Access the final fusion gain
  Q_fusion_final = results.sim_history[-1].post_processed_outputs.Q_fusion

  # Optionally save the results as an xarray datatree and use xarray methods
  # to access the data.
  data_tree = output.StateHistory(sim_outputs, sim.source_models).simulation_output_to_xr()
  Q_fusion_t2 = data_tree.post_processed_outputs.Q_fusion.sel(time=2, method='nearest')

.. _running_programmatically:

Running simulations programmatically
####################################

This short guide describes how to integrate Torax into you codebase and allows you
to run it multiple times efficiently. Due to potential breaking changes at head
it is currently recommended to use the v0.3.0 release for running programmatically.

First we need to load up the config and make sure it is configured correctly.
In this example below, we load the iterhybrid_rampup example from the torax
package itself. The key object from a config module is the CONFIG python
dictionary used to construct the pydantic ToraxConfig object.

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
  torax_config_dict = config_module.CONFIG

``torax_config_dict`` is a python dictionary and any fields can be optimially
overridden at this point to customize the running.

To run the simulation we need to construct the ToraxConfig pydantic object.

.. code-block:: python

  torax_config = torax.ToraxConfig.from_dict(torax_config_dict)

And then we can run the simulation using

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

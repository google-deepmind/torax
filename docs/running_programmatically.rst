.. _running_programmatically:

Running simulations programmatically
####################################

This guide describes how to integrate Torax into you codebase and allows you
to run it multiple times efficiently.

First we need to load up the config and make sure it is configured correctly.
NOTE: This is an optional step, if you are working in a notebook or have already
loaded the config in another way you can skip this step. The config dict
just needs to be a dictionary that matches the description in the
configuration section.

```python
from torax import torax

module = torax.import_module(config_path)

if not hasattr(module, 'CONFIG'):
  raise ValueError(
      f'Config module {simulator_config.config_file_path} must define a '
      'CONFIG dictionary.'
  )
torax_config = module.CONFIG
```

The torax config is now a python dictionary and any fields can be optimially
overridden at this point to customize the running.

To run the simulation we need to create a Sim object.

```python
torax_sim = torax.build_sim_from_config(torax_config)
```

And then we can run the simulation using

```python
results = torax_sim.run()

# Check that the simulation completed successfully.
if results.sim_error != torax.SimError.NO_ERROR:
  raise ValueError(
      f'TORAX failed to run the simulation with error: {results.sim_error}.'
  )

Q_fusion = results.sim_history[-1].post_processed_outputs.Q_fusion
```

# TODO add information on how to change the simulation from run
# to run.

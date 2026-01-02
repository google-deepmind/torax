.. _jitted_simulation:

Running a completely JITted simulation
######################################

TORAX contains various helpers for running simulations completely under a
[``jax.jit``](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html) context.
This can be useful for performance as well as making use of other
JAX functionality such as batching, automatic differentation and using
accelerators such as GPU or TPU.

JIT compatible version of run run_loop
======================================

Under the experimental API we provide a ``run_loop_jit`` function that can be
used to run a simulation in a JITted context.

.. code-block:: python

  from torax import experimental as torax_experimental

 step_fn = torax_experimental.make_step_fn(torax_config)
  # The simulation loop will exit after executing at most this many time steps.
  # This is needed to provide a constant size graph for JAX to compile but also
  # means that a simulation could be incomplete if it needs to run for more than
  # the provided max_steps.
  max_steps = 100
  sim_states, post_processed_outputs, final_i = torax_experimental.run_loop_jit(
      step_fn=step_fn,
      max_steps=max_steps,
  )

Simulation overrides
====================

We also provide functionality for overriding runtime parameters for a
simulation. Importantly these helpers are themselves JIT compatible so can be
used as part of a larger JITted function involving a TORAX simulation.

The mechanism for providing overrides is via a ``RuntimeParamsProvider`` object.

We can call ``update_provider_from_mapping`` on this object with a mapping of
dot-separated parameters paths to override values and get a new provider with
the overridden values.

See ``examples/iter_hybrid_rampup_grad_and_vmap.ipynb`` for an example of how
to use this functionality and the docstrings of the methods below for more
details on usage.

.. code-block:: python

  # Replace the `TimeVaryingScalar` `Ip`.
  ip_update = torax_experimental.TimeVaryingScalarReplace(
      value=new_ip_value,
  )
  # Replace the `TimeVaryingArray` profile `T_e`.
  T_e_update = torax_experimental.TimeVaryingArrayReplace(
      cell_value=T_e_cell_value * 3.0,
      rho_norm=old_T_e.grid.cell_centers,
  )
  new_provider = step_fn.runtime_params_provider.update_provider_from_mapping(
      {
          'profile_conditions.Ip': ip_update,
          'profile_conditions.T_e': T_e_update,
          'sources.ei_exchange.Qei_multiplier': 2.0,
      }
  )
  sim_states, post_processed_outputs, final_i = torax_experimental.run_loop_jit(
      step_fn=step_fn,
      max_steps=max_steps,
      runtime_params_overrides=new_provider,
  )





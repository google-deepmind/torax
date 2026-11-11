.. _explicit_geometry:

Geometry overrides in TORAX
###########################

We provide a mechanism for overriding the provided geometry for a TORAX
simulation step. This can be useful when coupling TORAX in an iteration loop
with an equilibrium code for example.

As a light demonstration, if we retrieve the ``L`` and ``LY`` objects from a MEQ
run we can then take a TORAX simulation step with the overridden geometry like
so:

.. code-block:: python

  from torax import experimental as torax_experimental
  from torax.experimental import geometry as geometry_experimental

  step_fn = torax_experimental.make_step_fn(torax_config)
  control_dt = 0.5
  # We can directly create a new geometry object from a config.
  geometry_config = geometry_experimental.MEQConfig.from_dict({
      'n_rho': 25,
      'LY_object': initial_LY,
      'L_object': L,
  })
  sim_state, post_processed_outputs = (
      torax_experimental.get_initial_state_and_post_processed_outputs(
          step_fn=step_fn,
          runtime_params_overrides=step_fn.runtime_params_provider.update_provider_from_mapping(
              {'geometry': geometry_config}
          ),
      )
  )

  while not step_fn.is_done(sim_state):
    # Here we use an equilibrium code that returns a LY object.
    # Similarly we could retrieve a different geometry object that TORAX
    # supports.
    new_LY = get_LY_from_eq_code(sim_state, sim_post_processed_outputs)
    # Update the geometry setting the new LY object for the next time step.
    geometry_config = geometry_experimental.MEQConfig.from_dict({
        'n_rho': 25,
        'L_object': L,
        'LY_object': new_LY,
    })
    sim_state, post_processed_outputs = step_fn.jitted_fixed_time_step(
        dt=control_dt,
        input_state=sim_state,
        previous_post_processed_outputs=post_processed_outputs,
        # New provider passed in as overrides here.
        runtime_params_overrides=step_fn.runtime_params_provider.update_provider_from_mapping(
            {'geometry': geometry_config}
        ),
    )

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
  geometry_config = geometry_experimental.Geometry.from_dict({
      'n_rho': 25,
      'geometry_type': 'fbt',
      'Ip_from_parameters': True,
      'LY_object': initial_LY,
      'L_object': L,
  })
  fbt_geometry_provider = geometry_config.build_provider
  sim_state, post_processed_outputs = (
      torax_experimental.get_initial_state_and_post_processed_outputs(
          geometry_provider=fbt_geometry_provider,
          step_fn=step_fn
      )
  )
  previous_LY = initial_LY

  while not step_fn.is_done(sim_state):
    # Here we use an equilibrium code that returns a LY object.
    # Similarly we could retrieve a different geometry object that TORAX
    # supports.
    new_LY = get_LY_from_eq_code(sim_state, sim_post_processed_outputs)
    # Update the geometry setting the new LY object for the next time step.
    # We need to provide the previous LY object to create a correct phibdot term
    geometry_config = geometry_experimental.Geometry.from_dict({
      'n_rho': 25,
      'geometry_type': 'fbt',
      'Ip_from_parameters': True,
      'geometry_configs': {
            sim_state.t: {
                'LY_object': previous_LY,
                'L_object': L,
            },
            sim_state.t + control_dt: {
                'LY_object': new_LY,
                'L_object': L,
            },
        },
    })
    fbt_geometry_provider = geometry_config.build_provider
    sim_state, post_processed_outputs = step_fn.jitted_fixed_time_step(
        dt=control_dt,
        sim_state=sim_state,
        previous_post_processed_outputs=post_processed_outputs,
        # New provider passed in as overrides here.
        geo_overrides=fbt_geometry_provider,
    )
    previous_LY = new_LY

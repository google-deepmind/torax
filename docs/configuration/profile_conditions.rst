.. _profile_conditions:

profile_conditions
##################

Configures boundary conditions, initial conditions, and (optionally) prescribed
time-dependence of temperature, density, and current.

.. torax_config:: torax._src.core_profiles.profile_conditions.ProfileConditions
   :include-fields: I_p, I_p_source, psi_bc_source, v_loop_lcfs, T_i_right_bc, T_e_right_bc, T_i, T_e, psi, psidot, n_e, normalize_n_e, n_e_bar, n_e_bar_is_f_GW, n_e_right_bc, n_e_right_bc_is_f_GW, n_e_right_bc_mode, n_e_right_bc_reference_rho, n_e_right_bc_multiplier, current_profile_nu, initial_j_is_total_current, psi_source, toroidal_angular_velocity, toroidal_angular_velocity_right_bc, internal_boundary_conditions, fast_ions

internal_boundary_conditions
============================

no_ibc
------

Disables internal boundary conditions (default).

.. torax_config:: torax._src.internal_boundary_conditions.config.NoIBCConfig

prescribed
----------

.. torax_config:: torax._src.internal_boundary_conditions.config.PrescribedIBCConfig
   :include-class-doc:

Whenever any prescribed internal boundary condition is active, all evolved
channels (``T_i``, ``T_e``, and ``n_e``, depending on ``numerics.evolve_*``)
must be specified at the exact same radial locations. Pinning only a subset of
evolved channels or pinning different channels at different radial points raises
a validation error, since gradients in a pinned channel couple into transport
and smoothing of the other channels.

For example, to specify a boundary condition at ``rho = 0.5`` (with all kinetic
channels evolved):

.. code-block:: python

  {
      'internal_boundary_conditions': {
          'model_name': 'prescribed',
          'T_i': {0.0: {0.5: 5.0}},
          'T_e': {0.0: {0.5: 5.0}},
          'n_e': {0.0: {0.5: 1.0e20}},
      }
  }

Multiple radial locations or ranges can be specified in the same dictionary:

.. code-block:: python

  {
      'internal_boundary_conditions': {
          'model_name': 'prescribed',
          'T_i': {0.0: {0.5: 5.0, 0.7: 3.0}},
          'T_e': {0.0: {0.5: 5.0, 0.7: 3.0}},
          'n_e': {0.0: {0.5: 1.0e20, 0.7: 0.8e20}},
      }
  }

beta_poloidal_prime
-------------------

.. torax_config:: torax._src.internal_boundary_conditions.config.BetaPoloidalPrimeIBCConfig
   :include-class-doc:
   :include-fields: model_name, mode, rho_norm_edge, n_e_edge, n_e_edge_multiplier, n_e_is_f_GW, beta_poloidal_prime, beta_poloidal_prime_min, P_SOL_scaling, T_i_T_e_ratio

Example:

.. code-block:: python

  {
      'internal_boundary_conditions': {
          'model_name': 'beta_poloidal_prime',
          'rho_norm_edge': 0.85,
          'mode': 'constant',
          'beta_poloidal_prime': 1.0,
      }
  }

To specify IBCs using multiple models, specify a list:

.. code-block:: python

  {
      'internal_boundary_conditions': [
          {
              'model_name': 'prescribed',
              'T_i': {0.0: {0.5: 5.0}},
          },
          {
              'model_name': 'beta_poloidal_prime',
              'rho_norm_edge': 0.85,
              'mode': 'constant',
              'beta_poloidal_prime': 1.0,
          },
      ]
  }

fast_ions
=========

Each entry in the ``fast_ions`` list prescribes a fast ion species for a
specific source, overriding any model-computed fast ion values for that
``(source, species)`` pair (requires ``numerics.enable_fast_ions`` to be
``True``):

.. torax_config:: torax._src.core_profiles.profile_conditions.PrescribedFastIon

Example:

.. code-block:: python

  'profile_conditions': {
      'fast_ions': [
          {
              'source': 'icrh',
              'species': 'He3',
              'n': {0.0: 1e19},
              'n_right_bc': 0.0,
              'T': {0.0: 100.0},
              'T_right_bc': 0.0,
          },
      ],
  }

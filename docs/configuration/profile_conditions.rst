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

For example, to specify a boundary condition for ``T_i`` at ``rho = 0.5``:

.. code-block:: python

  {
      'internal_boundary_conditions': {
          'model_name': 'prescribed',
          'T_i': {0.0: {0.5: 5.0}},
      }
  }

To specify multiple IBCs across different radial locations, specify a list:

.. code-block:: python

  {
      'internal_boundary_conditions': [
          {
              'model_name': 'prescribed',
              'T_i': {0.0: {0.5: 5.0}},
          },
          {
              'model_name': 'prescribed',
              'T_i': {0.0: {0.7: 3.0}},
          },
      ]
  }

beta_poloidal_prime
-------------------

.. torax_config:: torax._src.internal_boundary_conditions.config.BetaPoloidalPrimeIBCConfig
   :include-class-doc:
   :include-fields: model_name, mode, rho_norm_edge, n_e_edge, n_e_edge_multiplier, n_e_is_f_GW, beta_poloidal_prime, beta_poloidal_prime_min, P_SOL_scaling, Ti_Te_ratio

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

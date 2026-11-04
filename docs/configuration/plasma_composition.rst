.. _plasma_composition_config:

plasma_composition
##################

Defines the distribution of ion species. The keys and their meanings are as
follows:

.. torax_config:: torax._src.core_profiles.plasma_composition.plasma_composition.PlasmaComposition
   :include-fields: main_ion, impurity, Z_eff, Z_i_override, A_i_override

The average charge state of each ion in each mixture is determined by Mavrin
polynomials |mavrin|, which are fitted to atomic data, and in the temperature
ranges of interest in the tokamak core, are well approximated as 1D functions of
electron temperature. All ions with atomic numbers below Carbon are assumed to
be fully ionized.

Plasma Composition Examples
===========================

1. Main Ion Settings
--------------------

In all examples below, the impurity mode is the default ``'fractions'`` mode,
with a single impurity species set for each case.

* Pure deuterium main ions:

  .. code-block:: python

    'plasma_composition': {
        'main_ion': 'D',
        'impurity': {
            'impurity_mode': 'fractions',
            'species': 'Ne',  # Neon
        },
        'Z_eff': 1.5,
    }

* 50-50 DT main ion mixture:

  .. code-block:: python

    'plasma_composition': {
        'main_ion': {'D': 0.5, 'T': 0.5},
        'impurity': {
            'impurity_mode': 'fractions',
            'species': 'Be',  # Beryllium
        },
        'Z_eff': 1.8,
    }

* Time-varying DT main ion mixture:

  .. code-block:: python

    'plasma_composition': {
      'main_ion': {
        'D': {0.0: 0.1, 5.0: 0.9},  # D fraction from 0.1 to 0.9
        'T': {0.0: 0.9, 5.0: 0.1},  # T fraction from 0.9 to 0.1
      },
      'impurity': {
          'impurity_mode': 'fractions',
          'species': 'W',  # Tungsten
      },
      'Z_eff': 1.1,
    }


2. Impurity Fractions Mode (``impurity_mode: 'fractions'``)
-----------------------------------------------------------

This is the default mode. You provide fractional abundances for a set of
impurities, which are then treated as a single effective impurity species.
``Z_eff`` is a required input to constrain the total impurity density.
Attributes in the ``impurity`` dict are as follows:

.. torax_config:: torax._src.core_profiles.plasma_composition.impurity_fractions.ImpurityFractions

Example: Pure deuterium plasma with a Neon impurity, constrained by a time- and
space-varying ``Z_eff``.

.. code-block:: python

  'plasma_composition': {
      'main_ion': 'D',
      'impurity': {
          'impurity_mode': 'fractions',
          'species': 'Ne',
      },
      'Z_eff': {
          0.0: {0.0: 1.2, 1.0: 1.5}, # At t=0, Z_eff profile from 1.2 to 1.5
          5.0: {0.0: 2.0, 1.0: 2.5}, # At t=5, Z_eff profile from 2.0 to 2.5
      },
  }

Example: Pure deuterium plasma with Neon and Argon impurity, consistent with a
flat and constant ``Z_eff``, and time varying fractional abundances.

.. code-block:: python

  'plasma_composition': {
        'main_ion': 'D',
        'impurity': {
            'impurity_mode': 'fractions',
            'species': {
                'Ne': {0.0: {0.0: 0.1}, 5.0: {0.0: 0.9}},
                'Ar': {0.0: {0.0: 0.9}, 5.0: {0.0: 0.1}},
            },
        },
        'Z_eff': 2.0,
    }

3. Impurity n_e Ratios Mode (``impurity_mode: 'n_e_ratios'``)
-------------------------------------------------------------

Specify the density of each impurity species as a ratio of the electron density
(:math:`n_{\text{impurity}} / n_e`). In this mode, ``Z_eff`` is an emergent
quantity calculated by TORAX, and any user-provided ``Z_eff`` will be ignored (a
warning will be issued).

.. torax_config:: torax._src.core_profiles.plasma_composition.electron_density_ratios.ElectronDensityRatios
   :include-fields: impurity_mode, species, Z_override, A_override

Example: A plasma with a time-varying Tungsten concentration and constant Neon.
Tungsten starts hollow, and evolves to a flat profile at t=10s.

.. code-block:: python

  'plasma_composition': {
      'main_ion': 'D',
      'impurity': {
          'impurity_mode': 'n_e_ratios',
          'species': {
              'W': {
                  0.0: {0.0: 1e-5, 1.0: 1e-4},
                  10.0: {0.0: 1e-4},
              },  # n_W/n_e ramps hollow to flat 1e-4
              'Ne': 0.01,  # n_Ne/n_e is constant
          },
      },
      # 'Z_eff': 2.0 # This would be ignored and a warning would be issued.
  }

.. note::

  **Edge Model Boundary Requirement:** If an edge model (such as the
  ``extended_lengyel`` model) is active, any impurity species whose core
  profile is rescaled by the edge model (e.g. seeded impurities in inverse
  mode, or fixed impurities when ``impurity_sot='edge'``) **must have a
  strictly positive value at the LCFS (rho_norm=1.0)** in the input
  configuration. If the profile goes to zero at ``rho_norm=1.0``, the rescaling
  mechanism fails, leaving the core profile at zero and silently breaking the
  edge-core coupling. A configuration validation error will be raised in this
  case. To resolve it, set a small positive value at the LCFS (the actual value
  will be overwritten by the edge model). See :ref:`edge_models` for details.

Too large impurity ratios may lead to negative main ion densities being
calculated, which will lead to TORAX returning an error.

4. Impurity n_e Ratios with Z_eff Constraint (``impurity_mode: 'n_e_ratios_Z_eff'``)
------------------------------------------------------------------------------------

Specify the density of each impurity species as a ratio of the electron
density (:math:`n_{\text{impurity}} / n_e`), with exactly one species designated
as ``None``. TORAX will calculate the density of the ``None`` species to be
self-consistent with the provided ``Z_eff`` profile and the densities of the
other "known" impurities.

.. torax_config:: torax._src.core_profiles.plasma_composition.electron_density_ratios_zeff.ElectronDensityRatiosZeff
   :include-fields: impurity_mode, species, Z_override, A_override

Example: A plasma with a known, constant Neon concentration, where the
Tungsten concentration is unknown but a flat ``Z_eff`` profile ramps up over
time.

.. code-block:: python

  'plasma_composition': {
      'main_ion': 'D',
      'impurity': {
          'impurity_mode': 'n_e_ratios_Z_eff',
          'species': {
              'Ne': 0.01,  # n_Ne/n_e is constant
              'W': None,  # n_W/n_e will be calculated
          },
      },
      'Z_eff': {
          0.0: {0.0: 2.0},
          10.0: {0.0: 2.2},
      },  # Flat Z_eff ramps from 2.0 to 2.2
  }

An error will be raised if the calculated density for the unknown impurity
species becomes negative.

Allowed ion symbols
===================

The following ion symbols are recognized for ``main_ion`` and ``impurity`` input
fields.

*   H  (Hydrogen)
*   D  (Deuterium)
*   T  (Tritium)
*   He3 (Helium-3)
*   He4 (Helium-4)
*   Li (Lithium)
*   Be (Beryllium)
*   C (Carbon)
*   N (Nitrogen)
*   O (Oxygen)
*   Ne (Neon)
*   Ar (Argon)
*   Kr (Krypton)
*   Xe (Xenon)
*   W (Tungsten)

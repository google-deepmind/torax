.. _plasma_composition_config:

plasma_composition
##################

Defines the distribution of ion species.  The keys and their meanings are as
follows:

``main_ion`` (dict[str, **time-varying-scalar**] | str [default =
``{'D': 0.5, 'T': 0.5}``]) Specifies the main ion species.

  *   If a string, it represents a single ion species (e.g., ``'D'`` for
      deuterium, ``'T'`` for tritium, ``'H'`` for hydrogen). See below for the
      full list of supported ions.
  *   If a dict, it represents a mixture of ion species with given fractions. By
      `mixture`, we mean key value pairs of ion symbols and fractional
      concentrations, which must sum to 1 within a tolerance of 1e-6. The
      effective mass and charge of the mixture is the weighted average of the
      species masses and charges. The fractions can be time-dependent, i.e. are
      **time-varying-scalar**. The ion mixture API thus supports features such
      as time varying isotope ratios.

``impurity`` (dict)
  Specifies the impurity species. The way impurities are defined is set by the
  ``impurity_mode`` field within this dictionary. Three modes are supported:
  ``'fractions'``, ``'n_e_ratios'``, and ``'n_e_ratios_Z_eff'``. See the
  "Plasma Composition Examples" section below for details.

``Z_eff`` ( **time-varying-array** [default = 1.0])
  Plasma effective charge, defined as :math:`Z_{eff}=\sum_i n_i Z_i^2 / n_e`.
  This is a required input when using the ``'fractions'`` or
  ``'n_e_ratios_Z_eff'`` impurity modes. When using the ``'n_e_ratios'``
  mode, ``Z_eff`` is an emergent calculated quantity, and any user-provided
  ``Z_eff`` value will be ignored (a warning will be issued).

``Z_i_override`` (**time-varying-scalar** | None [default = None])
  An optional override for the main ion's charge (Z) or average charge of an
  ion mixture. If provided, this value will be used instead of the Z calculated
  from the ``main_ion`` specification.

``A_i_override`` (**time-varying-scalar** | None [default = None])
  An optional override for the main ion's mass (A) in amu units or average mass
  of an ion mixture. If provided, this value will be used instead of the A
  calculated from the ``main_ion`` specification.

The average charge state of each ion in each mixture is determined by
`Mavrin polynomials <https://doi.org/10.1080/10420150.2018.1462361>`_, which are
fitted to atomic data, and in the temperature ranges of interest in the tokamak
core, are well approximated as 1D functions of electron temperature. All ions
with atomic numbers below Carbon are assumed to be fully ionized.

Plasma Composition Examples
===========================

**1. Main Ion settings**

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


**2. Impurity Fractions Mode (`impurity_mode: 'fractions'`)**

This is the default mode. You provide fractional
abundances for a set of impurities, which are then treated as a single
effective impurity species. ``Z_eff`` is a required input to constrain the
total impurity density. Attributes in the ``impurity`` dict are as follows:

*   ``impurity_mode`` (str): Must be ``'fractions'``.
*   ``species`` (dict[str, **time-varying-array**] | str): A single impurity
    string (e.g., ``'Ne'``) or a dict of impurities and their fractional
    abundances within the impurity mix (e.g., ``{'Ne': 0.8, 'Ar': 0.2}``).
    The time-varying-array values must be defined on the same grid and their
    values across all species must sum to 1.
*   ``Z_override`` (**time-varying-scalar** | None): Optional override for the
    impurity's average charge (Z).
*   ``A_override`` (**time-varying-scalar** | None): Optional override for the
    impurity's average mass (A).

Example: Pure deuterium plasma with a Neon impurity, constrained by a time- and
space-varying Z_eff.

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
flat and constant Z_eff, and time varying fractional abundances.

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

**3. Impurity n_e Ratios Mode (`impurity_mode: 'n_e_ratios'`)**

Specify the density of each impurity species as a ratio of the electron density
(n_impurity / n_e). In this mode, ``Z_eff`` is an emergent quantity calculated
by TORAX, and any user-provided ``Z_eff`` will be ignored (a warning will be
issued).

*   ``impurity_mode`` (str): Must be ``'n_e_ratios'``.
*   ``species`` (dict[str, **time-varying-array**]): A dict mapping each
    impurity symbol to its density ratio with n_e. The values must be
    non-negative.
*   ``Z_override`` (**time-varying-scalar** | None): Optional override for the
    impurity's average charge (Z).
*   ``A_override`` (**time-varying-scalar** | None): Optional override for the
    impurity's average mass (A).

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

**4. Impurity n_e Ratios with Z_eff Constraint (`impurity_mode: 'n_e_ratios_Z_eff'`)**

Specify the density of each impurity species as a ratio of the electron
density (`n_impurity / n_e`), with exactly one species designated as `None`.
TORAX will calculate the density of the `None` species to be self-consistent
with the provided `Z_eff` profile and the densities of the other "known"
impurities.

*   ``impurity_mode`` (str): Must be ``'n_e_ratios_Z_eff'``.
*   ``species`` (dict[str, **time-varying-array** | None]): A dict mapping each
    impurity symbol to its density ratio with n_e. Exactly one value must be
    `None`. All other values must be non-negative.
*   ``Z_override`` (**time-varying-scalar** | None): Optional override for the
    impurity's average charge (Z).
*   ``A_override`` (**time-varying-scalar** | None): Optional override for the
    impurity's average mass (A).

Example: A plasma with a known, constant Neon concentration, where the
Tungsten concentration is unknown but a flat `Z_eff` profile ramps up over time.

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

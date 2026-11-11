.. _sources_config:

sources
#######

Dictionary with nested dictionaries containing the configurable runtime
parameters of all TORAX heat, particle, and current sources. See
:ref:`physics_models` for details on the source physics models.

Any source not explicitly included in ``sources`` is disabled (set to ``None``,
or ``{'mode': 'zero'}`` for ``ei_exchange``). To enable a source with its
default model and parameters, pass an empty dictionary. For example, to enable
``ei_exchange`` with default options as the only active source:

.. code-block:: python

    'sources': {
        'ei_exchange': {},
    }

The following runtime parameters are common to all source configurations:

.. torax_config:: torax._src.sources.base.SourceConfigBase

When ``mode`` is ``'prescribed'``, ``prescribed_values`` accepts a dictionary
with the following channel keys (only channels supported by the source may be
set):

.. torax_config:: torax._src.sources.base.PrescribedValues

For sources that produce fast ions (``fusion``, ``icrh``, and ``nbi``), each
species entry in ``prescribed_values['fast_ions']`` (keyed by ``'H'``, ``'D'``,
``'T'``, ``'He3'``, or ``'He4'``) accepts:

.. torax_config:: torax._src.sources.base.FastIonPrescribedValues

For example, to disable ``fusion`` explicitly for sensitivity testing:

.. code-block:: python

    'sources': {
        'fusion': {'mode': 'zero'},
    }

To prescribe ``generic_current`` from a tuple of NumPy arrays:

.. code-block:: python

    'sources': {
        'generic_current': {
            'mode': 'prescribed',
            'prescribed_values': {
                'j_parallel': (times, rhon, current_profiles),
            },
        },
    }

where ``times`` is a 1D NumPy array of times, ``rhon`` is a 1D NumPy array of
normalized toroidal flux coordinates, and ``current_profiles`` is a 2D NumPy
array of the current profile at each time.

The available source keys in ``sources`` are:

.. torax_config:: torax._src.sources.config.SourcesConfig

ei_exchange
===========

Collisional ion-electron heat exchange.

.. torax_config:: torax._src.sources.qei_source.QeiSourceConfig
   :own-fields-only:

bremsstrahlung
==============

Bremsstrahlung model from Wesson, with an optional correction for relativistic
effects from Stott (PPCF 2005).

.. torax_config:: torax._src.sources.bremsstrahlung_heat_sink.BremsstrahlungHeatSinkConfig
   :own-fields-only:

cyclotron_radiation
===================

Cyclotron radiation model from |albajar2001| with a deposition profile from
|artaud2018|. Because the Albajar profile parameterization is only defined
for peaked or flat profiles (:math:`\alpha \ge 0`), hollow or inverted density
and temperature profiles are clamped to the flat-profile limit
(:math:`\alpha = 0`) in the fit.

.. torax_config:: torax._src.sources.cyclotron_radiation_heat_sink.CyclotronRadiationHeatSinkConfig
   :own-fields-only:

ecrh
====

Electron-cyclotron heating and current drive, based on the local efficiency
model in |lin-liu|. Given an EC power density profile and efficiency profile,
the model produces the corresponding EC-driven current density profile. Three
configuration patterns are supported:

1. Provide an entire EC power density profile manually (via
   ``extra_prescribed_power_density``).
2. Provide the parameters of a Gaussian EC deposition (via ``gaussian_width``,
   ``gaussian_location``, and ``P_total``).
3. Any combination of the above (the manual and Gaussian profiles are summed to
   produce the final EC deposition profile; both default to zero).

.. torax_config:: torax._src.sources.electron_cyclotron_source.ElectronCyclotronSourceConfig
   :own-fields-only:

fusion
======

DT fusion power from the Bosch-Hale parameterization. Uses the D and T fractions
from the ``main_ion`` ion mixture.

.. torax_config:: torax._src.sources.fusion_heat_source.FusionHeatSourceConfig
   :own-fields-only:

gas_puff
========

Gas puff particle source for the electron density equation deposited radially
using an exponential profile decaying inwards from the outer edge boundary
(:math:`\hat{\rho} = 1.0`).

.. torax_config:: torax._src.sources.gas_puff_source.GasPuffSourceConfig
   :own-fields-only:

generic_current
===============

Generic external current profile, parameterized as a Gaussian.

.. torax_config:: torax._src.sources.generic_current_source.GenericCurrentSourceConfig
   :own-fields-only:

generic_heat
============

Time-dependent Gaussian ion and electron heat source.

.. torax_config:: torax._src.sources.generic_ion_el_heat_source.GenericIonElHeatSourceConfig
   :own-fields-only:

generic_particle
================

Generic particle source for the electron density equation.

.. torax_config:: torax._src.sources.generic_particle_source.GenericParticleSourceConfig
   :include-fields: model_name

gaussian
--------

Time-dependent Gaussian particle source with prescribed injection rate
``S_total``.

.. torax_config:: torax._src.sources.generic_particle_source.GenericParticleSourceConfig
   :own-fields-only:
   :exclude-fields: model_name

feedback
--------

.. torax_config:: torax._src.sources.particle_feedback_source.ParticleFeedbackSourceConfig
   :include-class-doc:
   :own-fields-only:
   :exclude-fields: model_name

**Example configuration:**

.. code-block:: python

  config['sources']['generic_particle'] = {
      'model_name': 'feedback',
      'average_type': 'line',
      'proportional_gain': 10.0,
      'integral_gain': 1.0,
      'derivative_gain': 0.5,
      'derivative_smoothing_time_scale': 0.01,
      'min_integral': -1.0e22,
      'max_integral': 1.0e22,
      'target_average_n_e': 1.0e20,
      'S_feedforward': 1.0e22,
      'profile': {
          'profile_type': 'exponential',
          'width': 0.05,
      },
  }

profile
^^^^^^^

.. torax_config:: torax._src.sources.particle_feedback_source.ExponentialProfileConfig
   :include-fields: profile_type

exponential
~~~~~~~~~~~

.. torax_config:: torax._src.sources.particle_feedback_source.ExponentialProfileConfig
   :include-class-doc:
   :own-fields-only:
   :exclude-fields: profile_type

gaussian
~~~~~~~~

.. torax_config:: torax._src.sources.particle_feedback_source.GaussianProfileConfig
   :include-class-doc:
   :own-fields-only:
   :exclude-fields: profile_type

prescribed
~~~~~~~~~~

.. torax_config:: torax._src.sources.particle_feedback_source.PrescribedProfileConfig
   :include-class-doc:
   :own-fields-only:
   :exclude-fields: profile_type

icrh
====

Ion cyclotron resonance heating (ICRH) source. Custom ICRH models can also be
registered via ``torax.sources.register_source_config`` (see
:ref:`model-integration`).

.. torax_config:: torax._src.sources.ion_cyclotron_source.scaled_profile.ScaledProfileIonCyclotronSourceConfig
   :include-class-doc:
   :exclude-fields: mode, is_explicit, prescribed_values

**Example:**

.. code-block:: python

    'sources': {
        'icrh': {
            'model_name': 'scaled_profile',
            'P_total': 25e6,
            'absorption_fraction': 0.9,
            'minority_species': 'He3',
            'heat_profile_ion': {0: {0.0: 0, 0.3: 1.0, 0.6: 0.2, 1.0: 0}},
            'heat_profile_electron': {0: {0.0: 0, 0.35: 0.8, 0.65: 0.1, 1.0: 0}},
            'reference_B0': 12.2,
        },
    }

See :ref:`physics_models` for more detail.

impurity_radiation
==================

Impurity radiation heat sink for the electron heat equation.

.. torax_config:: torax._src.sources.impurity_radiation_heat_sink.impurity_radiation_mavrin_fit.ImpurityRadiationHeatSinkMavrinFitConfig
   :include-fields: model_name

mavrin_fit
----------

Polynomial fits to ADAS data from |mavrin|.

.. torax_config:: torax._src.sources.impurity_radiation_heat_sink.impurity_radiation_mavrin_fit.ImpurityRadiationHeatSinkMavrinFitConfig
   :own-fields-only:
   :exclude-fields: model_name

P_in_scaled_flat_profile
------------------------

Sets impurity radiation to be a constant fraction of the total external input
heating power.

.. torax_config:: torax._src.sources.impurity_radiation_heat_sink.impurity_radiation_constant_fraction.ImpurityRadiationHeatSinkConstantFractionConfig
   :own-fields-only:
   :exclude-fields: model_name

nbi
===

Neutral beam injection (NBI) heating, current drive, and particle source. No
first-principle-based model is yet implemented in TORAX; NBI profiles are
provided via ``mode = 'prescribed'`` (the default), supporting
``power_density_i``, ``power_density_e``, ``j_parallel``,
``particle_source_e``, and ``fast_ions`` in ``prescribed_values``.

.. torax_config:: torax._src.sources.neutral_beam_injection_source.NeutralBeamInjectionSourceConfig
   :own-fields-only:

ohmic
=====

Ohmic heating power computed from plasma current and neoclassical conductivity.

.. torax_config:: torax._src.sources.ohmic_heat_source.OhmicHeatSourceConfig
   :own-fields-only:

pellet
======

Time-dependent Gaussian pellet fueling source for the electron density equation.

.. torax_config:: torax._src.sources.pellet_source.PelletSourceConfig
   :own-fields-only:

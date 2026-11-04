.. include:: ../links.rst

.. _sources_config:

sources
#######

Dictionary with nested dictionaries containing the configurable runtime
parameters of all TORAX heat, particle, and current sources. The following
runtime parameters are common to all sources, with defaults depending on the
specific source. See :ref:`physics_models` For details on the source physics
models.

Any source which is not explicitly included in the sources dict, is set to zero.
To include a source with default options, the source dict should contain an
empty dict. For example, for setting ``ei_exchange``, with default options, as
the only active source in ``sources``, set:

.. code-block:: python

    'sources': {
        'ei_exchange': {},
    }

The configurable runtime parameters of each source are as follows:

``prescribed_values`` (**dict** [default = {}])
  Prescribed profile values for the source, keyed by output channel name
  (``power_density_i``, ``power_density_e``, ``j_parallel``,
  ``particle_source_e``), plus an optional ``fast_ions`` dictionary for sources
  that produce fast ions. Used if ``mode`` is ``'prescribed'``.

``mode`` (str [default = 'zero'])
  Defines how the source values are computed. Currently the options are:

* ``'zero'``
    Source is set to zero.

* ``'model_based'``
    Source values come from a model in code. Specific model selection where more
    than one model is available can be done by specifying a ``model_name``.
    This is documented in the individual source sections.

* ``'prescribed'``
    Source values are arbitrarily prescribed by the user. The value is set by
    ``prescribed_values`` as a dictionary mapping channel names
    (``power_density_i``, ``power_density_e``, ``j_parallel``,
    ``particle_source_e``) to :ref:`Time-varying arrays`. Note that each channel
    is evaluated independently at each time step.
    Only channels matching the source's ``POPULATED_SOURCE_OUTPUTS`` should be
    specified. Sources that produce fast ions (``fusion``, ``icrh``, ``nbi``)
    may also include ``'fast_ions'`` mapping fast-ion species symbols (``'H'``,
    ``'D'``, ``'T'``, ``'He3'``, ``'He4'``) to dictionaries with non-negative
    :ref:`Time-varying arrays` ``'n'`` [:math:`m^{-3}`] and ``'T'`` [keV].

For example, to set 'fusion_power' to zero, e.g. for testing or sensitivity
purposes, set:

.. code-block:: python

    'sources': {
        'fusion': {'mode': 'zero'},
    }

To set 'generic_current' to a prescribed value based on a tuple of numpy arrays,
e.g. as defined or loaded from a file in the preamble to the CONFIG dict within
config module, set:

.. code-block:: python

    'sources': {
        'generic_current': {
            'mode': 'prescribed',
            'prescribed_values': {
                'j_parallel': (times, rhon, current_profiles),
            },
        },

where the example ``times`` is a 1D numpy array of times, ``rhon`` is a 1D numpy
array of normalized toroidal flux coordinates, and ``current_profiles`` is a 2D
numpy array of the current profile at each time. These names are arbitrary, and
can be set to anything convenient.


``is_explicit`` (bool [default = False])
  Defines whether the source is to be considered explicit or implicit. Explicit
  sources are calculated based on the simulation state at the beginning of a
  time step, or do not have any dependence on state. Implicit sources depend on
  updated states as the iterative solvers evolve the state through the course of
  a time step. If a source model is complex but evolves over slow timescales
  compared to the state, it may be beneficial to set it as explicit.

ei_exchange
===========

Ion-electron heat exchange.

``mode`` (str [default = 'model'])

``Qei_multiplier`` (float [default = 1.0])
  Multiplication factor for ion-electron heat exchange term for testing
  purposes.

bremsstrahlung
==============

Bremsstrahlung model from Wesson, with an optional correction for relativistic
effects from Stott PPCF 2005.

``mode`` (str [default = 'model'])

``use_relativistic_correction`` (bool [default = False])

``exclude_impurity_bremsstrahlung`` (bool [default = False])
  If ``True``, calculates bremsstrahlung using main-ion contributions only.
  As the Mavrin model computes both line and bremsstrahlung radiation
  contributions from impurities, this is automatically set to ``True`` when
  Mavrin impurity radiation is active to prevent double-counting.

cyclotron_radiation
===================

Cyclotron radiation model from |albajar2001| with a deposition profile from
|artaud2018|. Because the Albajar profile parameterization is only defined
for peaked or flat profiles (:math:`\alpha \ge 0`), hollow or inverted density
and temperature profiles are clamped to the flat-profile limit
(:math:`\alpha = 0`) in the fit.

``mode`` (str [default = 'model'])

``wall_reflection_coeff`` (float [default = 0.9])
  Machine-dependent dimensionless parameter corresponding to the fraction of
  cyclotron radiation reflected off the wall and reabsorbed by the plasma.

``beta_min`` (float [default = 0.5])

``beta_max`` (float [default = 8.0])

``beta_grid_size`` (int [default = 32])
  beta in this context is a variable in the temperature profile parameterization
  used in the Albajar model. The parameter is fit with simple grid search
  performed over the range ``[beta_min, beta_max]``, with ``beta_grid_size``
  uniformly spaced steps. This parameter must be positive.

ecrh
====
Electron-cyclotron heating and current drive, based on the local efficiency
model in `Lin-Liu et al., 2003 <https://doi.org/10.1063/1.1610472>`_. Given an
EC power density profile and efficiency profile, the model produces the
corresponding EC-driven current density profile. The user has three options:

1. Provide an entire EC power density profile manually (via
   ``extra_prescribed_power_density``).
2. Provide the parameters of a Gaussian EC deposition (via ``gaussian_width``,
   ``gaussian_location``, and ``P_total``).
3. Any combination of the above.

At least one of ``P_total`` or ``extra_prescribed_power_density`` must be
provided when ``mode`` is ``'model_based'``. Any omitted component contributes
zero, and the manual and Gaussian profiles are summed together to produce the
final EC deposition profile.

    ``mode`` (str [default = 'model'])

    ``extra_prescribed_power_density`` (**time-varying-array** [default = None])
        EC power density deposition profile, in units of :math:`W/m^3`.

    ``gaussian_width`` (**time-varying-scalar** [default = None])
        Width of Gaussian EC power density deposition profile. Required when
        ``mode == 'model_based'`` and ``P_total`` is provided.

    ``gaussian_location`` (**time-varying-scalar** [default = None])
        Location of Gaussian EC power density deposition profile on the
        normalized rho grid. Required when ``mode == 'model_based'`` and
        ``P_total`` is provided.

    ``P_total`` (**time-varying-scalar** [default = None])
        Integral of the Gaussian EC power density profile, setting the total
        power.

    ``current_drive_efficiency`` (**time-varying-array** [default = {0: {0: 0.2, 1: 0.2}}])
        Dimensionless local efficiency profile for conversion of EC power to
        current.

fusion
======

DT fusion power from the Bosch-Hale parameterization. Uses the D and T fractions
from the ``main_ion`` ion mixture.

``mode`` (str [default = 'model'])

gas_puff
========

Gas puff particle source for the electron density equation deposited radially
using an exponential profile decaying inwards from the outer edge boundary
(:math:`\hat{\rho} = 1.0`).

``mode`` (str [default = 'model'])

``puff_decay_length`` (**time-varying-scalar** [default = None])
  Gas puff decay length from edge in units of :math:`\hat{\rho}`. Required
  when ``mode == 'model_based'``.

``S_total`` (**time-varying-scalar** [default = None])
  Total number of particle source in units of particles/s. Required when
  ``mode == 'model_based'``.

generic_current
===============

Generic external current profile, parameterized as a Gaussian.

``mode`` (str [default = 'model'])

``gaussian_location`` (**time-varying-scalar** [default = None])
  Gaussian center of current profile in units of :math:`\hat{\rho}`. Required
  when ``mode == 'model_based'``.

``gaussian_width`` (**time-varying-scalar** [default = None])
  Gaussian width of current profile in units of :math:`\hat{\rho}`. Required
  when ``mode == 'model_based'``.

``I_generic`` (**time-varying-scalar** [default = None])
  Total current in A. Required when ``mode == 'model_based'`` and
  ``use_absolute_current == True``; ignored otherwise.

``fraction_of_total_current`` (**time-varying-scalar** [default = None])
  Sets total ``generic_current`` to be a fraction ``fraction_of_total_current``
  of the total plasma current. Required when ``mode == 'model_based'`` and
  ``use_absolute_current == False``; ignored otherwise.

``use_absolute_current`` (bool [default = False])
  Toggles relative vs absolute external current setting.

generic_heat
============

A utility source module that allows for a time-dependent Gaussian ion and
electron heat source.

``mode`` (str [default = 'model'])

``gaussian_location`` (**time-varying-scalar** [default = None])
  Gaussian center of source profile in units of :math:`\hat{\rho}`. Required
  when ``mode == 'model_based'``.

``gaussian_width`` (**time-varying-scalar** [default = None])
  Gaussian width of source profile in units of :math:`\hat{\rho}`. Required
  when ``mode == 'model_based'``.

``P_total`` (**time-varying-scalar** [default = None])
  Total source power in W. Required when ``mode == 'model_based'``.

``electron_heat_fraction`` (**time-varying-scalar** [default = 0.66666])
  Electron heating fraction.

``absorption_fraction`` (**time-varying-scalar** [default = 0.0])
  Fraction of input power that is absorbed by the plasma.

generic_particle
================

Generic particle source for the electron density equation. Two models are
supported via ``model_name``:

1. ``'gaussian'`` (default): A time-dependent Gaussian particle source with
   prescribed injection rate ``S_total``.
2. ``'feedback'``: Computes the injection rate using a PID feedback control
   loop added to a baseline feedforward rate:
   :math:`S_{\text{total}} = \max(0, S_{\text{feedforward}} + S_p + S_i + S_d)`,
   with tracking error :math:`e(t) = \text{target\_average\_n\_e}(t) - \text{current\_average\_n\_e}(t)`,
   :math:`S_p = \text{proportional\_gain} \cdot e(t)`,
   :math:`S_i = \text{clip}(S_{i,\text{prev}} + \text{integral\_gain} \cdot e(t) \cdot \Delta t, \text{min\_integral}, \text{max\_integral})`,
   and :math:`S_d = \text{derivative\_gain} \cdot (de/dt)_{\text{smoothed}}`.
   The total injection rate is deposited radially using a configurable
   ``profile`` shape (``'exponential'``, ``'gaussian'``, or ``'prescribed'``).

**Common Parameters:**

``mode`` (str [default = 'model'])

**Parameters for** ``model_name = 'gaussian'`` **:**

``deposition_location`` (**time-varying-scalar** [default = None])
  Gaussian center of source profile in units of :math:`\hat{\rho}`. Required
  when ``mode == 'model_based'``.

``particle_width`` (**time-varying-scalar** [default = None])
  Gaussian width of source profile in units of :math:`\hat{\rho}`. Required
  when ``mode == 'model_based'``.

``S_total`` (**time-varying-scalar** [default = None])
  Total particle source in units of particles/s. Required when
  ``mode == 'model_based'``.

**Parameters for** ``model_name = 'feedback'`` **:**

``profile`` (dict [default = ``{'profile_type': 'exponential', 'width': 0.05}``])
  Radial deposition shape configuration, discriminated by ``profile_type``:

  * ``'exponential'``: ``width`` (**time-varying-scalar** [default = 0.05]),
    exponential decay length from the edge inwards in units of
    :math:`\hat{\rho}`.
  * ``'gaussian'``: ``center`` (**time-varying-scalar** [default = 0.0]) and
    ``width`` (**time-varying-scalar** [default = 0.25]) in units of
    :math:`\hat{\rho}`.
  * ``'prescribed'``: ``values`` (**time-varying-array**
    [default = ``{0.0: {0.0: 1.0, 1.0: 1.0}}``]), a non-negative radial shape
    rescaled so its volume integral equals :math:`S_{\text{total}}`.

``S_feedforward`` (**time-varying-scalar** [default = 1e22])
  Baseline feedforward particle injection rate [particles/s].

``average_type`` (str [default = 'line'])
  Method for spatial electron density averaging (``'line'`` or ``'volume'``)
  used to calculate the feedback error.

``proportional_gain`` (**time-varying-scalar** [default = 1.0])
  Proportional feedback gain [ :math:`m^3/s` ] mapping density error to an
  incremental particle fueling rate.

``integral_gain`` (**time-varying-scalar** [default = 0.0])
  Integral feedback gain [ :math:`m^3/s^2` ] mapping time-integrated density
  error to an incremental particle fueling rate.

``derivative_gain`` (**time-varying-scalar** [default = 0.0])
  Derivative feedback gain [ :math:`m^3` ] mapping the time derivative of
  density error to an incremental particle fueling rate.

``derivative_smoothing_time_scale`` (**time-varying-scalar** [default = 0.0])
  Non-negative exponential smoothing time constant [ :math:`s` ] for the error
  time derivative :math:`de/dt`. ``0.0`` uses the raw finite difference.

``min_integral`` (**time-varying-scalar** | None [default = None])
  Optional lower bound [particles/s] on the integral term :math:`S_i` for
  anti-windup clamping.

``max_integral`` (**time-varying-scalar** | None [default = None])
  Optional upper bound [particles/s] on the integral term :math:`S_i` for
  anti-windup clamping.

``target_average_n_e`` (**time-varying-scalar** [default = 0.77e20])
  Desired target average electron density [ :math:`m^{-3}` ] for feedback
  regulation.

**Example Configuration:**

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

icrh
====
Ion cyclotron resonance heating (ICRH) source. Custom ICRH models can also be
registered via ``torax.sources.register_source_config`` (see
:ref:`model-integration`).

``mode`` (str [default = 'model'])

``P_total`` (**time-varying-scalar** [default = None])
  Total injected source power in W. Required when ``mode == 'model_based'``.

``absorption_fraction`` (**time-varying-scalar** [default = 1.0])
  Fraction of ``P_total`` absorbed by the plasma.

``minority_species`` (str [default = 'He3'])
  Symbol of the minority species (e.g. 'He3'). The minority concentration is
  automatically extracted from the plasma composition.

``model_name`` (str [default = 'scaled_profile'])
  Selects the ICRH model. Available options:

* ``'scaled_profile'`` (default)
    Prescribed-profile model with magnetic-field-dependent resonance shifting.
    Adapts reference heating profiles to different magnetic field operating points
    and normalises them to the target power. See :ref:`physics_models` for
    detailed physics description.

    ``heat_profile_ion`` (**time-varying-array** [default = {0: {0: 0, 1: 0}}])
      Reference ion heating power density shape [W/m³], provided on the
      normalised radius grid. The shape is rescaled during normalisation; only
      the relative profile matters.

    ``heat_profile_electron`` (**time-varying-array** [default = {0: {0: 0, 1: 0}}])
      Reference electron heating power density shape [W/m³], provided on the
      normalised radius grid.

    ``reference_B0`` (**time-varying-scalar** [default = 12.2])
      Vacuum toroidal magnetic field at which the reference profiles were
      computed [T].

    Example:

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

Various models for impurity radiation. Runtime params for each available model
are listed separately

``mode`` (str [default = 'model'])

``model_name`` (str [default = 'mavrin_fit'])

The following models are available:

* ``'mavrin_fit'``
    Polynomial fits to ADAS data from
    `Mavrin, 2018. <https://doi.org/10.1080/10420150.2018.1462361>`_

    ``radiation_multiplier`` (float [default = 1.0]). Multiplication factor for
    radiation term for testing sensitivities.

* ``'P_in_scaled_flat_profile'``
    Sets impurity radiation to be a constant fraction of the total external
    input power.

    ``fraction_P_heating`` (**time-varying-scalar** [default = None]). Fraction
    of total external input heating power to use for impurity radiation.
    Required when ``mode == 'model_based'``.

nbi
===

Neutral beam injection (NBI) heating, current drive, and particle source. No
first-principle-based model is yet implemented in TORAX; NBI profiles are
provided via ``mode = 'prescribed'`` (the default).

``mode`` (str [default = 'prescribed'])

``model_name`` (str [default = 'prescribed'])

``prescribed_values`` (**dict** [default = {}])
  Prescribed profile values for the NBI source:

  * ``power_density_i`` (**time-varying-array**): Ion heating power density
    [:math:`W/m^3`].
  * ``power_density_e`` (**time-varying-array**): Electron heating power density
    [:math:`W/m^3`].
  * ``j_parallel`` (**time-varying-array**): Parallel current density
    :math:`\langle \mathbf{j} \cdot \mathbf{B} \rangle / B_0` [:math:`A/m^2`].
  * ``particle_source_e`` (**time-varying-array**): Electron particle source
    density [:math:`m^{-3} s^{-1}`].
  * ``fast_ions`` (optional, dict[str, dict]): Mapping from fast-ion species
    symbol (``'H'``, ``'D'``, ``'T'``, ``'He3'``, or ``'He4'``) to a dictionary
    containing non-negative :ref:`Time-varying arrays` ``'n'`` [:math:`m^{-3}`]
    and ``'T'`` [:math:`keV`].

ohmic
=====

Ohmic power.

``mode`` (str [default = 'model'])

pellet
======

Time-dependent Gaussian pellet source. No first-principle-based model is yet
implemented in TORAX.

``mode`` (str [default = 'model'])

``pellet_deposition_location`` (**time-varying-scalar** [default = None])
  Gaussian center of source profile in units of :math:`\hat{\rho}`. Required
  when ``mode == 'model_based'``.

``pellet_width`` (**time-varying-scalar** [default = None])
  Gaussian width of source profile in units of :math:`\hat{\rho}`. Required
  when ``mode == 'model_based'``.

``S_total`` (**time-varying-scalar** [default = None])
  Total particle source in units of particles/s. Required when
  ``mode == 'model_based'``.

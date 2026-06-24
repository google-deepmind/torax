.. include:: links.rst

.. _hpi2nn_pellet_source:

HPI2-NN pellet source
#####################

`HPI2-NN <https://github.com/DIFFER-NL/hpi2nn>`_ is a machine-learning surrogate
of the HPI2 pellet ablation and deposition code, developed to accelerate
integrated modelling of pellet-fuelled tokamak discharges.

.. note::

   The ``hpi2nn`` package (the surrogate model and its weights) must be
   installed to use this source. Install it as an editable package with
   ``pip install -e <path to the hpi2nn repo>``.

The particle source ``hpi2nn_pellet_source``
(``torax/_src/sources/hpi2nn_pellet_source.py``) calls HPI2-NN to obtain the
particle deposition profile of a pellet with the characteristics and at the
time chosen by the user (trigger time), using the ``T_e``, ``T_i``, ``n_e`` and
``q`` profiles from TORAX:

.. code-block:: python

    dne, dTe, t_abl = evaluate_hpi2nn_model(
        radius, velocity, rho_norm, Te_eV, ne, Ti_eV, q_cell, B_0,
        injection_point_1, injection_point_2, injection_line,
    )

The magnetic field ``B_0`` is passed as an input but is no longer used inside
HPI2-NN in the latest version for WEST.

``injection_point_1`` and ``injection_point_2`` can be used by HPI2-NN to find
the closest matching injection line, but this feature is currently not exposed
through the HPI2-NN pellet source (the injection line is selected directly via
``injection_line``).

See ``torax/_src/sources/hpi2nn_pellet_source.py`` for the full list of source
attributes (pellet radius/velocity, per-trigger ``pellet_radii`` /
``pellet_velocities``, ``trigger_times`` or ``frequency``, ``injection_line``,
``ablation_time``, ``use_hpi2nn_ablation_time``).

The source is explicit (is_explicit = True), so HPI2-NN is called once at each trigger time
and the deposit is held fixed during the implicit solve, instead of being re-evaluated
at every solver iteration.

Ablation time and source normalisation
=======================================

The pellet source is active during the *ablation time*, which represents the
time for the pellet to be fully ablated. Over this window the source is assumed
constant, with the total deposited density spread over the ablation time:

.. math::

    S_\mathrm{pellet} = \frac{\mathrm{d}n_e}{t_\mathrm{ablation}}

The ablation time is, by default, the value ``t_abl`` predicted by HPI2-NN
(``use_hpi2nn_ablation_time = True``). Setting ``use_hpi2nn_ablation_time =
False`` falls back to the user-provided ``ablation_time`` from the
configuration.

The pellet-aware time step calculator
(``torax/_src/time_step_calculator/pellet_aware_time_step_calculator.py``) is
required to ensure that the trigger times and the ablation window are resolved
exactly. For particle conservation it sets the time step over the ablation
window equal to the ablation time used for the source normalisation.

Advice and known issues
=======================

- Our tests showed that QLKNN produced better results than TGLF-NN during the
  post-pellet relaxation phase.
- Using a sawtooth model can sometimes create a temperature collapse during the
  relaxation phase. A pellet makes the pressure profile non-monotone, and a
  sawtooth crash on such a profile can drive a low-temperature collapse.
  If this happens, you can either slightly adjust the configuration parameters,
  or use the ``pellet_aware_simple`` sawtooth trigger model
  (``torax/_src/mhd/sawtooth/pellet_aware_simple_trigger.py``), which
  suppresses sawtooth crashes for a configurable ``pellet_lockout`` window after
  each pellet. ``pellet_lockout`` can be a scalar (applied to every pellet) or a
  per-trigger array aligned with ``trigger_times``.

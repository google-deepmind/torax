.. include:: ../links.rst

.. _neoclassical_config:

neoclassical
############

Configures the neoclassical physics model.

``model_name`` (str [default = ``'analytical'``])
  Selects the top-level neoclassical model. The built-in ``'analytical'`` model
  combines four modular analytical sub-models configured under
  ``bootstrap_current``, ``conductivity``, ``transport``, and
  ``poloidal_velocity``. Custom top-level neoclassical models can also be
  registered via ``torax.neoclassical.register_neoclassical_model`` (see
  :ref:`model-integration`).

When ``model_name = 'analytical'`` (the default), the following four sub-model
sections are available:

bootstrap_current
=================
``model_name`` (str [default = ``'sauter'``])
  The name of the bootstrap current model to use. If the ``bootstrap_current``
  section is omitted, ``'zeros'`` is used. If the section is present without a
  ``model_name``, ``'sauter'`` is used. Options are:

  * ``'sauter'``: Analytical model from |sauter99|.
  * ``'redl'``: Analytical model from |redl2021|.
  * ``'zeros'``: Sets bootstrap current to zero.

If the ``'sauter'`` or ``'redl'`` model is used, the following parameter can be
set:

``bootstrap_multiplier`` (float [default = 1.0])
  Multiplier for the bootstrap current.

conductivity
============
``model_name`` (str [default = ``'sauter'``])
  The name of the neoclassical parallel conductivity model to use. If the
  ``conductivity`` section or ``model_name`` is omitted, ``'sauter'`` is used.
  Options are:

  * ``'sauter'`` (default): Analytical conductivity model from |sauter99|.
  * ``'redl'``: Analytical conductivity model from |redl2021|.

``conductivity_multiplier`` (float > 0 [default = 1.0])
  Multiplier for the neoclassical parallel conductivity. This scales the
  physical conductivity used throughout the simulation, so it affects the
  current diffusion timescale as well as the loop voltage and Ohmic heating.
  Useful for sensitivity studies.

``neoclassical_correction`` (bool [default = True])
  If ``True``, the Spitzer conductivity is multiplied by the neoclassical
  trapped-particle correction of the selected model. If ``False``, the pure
  Spitzer conductivity is used. ``conductivity_multiplier`` is applied in both
  cases.

transport
=========
``model_name`` (str [default = ``'angioni_sauter'``])
  The name of the neoclassical transport model. If the ``transport`` section is
  omitted, ``'zeros'`` is used. If the section is present without a
  ``model_name``, ``'angioni_sauter'`` is used. The following models are
  supported:

  * ``'zeros'``
    Sets all neoclassical transport coefficients to zero.

  * ``'angioni_sauter'``
    The Angioni-Sauter neoclassical transport model from |angioni2000|, with an
    optional near-axis ion thermal conductivity correction from |shaing1997|.
    Supports the following additional parameters:

    - ``use_shaing_ion_correction`` (bool [default = False]): If ``True``,
      smoothly blends the near-axis Shaing ion thermal conductivity into the
      Angioni-Sauter ion thermal conductivity.
    - ``shaing_ion_multiplier`` (float [default = 1.8]): Multiplier applied to
      the Shaing ion thermal conductivity term.
    - ``shaing_blend_start`` (float in [0, 1] [default = 0.2]): Normalized
      toroidal flux coordinate :math:`\hat{\rho}` where the sigmoid transition
      between Shaing and Angioni-Sauter is centered.
    - ``shaing_blend_rate`` (float [default = 5.0]): Steepness of the sigmoid
      transition between the Shaing and Angioni-Sauter models.

All neoclassical transport models share the following clipping bounds:

``chi_min`` (float [default = 0.0])
  Lower allowed bound for neoclassical heat conductivities
  :math:`\chi_\mathrm{neo}`, in units of :math:`m^2/s`.

``chi_max`` (float [default = 100.0])
  Upper allowed bound for neoclassical heat conductivities
  :math:`\chi_\mathrm{neo}`, in units of :math:`m^2/s`.

``D_e_min`` (float [default = 0.0])
  Lower allowed bound for neoclassical particle diffusivity
  :math:`D_\mathrm{neo}`, in units of :math:`m^2/s`.

``D_e_max`` (float [default = 100.0])
  Upper allowed bound for neoclassical particle diffusivity
  :math:`D_\mathrm{neo}`, in units of :math:`m^2/s`.

``V_e_min`` (float [default = -50.0])
  Lower allowed bound for neoclassical particle convection
  :math:`V_\mathrm{neo}` in units of :math:`m/s`.

``V_e_max`` (float [default = 50.0])
  Upper allowed bound for neoclassical particle convection
  :math:`V_\mathrm{neo}` in units of :math:`m/s`.

poloidal_velocity
=================
``model_name`` (str [default = ``'kim'``])
  The name of the neoclassical ion poloidal velocity model. If the
  ``poloidal_velocity`` section or ``model_name`` is omitted, ``'kim'`` is
  used. The following models are supported:

  * ``'kim'`` (default): Analytical poloidal velocity model from |kim1991|.
  * ``'zeros'``: Sets the neoclassical poloidal velocity to zero.

``poloidal_velocity_multiplier`` (float [default = 1.0])
  Multiplier applied to the neoclassical poloidal velocity.

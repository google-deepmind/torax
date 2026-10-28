.. include:: ../links.rst

.. _physics_neoclassical:

Neoclassical physics
####################

TORAX computes the bootstrap current density :math:`j_{\parallel,\mathrm{bs}}`,
the parallel conductivity :math:`\sigma_\parallel`, neoclassical heat and
particle transport coefficients, and the neoclassical ion poloidal velocity
:math:`v_\theta`. By default, each is provided by a separately configurable
analytical sub-model (see :ref:`neoclassical_config` for options and defaults):

* **Bootstrap current** (``bootstrap_current``): Sauter |sauter99|, Redl
  |redl2021|, or zero.
* **Conductivity** (``conductivity``): Sauter |sauter99| or Redl |redl2021|.
* **Transport** (``transport``): Angioni-Sauter |angioni2000|, providing
  :math:`\chi_{i,\mathrm{neo}}`, :math:`\chi_{e,\mathrm{neo}}`,
  :math:`D_{e,\mathrm{neo}}`, and :math:`V_{e,\mathrm{neo}}` (including the
  Ware pinch), with an optional near-axis ion heat conductivity correction from
  |shaing1997|; or zero.
* **Poloidal velocity** (``poloidal_velocity``): Kim |kim1991|, or zero. See
  :ref:`rotation_physics`.

The analytical bootstrap current, conductivity, and transport formulations all
depend on the effective trapped particle fraction, :math:`f_t`. By default this
is calculated with the analytical approximation of |sauter16| (Eqs. 33-34),
which only requires the local inverse aspect ratio and triangularity of each
flux surface.

Where available, TORAX can instead use the full bounce-averaged trapped
particle fraction integral |lin-liu-miller1995|,

.. math::

  f_t = 1 - \frac{3}{4} \langle B^2 \rangle
  \int_0^{1/B_\mathrm{max}} \frac{\lambda \, d\lambda}{\langle \sqrt{1 -
  \lambda B} \rangle},

evaluated directly from the poloidal variation of :math:`B` on each flux
surface, where :math:`\langle \cdot \rangle` denotes a flux surface average.
To eliminate the logarithmic derivative singularity of
:math:`\langle \sqrt{1 - \lambda B} \rangle` at the trapping boundary
:math:`\lambda = 1 / B_\mathrm{max}`, TORAX evaluates the integral using the
coordinate transformation :math:`y = \sqrt{1 - \lambda B_\mathrm{max}}` from
|lin-liu-miller1995|:

.. math::

  \int_0^{1/B_\mathrm{max}} \frac{\lambda \, d\lambda}{\langle \sqrt{1 -
  \lambda B} \rangle}
  = \frac{2}{B_\mathrm{max}^2} \int_0^1 \frac{y (1 - y^2)}{\left\langle
  \sqrt{1 - (1 - y^2) \frac{B}{B_\mathrm{max}}} \right\rangle} \, dy.

This is most impactful at low aspect ratio (e.g. spherical tokamaks), where
the analytical approximation is least accurate.

Custom neoclassical models, for example an integrated solver computing all of
the above together, can be registered instead. See :ref:`model-integration`.

.. include:: ../links.rst

.. _physics_sources:

Sources
#######

The source terms in the :ref:`equations` are comprised of a summation of
individual source/sink terms. Each of these terms can be configured to be
either:

- **Implicit:** Where needed in the theta-method, the source term is
  calculated based on the current guess for the state at :math:`t+\Delta t`.

- **Explicit:**  The source term is always calculated based on the state of
  the system at the beginning of the timestep, even if the solver
  :math:`\theta>0`. This makes the PDE system less nonlinear, avoids the
  incorporation of the source in the residual Jacobian if solving with
  Newton-Raphson, and leads to a single source calculation per timestep.

Explicit treatment is less accurate, but can be justified and computationally
beneficial for sources with complex but slow-evolving physics. Furthermore,
explicit source calculations do not need to be JAX-compatible, since explicit
sources are an input into the PDE solver, and do not require JIT compilation.
Conversely, implicit treatment can be important for accurately resolving the
impact of fast-evolving source terms.

All sources can optionally be set to zero, prescribed with explicit values or
calculated with a dedicated physics-based model. However, the code modular
structure facilitates easy coupling of additional source models in future work.
Specifics of source models currently implemented in TORAX follow:

Ion-electron heat exchange
==========================
The collisional heat exchange power density is calculated as

.. math::

  Q_{ei} = \frac{1.5 n_e (T_i - T_e)}{A_i m_p \tau_e},

where :math:`A_i` is the atomic mass number of the main ion species,
:math:`m_p` is the proton mass, and :math:`\tau_e` is the electron collision
time, given by:

.. math::

  \tau_e = \frac{12 \pi^{3/2} \epsilon_0^2 m_e^{1/2} (k_B T_e)^{3/2}}
  {n_e e^4 \ln \Lambda_{ei}},

where :math:`\epsilon_0` is the permittivity of free space, :math:`m_e` is the
electron mass, :math:`e` is the elementary charge, and :math:`\ln \Lambda_{ei}`
is the Coulomb logarithm for electron-ion collisions given by:

.. math::

  \ln \Lambda_{ei} = 15.2 - 0.5 \ln \left(\frac{n_e}{10^{20}
  \text{ m}^{-3}}\right) + \ln (T_e)

:math:`Q_{ei}` is added to the electron heat sources, meaning that positive
:math:`Q_{ei}` with :math:`T_i>T_e` heats the electrons. Conversely,
:math:`-Q_{ei}` is added to the ion heat sources.

Fusion power
============
TORAX optionally calculates the fusion power density assuming a 50-50
deuterium-tritium (D-T) fuel mixture using the |bosch-hale| parameterization
for the D-T fusion reactivity :math:`\langle \sigma v \rangle`:

.. math::

  P_{fus} = E_{fus} \frac{1}{4} n_i^2 \langle \sigma v \rangle

where :math:`E_{fus} = 17.6` MeV is the energy released per fusion reaction,
:math:`n_i` is the ion density, and :math:`\langle \sigma v \rangle` is given
by:

.. math::

  \langle \sigma v \rangle = C_1 \theta \sqrt{\frac{\xi}{m_rc^2 T_i^3}}
  \exp(-3\xi)

with

.. math::

  \theta =
  \frac{T_i}{1-\frac{T_i (C_2+T_i(C_4+T_iC_6))}{1+T_i(C_3+T_i(C_5+T_i C_7))}}

and

.. math::

  \xi = \left(\frac{B_G^2}{4\theta}\right)^{1/3}

where :math:`T_i` is the ion temperature in keV, :math:`m_rc^2` is the reduced
mass of the D-T pair. The values of :math:`m_rc^2`, the Gamov constant
:math:`B_G`, and the constants :math:`C_1` through :math:`C_7` are provided in
the Bosch-Hale paper.

TORAX partitions the fusion power between ions and electrons using the
parameterized fusion particle slowing down model of Mikkelsen, which neglects
the slowing down time itself.

Ohmic power
===========
The Ohmic power density, :math:`P_\mathrm{ohm}`, arising from collisional Joule
dissipation of the Ohmic current, is calculated as:

.. math::

  P_\mathrm{ohm} =
   \frac{\sigma_{\parallel} \langle \mathbf{E} \cdot \mathbf{B} \rangle^2}
   {\langle B^2 \rangle}, \qquad
  \langle \mathbf{E} \cdot \mathbf{B} \rangle =
   \frac{F g_3}{2\pi} \left(
   \frac{\partial \psi}{\partial t}
   - \frac{\hat{\rho}\dot{\Phi}_b}{2\Phi_b}\frac{\partial \psi}{\partial \hat{\rho}}
   \right)

where :math:`\sigma_{\parallel}` is the neoclassical parallel conductivity,
:math:`\langle B^2 \rangle` is the flux-surface-averaged magnetic field squared
(``gm5``), and
:math:`\frac{\partial \psi}{\partial t}` is calculated according to the
:math:`\psi` equation in the :ref:`equations`. :math:`P_\mathrm{ohm}` is then
included as a source term in the electron heat transport equation.

The inductive electric field requires :math:`\partial \psi / \partial t` at
fixed toroidal flux :math:`\Phi`, whereas the :math:`\psi` equation evolves
:math:`\psi` at fixed :math:`\hat{\rho}`. Since
:math:`\Phi = \Phi_b \hat{\rho}^2`, a surface at fixed :math:`\hat{\rho}`
moves to a different :math:`\Phi` when :math:`\Phi_b` changes in time, and the
two time derivatives differ by the grid-motion term
:math:`\frac{\hat{\rho}\dot{\Phi}_b}{2\Phi_b}\frac{\partial \psi}{\partial
\hat{\rho}}` subtracted above (see e.g. Pereverzev & Yushmanov, ASTRA Report
IPP 5/98, 2002, Eqs. 27 and 44).

This is the Joule dissipation of the Ohmic (inductive) current
:math:`\mathbf{j}_\mathrm{ohm}`, defined by the flux-surface-averaged parallel
Ohm's law :math:`\langle \mathbf{j}_\mathrm{ohm} \cdot \mathbf{B} \rangle =
\sigma_{\parallel} \langle \mathbf{E} \cdot \mathbf{B} \rangle`
(|hinton1976|). :math:`\mathbf{j}_\mathrm{ohm}` is field-aligned
and divergence-free, :math:`\mathbf{j}_\mathrm{ohm} = K \mathbf{B}` with
:math:`0 = \nabla \cdot (K \mathbf{B}) = \mathbf{B} \cdot \nabla K`, so
:math:`K` is constant on a flux surface. Then
:math:`P_\mathrm{ohm} = \langle \mathbf{E} \cdot \mathbf{j}_\mathrm{ohm}
\rangle = K \langle \mathbf{E} \cdot \mathbf{B} \rangle` with
:math:`K = \langle \mathbf{j}_\mathrm{ohm} \cdot \mathbf{B} \rangle /
\langle B^2 \rangle`, which gives the expression above and
:math:`P_\mathrm{ohm} \geq 0`. It excludes the power
:math:`\langle \mathbf{E} \cdot \mathbf{j}_\mathrm{ni} \rangle`
exchanged between the non-inductive current and the poloidal field, which is
not collisional heating of the thermal electrons.

Auxiliary Heating and Current Drive
===================================
For prescribing generic non-physics-based auxiliary heating and current drive
sources, TORAX provides built-in Gaussian formulations of a generic ion and
electron heat source, and a generic current drive source, with time-dependent
user configurable locations, Gaussian width, amplitude, and fractional heating
of ions and electrons.

More sophisticated physics-based models and/or ML-surrogates of specific
auxiliary heating and current drive sources can be coupled modularly to TORAX,
enhancing the fidelity of the simulation. By setting these as explicit sources,
these can also come from external codes (not necessarily JAX compatible) coupled
to TORAX in larger workflows.

Available physics-based models and/or ML-surrogates are listed below.

Electron-Cyclotron Heating and Current Drive
--------------------------------------------
The electron-cyclotron current drive can be calculated from the heating power
density, :math:`Q_\mathrm{EC}(\rho) [Wm^{-3}]`, and a dimensionless EC current
drive efficiency profile, :math:`\zeta_\mathrm{EC}(\rho)` . The current drive
parallel to the magnetic field, in :math:`[Am^{-2}]`, is then given by:

.. math::

    \langle j_\mathrm{EC} \cdot B \rangle = \frac{2\pi\epsilon_0^2 F}
    {e^3 R_\mathrm{maj}} \frac{T_e}{n_e} \zeta_{EC} Q_\mathrm{EC}

where :math:`\epsilon_0` is the vacuum permittivity, :math:`F = B_\phi R` is the
flux function, :math:`e` is the elementary charge, :math:`R_\mathrm{maj}` is the
device major radius, :math:`T_e` is the electron temperature in joules, and
:math:`n_e` is the electron density per cubic meter. This relationship is based
on the Lin-Liu model |lin-liu|. The derivation can be found
:ref:`here <ec-derivation>`.


Particle Sources
================
Similar to auxiliary heating and current drive, particle sources can also be
configured using either prescribed formulas. Presently, TORAX provides three
built-in formula-based particle sources for the :math:`n_e` equation:

- **Gas Puff:** An exponential function with configurable parameters models
  the ionization of neutral gas injected from the plasma edge.

- **Pellet Injection:** A Gaussian function approximates the deposition of
  particles from pellets injected into the plasma core. The time-dependent
  configuration parameter feature allows either a continuous approximation or
  discrete pellets to be modelled.

- **Generic particle source:**  An additional Gaussian function which can
  be configured to model arbitrary particle sources, e.g. to mock-up an NBI
  source.

Radiation
=========

Bremsstrahlung
--------------

Uses the model from Wesson, John, and David J. Campbell. Tokamaks. Vol. 149.
An optional correction for relativistic effects from Stott PPCF 2005 can be
enabled with the flag ``use_relativistic_correction``.

When the Mavrin impurity radiation model is also active, the bremsstrahlung
source automatically excludes the impurity bremsstrahlung component (using only
the main-ion contribution to :math:`Z_\text{eff}`) to avoid double-counting,
since the Mavrin model already accounts for impurity bremsstrahlung via
higher-fidelity fits to ADAS data.

Cyclotron Radiation
-------------------

Uses the total radiation power from |albajar2001|, with a deposition profile
from |artaud2018|. The Albajar model includes a parameterization of the
temperature profile which in TORAX is fit by simple grid search for
computational efficiency.

Impurity Radiation
------------------

The following models are available:

* Set the impurity radiation to be a constant fraction of the total external
  input power.

* Polynomial fits to ADAS data from Mavrin |mavrin|. Provides radiative cooling
  rates for the following impurity species:

  * Helium
  * Lithium
  * Beryllium
  * Carbon
  * Nitrogen
  * Oxygen
  * Neon
  * Argon
  * Krypton
  * Xenon
  * Tungsten

  These cooling curves are multiplied by the electron density and impurity
  densities to obtain the impurity radiation power density.

  The valid temperature range of the fit is [0.1-100] keV.

Ion Cyclotron Resonance Heating
===============================

TORAX supports ICRH models selectable via the ``model_name`` discriminator in
the ``icrh`` source configuration. Custom models can also be registered via
``torax.sources.register_source_config`` (see :ref:`model-integration`).

**Common parameters** shared by all ICRH models are the total injected power
:math:`P_\mathrm{total}` and the absorption fraction :math:`\alpha`. The
absorbed power is :math:`P_\mathrm{abs} = P_\mathrm{total} \times \alpha`.

Scaled Profile (``scaled_profile``)
-----------------------------------

This is the default ICRH model. It is a lightweight model that takes
user-prescribed reference heating profiles and adapts them to different magnetic
field operating points using a physically-motivated radial shift.

The model performs two operations:

1. **Resonance shift:** The ICRH resonance occurs where
   :math:`\omega = n \omega_{ci}(R)`, and since the vacuum toroidal field scales
   as :math:`B_t \propto 1/R`, the resonance radius scales linearly with
   the on-axis field: :math:`R_\mathrm{res} \propto B_0`. When the actual
   :math:`B_0` differs from the reference field :math:`B_{0,\mathrm{ref}}`, the
   heating profile shifts in normalised radius space.

   For each grid point, the shifted outboard midplane radius is:

   .. math::

     R'_\mathrm{out}(\rho) = R_\mathrm{out}(\rho) \cdot \frac{B_0}{B_{0,\mathrm{ref}}}

   where :math:`R_\mathrm{out}(\rho) = R_\mathrm{major} + r(\rho)` is the
   outboard midplane radius, which is monotonically increasing with
   :math:`\rho`. The reference profiles are then evaluated at the :math:`\rho`
   coordinate that corresponds to :math:`R'_\mathrm{out}` on the original
   :math:`R_\mathrm{out}(\rho)` mapping.

2. **Power normalisation:** The shifted ion and electron heating profiles are
   rescaled so that the volume-integrated total heating equals
   :math:`P_\mathrm{total} \times \alpha`.

This model does not include a fast-ion calculation; all fast-ion outputs are
zero. It is useful when reference profiles from a full-wave RF code are
available and need to be quickly approximated for different field scenarios.

.. toctree::
   :hidden:
   :maxdepth: 1

   electron-cyclotron-derivation

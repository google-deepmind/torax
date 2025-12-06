.. include:: links.rst

.. _physics_models:

Physics models
##############

TORAX provides a modular framework for incorporating various physics models.
This section summarizes the presently implemented models for geometry,
transport, sources, and neoclassical physics. Extended physics features are a
focus of the short-term development roadmap. Specific plans are summarized
following each subsection.

Magnetic geometry
=================
TORAX presently supports four geometry models for determining the metric
coefficients and flux surface averaged geometry variables required for the
transport equations.

  - **CHEASE:** This model utilizes equilibrium data from the |chease| fixed
    boundary Grad-Shafranov equilibrium code. Users provide a CHEASE output
    file, and TORAX extracts the relevant geometric quantities.

  - **FBT:** This model utilizes equilibrium data from the |fbt| free boundary
    Grad-Shafranov equilibrium code. Users provide FBT output files, and TORAX
    extracts the relevant geometric quantities.

  - **EQDSK:** This model utilizes equilibrium data provided in the EQDSK
    format. Users provide EQDSK formatted files. TORAX extracts the relevant
    geometric quantities and carries out the necessary flux surface averaging
    based on traced contours of the 2D poloidal flux data.

  - **Circular:** For testing and demonstration purposes, TORAX includes a
    simplified circular geometry model. This model assumes a circular plasma
    cross-section and corrects for elongation to approximate the metric
    coefficients.

Using :math:`\psi` and the magnetic geometry terms, the toroidal current density
is calculated as follows:

.. math::

  j_\mathrm{tor} = \frac{R_0^2}{8\pi^3\mu_0}\frac{\partial }{\partial V}
  \left(\frac{g_2 g_3 J}{\rho} \frac{\partial \psi}{\partial \rho}\right)

where :math:`V` is the volume enclosed by a flux surface.

The safety factor :math:`q` is calculated as follows:

.. math::

  q = 2\pi B_0 \rho \left( \frac{\partial \psi}{\partial \rho} \right)^{-1}

With :math:`q(\rho=0)=\frac{2B_0}{\mu_0 j_{tot}(\rho=0) R_0}`.

For simulations of tokamak scenarios with dynamic equilibra, sequences of
geometry data can be provided. TORAX interpolates geometry data at each end
of each timestep interval, using both sets of values in the theta solver method.
Time-derivative terms are calculated based on finite differences. Generalization
to enable self-consistent geometry calculations by models within the solver loop
is planned.

Plasma composition, initial and prescribed conditions
=====================================================

Presently, TORAX accommodates a single main ion species and single impurity
species, which can be comprised of time-dependent mixtures of ions with
fractional abundances summing to 1. This is useful for example for simulating
isotope mixes. Based on the ion symbols and fractional abundances, the average
mass of each species is determined. The average charge state of each ion in each
mixture is determined by Mavrin polynomial fits to ADAS atomic data |mavrin|.
For the temperature ranges of interest in the tokamak core, these are well
approximated as 1D functions of electron temperature. All ions with atomic
numbers below Carbon are assumed to be fully ionized.

The impurity and main ion densities are constrained by the plasma effective
charge, :math:`Z_\mathrm{eff}`, which is a user-provided 2D array in both time
and space, as well as quasineutrality.

:math:`n_i`, and :math:`n_{imp}`, are solved from the
following system of equations, where :math:`Z_\mathrm{eff}` and the electron
density are known, and :math:`Z_\mathrm{imp}` is the average impurity charge of
the impurity mixture, with the average charge state for each ion determined from
the Mavrin polynomials.

.. math::

  n_\mathrm{i}Z_\mathrm{i}^2 + n_\mathrm{imp}Z_\mathrm{imp}^2 =
  n_\mathrm{e}Z_\mathrm{eff}

  n_\mathrm{i}Z_\mathrm{i} + n_\mathrm{imp}Z_\mathrm{imp} = n_\mathrm{e}

Initial conditions for the evolving profiles :math:`T_i`, :math:`T_e`,
:math:`n_e`, and :math:`\psi` are user-configurable.

Additionally, if any of the :math:`T_{i,e}`, :math:`n_e` equations are set to be
non-evolving (i.e., not evolved by the PDE solver), then time-dependent
prescribed profiles are user-configurable.

For the poloidal flux :math:`\psi(\hat{\rho})`, the user specifies if the
initial condition is based on a prescribed current profile,
:math:`j_\mathrm{tor}=j_0(1-\hat{\rho}^2)^\nu` (with :math:`j_0`
scaled to match :math:`I_p`, and :math:`\nu` is user-configurable), or from the
:math:`\psi` provided in a geometry file. The prescribed current profile
option is always used for the circular geometry. The total current :math:`I_p`
can be user-configured or determined by the geometry data. In the latter
case, the :math:`\psi` provided by the geometry data can still be used, but is
scaled by the ratio of the desired :math:`I_p` and the geometry-provided
:math:`I_p`.

Transport models
================
Turbulent transport determines the values of the transport coefficients
(:math:`\chi_i`, :math:`\chi_e`, :math:`D_e`, :math:`V_e`) in the
:ref:`equations`, and is a key ingredient in core transport simulations.
Theory-based turbulent transport models provide the largest source of
nonlinearity in the PDE system. TORAX currently offers five transport models:

  - **Constant:** This simple model sets all transport coefficients to constant,
    user-configurable values. While not physically realistic, it can be useful
    for testing purposes.

  - **CGM:** The critical gradient model (CGM) is a simple theory-based model,
    capturing the basic feature of tokamak turbulent transport, critical
    temperature gradient transport thresholds. The model is a simple way to
    introduce transport coefficient nonlinearity, and is mainly used for testing
    purposes.

    .. math::

      \chi_i = \begin{cases}
      \chi_{GB} \text{C} (\frac{R}{L_{Ti}} -
      \frac{R}{L_{Ti,\textit{crit}}})^{\\alpha} & \text{if } \frac{R}{L_{Ti}}
      \ge \frac{R}{L_{Ti,\textit{crit}}} \\
      \chi_{min}  & \text{if } \frac{R}{L_{Ti}}
      < \frac{R}{L_{Ti,\textit{crit}}}
      \end{cases}

    with the GyroBohm scaling factor

    .. math::

      \chi_{GB} = \frac{(A_i m_p)^{1/2}}{(eB_0)^2}
      \frac{(T_i k_B)^{3/2}}{R_\textit{maj}}

    and the |guo-romanelli| ion-temperature-gradient (ITG) mode critical
    gradient formula.

    .. math::

      R/L_{Ti,crit} = \frac{4}{3}(1 + T_i/T_e)(1 + 2|\hat{s}|/q)

    where :math:`\chi_\textit{min}` is a user-configurable minimum allowed
    :math:`\chi`, :math:`L_{Ti}\equiv-\frac{T_i}{\nabla T_i}` is the ion
    temperature gradient length, :math:`A_i` is the main ion atomic mass number,
    :math:`m_p` the proton mass, :math:`e` the electron charge, :math:`B_0` the
    magnetic field on axis, and :math:`R_\mathrm{maj}` the major radius. The
    stiffness factor :math:`C` and the exponent :math:`\alpha` are
    user-configurable model parameters.

    Regarding additional transport coefficient outputs, the electron heat
    conductivity, :math:`\chi_e`, and particle diffusivity, :math:`D_e`, are
    scaled to :math:`\chi_i` using user-configurable model parameters. The
    particle convection velocity, :math:`V_e`, is user-defined.

  - **Bohm-GyroBohm:** A widely used semi-empirical model summing terms
    proportional to Bohm and gyro-Bohm scaling factors |bohm-gyrobohm|.

    The heat diffusivities for electrons and ions are given by:

    .. math::

      \chi_e = \alpha_{e, \text{B}} \chi_{e, \text{B}} + \alpha_{e, \text{gB}}
      \chi_{e, \text{gB}}

    .. math::

      \chi_i = \alpha_{i, \text{B}} \chi_{i, \text{B}} + \alpha_{i, \text{gB}}
      \chi_{i, \text{gB}}

    where :math:`\alpha_{s, \text{B}}` and :math:`\alpha_{s, \text{gB}}` are
    the coefficients for the Bohm and gyro-Bohm contribution for species
    :math:`s` respectively. These are given by:

    .. math::

      \chi_{e, \text{B}}
        = 0.5 \chi_{i, \text{B}}
        = \frac{R_\text{min} q^2}{e B_\text{0} n_e}
          \left|
            \frac{\partial p_e}{\partial \rho_{\text{tor}}}
          \right|

    .. math::

      \chi_{e, \text{gB}}
        = 2 \chi_{i, \text{gB}}
        =  \frac{\sqrt{T_e}}{B_\text{0}^2}
          \left|
            \frac{\partial T_e}{\partial \rho_{\text{tor}}}
          \right|

    where :math:`R_\text{min}` is the minor radius, :math:`q` is the safety
    factor, :math:`e` is the elementary charge, :math:`B_\text{0}` is the
    toroidal magnetic field at the magnetic axis, :math:`n_e` is the electron
    density, :math:`\rho_{\text{tor}}` is the (unnormalized) toroidal flux
    coordinate, :math:`p_e` is the electron pressure, and :math:`T_e` is the
    electron temperature.

    The electron diffusivity is given by |garzotti2003|:

    .. math::
      D_e = \eta \frac{\chi_e \chi_i}{\chi_e + \chi_i}

    where :math:`\eta` is a weighting factor given by:

    .. math::

      \eta = c_1 + (c_2 - c_1) \rho_{\text{tor}}

    where :math:`c_1` and :math:`c_2` are user-defined parameters.

    There is little discussion in the literature about setting the electron
    convectivity from the Bohm/gyro-Bohm model. Following RAPTOR's
    ``vpdn_chiescal`` method, in TORAX, we set the electron convectivity
    proportional to the diffusivity,

    .. math::
      v_e = c_v D_e

    where :math:`c_v` is a user-defined parameter.

    The default values for the model parameters are as follows:

    :math:`\alpha_{e,i,\text{B}} = 8e^{-5}`

    :math:`\alpha_{e,i,\text{gB}} = 5e^{-6}`

    :math:`c_1 = 1.0`

    :math:`c_2 = 0.3`

    :math:`c_v = -0.1`

    Please note that the Bohm-GyroBohm model TORAX implementation is presently
    experimental and subject to ongoing verification against established
    simulations.

  - **QLKNN:** This is a ML-surrogate model trained on a large dataset of the
    |qualikiz| quasilinear gyrokinetic code. TORAX presently employs the
    recently released open source |qlknn_7_11| model, based on QuaLiKiz v2.8.1
    data. Its training data contains a combination of a 11D input hypercube for
    the plasma bulk region, and a 7D input hypercube focusing on the L-mode edge
    region near the LCFS. The model provides separate ouputs for
    ion-temperature-gradient (ITG),  trapped-electron-mode (TEM), and
    electron-temperature-gradient (ETG) mode turbulent fluxes. The NNs take as
    input local plasma parameters, such as normalized gradients of temperature
    and density (both electron and impurity density gradients), temperature
    ratios, safety factor, magnetic shear, main ion dilution, and normalized
    collisionality, and outputs turbulent fluxes for ion and electron heat and
    particle transport. The QLKNN model is significantly faster than direct
    gyrokinetic simulations, enabling fast and accurate simulation within its
    range of validity. The ability to seamlessly couple ML-surrogate models is a
    key TORAX feature. TORAX is also coupled to the QLKNN-hyper-10D model
    |qlknn10d|, including dedicated JAX inference code written in |flax|.

  - **QuaLiKiz:** TORAX can be configured to use the |qualikiz| quasilinear
    gyrokinetic transport model itself. Since QuaLiKiz is an external code
    (written in Fortran), both |qualikiz| and its associated
    |qualikiz-pythontools| must be installed separately. The path to the
    QuaLiKiz executable must be set in a ``TORAX_QLK_EXEC_PATH`` environment
    variable. If this environment variable is not set, then the default is
    ``~/qualikiz/QuaLiKiz``. See above links for installation instructions.
    QuaLiKiz and TORAX exchange data via file I/O on a temporary directory.
    Since transport model calls are ostensibly carried out within JAX-compiled
    functions, running with QuaLiKiz requires disabling JAX compilation by
    setting ``TORAX_COMPILATION_ENABLED=False``. While other solutions exist,
    this is the simplest and most straightforward approach. In any case QuaLiKiz
    becomes the simulation bottleneck and limits the overall simulation speed
    regardless of JAX compilation in the rest of the stack. Furthermore,
    QuaLiKiz must be run with the ``linear`` solver, ideally with the
    ``use_predictor_corrector`` solver option, since ``newton_raphson`` requires
    autodiff which is not supported for QuaLiKiz. Running with QuaLiKiz is not a
    typical workflow due to its computational expense (2-3 orders of
    magnitude slower than with QLKNN). Its use-cases are:

      1. Evaluating ML-surrogates against their ground truth, i.e. for QLKNN.

      2. Carrying out higher-fidelity simulations to verify faster simulations
         carried out with ML-surrogates. It is a major advantage to be able to
         use TORAX as the same framework for both ML-surrogate and high-fidelity
         simulations.

For all transport models, optional spatial smoothing of the transport
coefficients using a Gaussian convolution kernel is implemented, to improve
solver convergence rates, an issue which can arise with stiff transport
coefficients such as from QLKNN. Furthermore, for all transport models, the user
can set inner (towards the center) and/or outer (towards the edge) radial zones
where the transport coefficients are prescribed to fixed values.

An edge-transport-barrier, or pedestal, is set up in TORAX through an adaptive
source term which sets a desired value (pedestal height) of
:math:`T_e`, :math:`T_i` and :math:`n_e`, at a user-configurable location
(pedestal width). Two different variants are available, one setting the pedestal
pressure and temperature ratios, and the other setting the pedestal temperatures
directly.

In the TORAX roadmap, coupling to additional transport models is envisaged,
including to additional semi-empirical models and H-mode confinement
scaling law adaptive models, as well as more ML-surrogates of theory-based
models, both for core turbulence and pedestal predictions. A more physically
consistent approach for setting up the pedestal will be implemented by
incorporating adaptive transport coefficients in the pedestal region, as opposed
to an adaptive local source/sink term.

Neoclassical physics
====================
TORAX employs the Sauter model |sauter99| to calculate the bootstrap current
density, :math:`j_{bs}`, and the neoclassical conductivity, :math:`\sigma_{||}`,
used in the current diffusion equation. The Sauter model is a widely-used
analytical formulation that provides a relatively fast and differentiable
approximation for these neoclassical quantities.

Future work can incorporate more recent neoclassical physics parameterizations,
and also set neoclassical transport coefficients themselves. This can be of
importance for ion heat transport in the inner core. When extending TORAX to
include impurity transport, incorporating fast analytical neoclassical models
for heavy impurity transport will be of great importance.

Edge models
===========

TORAX supports coupling to reduced edge/divertor models to provide physically
consistent boundary conditions for the core transport solver. Currently, only
the Extended Lengyel model is supported.

Extended Lengyel Model
----------------------
The Extended Lengyel model describes the 1D parallel heat and particle transport
in the Scrape-Off Layer (SOL) and divertor. It extends the classic Lengyel model
by including cross-field transport in the divertor, power and momentum loss due
to neutral ionization close to the divertor target and turbulent broadening of
the upstream heat flux channel. The implementation follows |body2025|.

The model relates upstream quantities (e.g. power crossing separatrix
:math:`P_{SOL}`, separatrix density :math:`n_{e,sep}`) and divertor impurity
content, to downstream quantities such as target temperature :math:`T_{e,t}`.
The separatrix temperature is calculated from a 2-point model.

**Modes of Operation:**

*   **Forward Mode**: Given the impurity mix and upstream conditions, calculate
    the target temperature :math:`T_{e,t}`.

*   **Inverse Mode**: Given a desired target temperature :math:`T_{e,t}`
    (e.g., 5 eV to represent detachment onset), calculate the required
    concentration of a given mix of seeded impurity species.

**Physics Features:**

*   **Impurity Radiation**: Radiative cooling rates :math:`L_z(T_e)` are
    calculated using polynomial fits from |mavrin2017|, which are valid at low
    edge temperatures (down to ~1 eV).

*   **Enrichment**: Impurity enrichment (:math:`c_{div}/c_{core}`) can be
    specified manually or calculated using the empirical scaling from
    |kallenbach2024|:
    :math:`E \propto Z^{-0.5} p_0^{-0.4} (E_{ion,z}/E_{ion,D})^{-5.8}`.

**Coupling to Core:**

When enabled, the edge model updates the following TORAX runtime parameters for
the next time step. This is done via explicit coupling: the parameters at time
:math:`t` are passed to the edge model, which calculates the updated parameters
at time :math:`t+\Delta t`.

1.  **Boundary Conditions**: The separatrix electron temperature calculated by
    the model sets the core :math:`T_e` boundary condition. :math:`T_i` boundary
    condition is set via a configured ratio.

2.  **Impurity Composition**:

    *   In **Inverse Mode**, the required seeded impurity concentration updates
        the core impurity profile (scaled by enrichment factor).

    *   The model also enforces consistency for fixed background impurities
        between core and edge.

Sources
=======
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
--------------------------
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
------------
TORAX optionally calculates the fusion power density assuming a 50-50
deuturium-tritium (D-T) fuel mixture using the |bosch-hale| parameterization
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
-----------
The Ohmic power density, :math:`P_\mathrm{ohm}`, arising from resistive
dissipation of the plasma current, is calculated as:

.. math::

  P_\mathrm{ohm} =
   \frac{j_\mathrm{tor} }{2 \pi R_\mathrm{maj}}\frac{\partial \psi}{\partial t}

where :math:`j_\mathrm{tor}` is the toroidal current density, and
:math:`R_\mathrm{maj}` is the major radius. The loop voltage
:math:`\frac{\partial \psi}{\partial t}` is calculated according to the
:math:`\psi` equation in the :ref:`equations`. :math:`P_\mathrm{ohm}` is then
included as a source term in the electron heat transport equation.

Auxiliary Heating and Current Drive
-----------------------------------
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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
----------------
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
---------

Bremsstrahlung
^^^^^^^^^^^^^^

Uses the model from Wesson, John, and David J. Campbell. Tokamaks. Vol. 149.
An optional correction for relativistic effects from Stott PPCF 2005 can be
enabled with the flag "use_relativistic_correction".

Cyclotron Radiation
^^^^^^^^^^^^^^^^^^^

Uses the total radiation power from |albajar2001|, with a deposition profile
from |artaud2018|. The Albajar model includes a parameterization of the
temperature profile which in TORAX is fit by simple grid search for
computational efficiency.

Impurity Radiation
^^^^^^^^^^^^^^^^^^

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
-------------------------------

Presently this source is implemented for a SPARC specific ICRH scenario.

A core Ion Cyclotron Range of Frequencies (ICRF) heating surrogate model trained
on TORIC ICRH spectrum solver simulations is used to provide power profiles for
Helium-3, Tritium (via its second harmonic) and electrons |wallace2024|.

A "Stix distribution" [Stix, Nuc. Fus. 1975] is used to model the non-thermal
Helium-3 distribution based on an analytic solution to the Fokker-Planck
equation to estimate the birth energy of Helium-3.

TORAX partitions the Helium-3 power between ions and electrons using the
parameterized model of Mikkelsen, as for Fusion Power.

It is assumed that all tritium heating goes to ions and all electron heating
goes to electrons.

MHD models
==========

Currently only a sawtooth model is implemented, although the TORAX config and
internal APIs are designed to accommodate other models in the future.

Sawtooth Model
--------------

Sawteeth are periodic oscillations in the core plasma temperature, density, and
current caused by the growth and subsequent rapid reconnection of an m=1, n=1
kink instability within the plasma volume where the safety factor, :math:`q`,
drops below unity. The sawtooth crash is triggered by a state-dependent critical
magnetic shear at the :math:`q=1` surface.

The TORAX sawtooth model comprises two components:

  * Trigger Model: determines the conditions under which a sawtooth crash is
    initiated.

  * Redistribution Model: Modifies the plasma profiles (temperature, density,
    poloidal flux) following a crash to simulate the rapid transport event.

Currently only simple Trigger and Redistribution models are implemented.

The ``simple`` trigger model checks for the following conditions at each time
step:

1.  **Existence of a q=1 surface:** The safety factor profile `q` must drop
    below 1.

2.  **Minimum radius:** The normalized radius of the q=1 surface
    (:math:`\hat{\rho}_{q=1}`) must be greater than a specified minimum value
    (``minimum_radius``). This prevents spurious triggers very close to the
    magnetic axis.

3.  **Critical magnetic shear:** The magnetic shear (:math:`\hat{s}`) at the
    :math:`q=1` surface must exceed a critical value (``s_critical``).

The ``simple`` redistribution model mplements a simplified redistribution
process based on conserving particle number, energy, and current within a
mixing radius.

1.  **Mixing Radius Calculation:** The mixing radius (:math:`\hat{\rho}_{mix}`)
    is calculated as ``mixing_radius_multiplier`` * :math:`\hat{\rho}_{q=1}`.

2.  **Profile Flattening:** Profiles
    (:math:`T_i`, :math:`T_e`, :math:`n_e`, :math:`j_{tot}`) within the mixing
    radius are partially flattened.

    *   Inside the q=1 surface (:math:`\hat{\rho} < \hat{\rho}_{q=1}`), a
        linear profile is created, where the value at :math:`\hat{\rho}=0` is
        set to a multiple ``flattening_factor`` of the value at
        :math:`\hat{\rho}_{q=1}`.

    *   Between the :math:`q=1` surface and the mixing radius
        (:math:`\hat{\rho}_{q=1} \le \hat{\rho} < \hat{\rho}_{mix}`),
        the profile is linearly interpolated between the flattened value at
        :math:`\hat{\rho}_{q=1}` and the original profile value at
        :math:`\hat{\rho}_{mix}`.

    The values at :math:`\hat{\rho}_{q=1}` are set by conservation laws.

3.  **Conservation:** The flattening process conserves the total number of
    particles (:math:`\int n_e dV`), total electron and ion thermal energy
    (:math:`\int \frac{3}{2} n_{e,i} T_{e,i} dV`), and total current
    (:math:`\int j_{tot} dS`) within the mixing radius.
    The values at :math:`\hat{\rho}_{q=1}` after flattening are adjusted to
    ensure these conservation laws are met. The poloidal flux profile is then
    recalculated to be consistent with the flattened current profile, and the
    pre-crash poloidal flux boundary condition.

4.  **Derived Quantities:** After redistribution, derived quantities like
    ion densities (:math:`n_i`, :math:`n_{imp}`), impurity charge states
    (:math:`Z_{imp}`), safety factor (:math:`q`), magnetic shear
    (:math:`\hat{s}`), and sources, are recalculated based on the modified
    profiles.

Simulation Step During Crash
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When a sawtooth crash is triggered:

1.  The standard PDE time step is **skipped**.

2.  The redistribution model is applied to modify the ``core_profiles``.

3.  A short, fixed time step (user-configurable ``crash_step_duration``) is
    taken.

4.  Derived quantities are recalculated based on the redistributed profiles.

The simulation then always proceeds to the next regular PDE time step.
Subsequent sawtooth crashes are not allowed, to prevent continuous consecutive
crashes if the trigger condition is still met immediately after redistribution.

See the :ref:`sawtooth configuration details <sawtooth_config>` for a summary
of the input parameters.

The current sawtooth model is a simplified representation. Future development
plans include: Implementing more sophisticated trigger models
(e.g., based on the Porcelli model). Developing more physically detailed
redistribution models (e.g., Kadomtsev reconnection models).

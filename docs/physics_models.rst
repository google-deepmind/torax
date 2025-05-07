.. _physics_models:

Physics models
##############

TORAX provides a modular framework for incorporating various physics models.
This section summarizes the presently implemented models for geometry, transport, sources, and neoclassical physics.
Extended physics features are a focus of the short-term development roadmap. Specific plans are summarized following each subsection.

Magnetic geometry
=================
TORAX presently supports two geometry models for determining the metric coefficients and flux surface averaged
geometry variables required for the transport equations.

  - **CHEASE:** This model utilizes equilibrium data from the `CHEASE <https://doi.org/10.1016/0010-4655(96)00046-X>`_ fixed boundary
    Grad-Shafranov equilibrium code. Users provide a CHEASE output file, and TORAX extracts the relevant geometric quantities.

  - **FBT:** This model utilizes equilibrium data from the `FBT <https://www.sciencedirect.com/science/article/pii/0010465588900410>`_ free boundary
    Grad-Shafranov equilibrium code. Users provide FBT output files, and TORAX extracts the relevant geometric quantities.

  - **Circular:** For testing and demonstration purposes, TORAX includes a simplified circular geometry model.
    This model assumes a circular plasma cross-section and corrects for elongation to approximate the metric coefficients.

Using :math:`\psi` and the magnetic geometry terms, the toroidal current density is calculated as follows:

.. math::

  j_\mathrm{tor} = \frac{R_0^2}{8\pi^3\mu_0}\frac{\partial }{\partial V}\left(\frac{g_2 g_3 J}{\rho} \frac{\partial \psi}{\partial \rho}\right)

where :math:`V` is the volume enclosed by a flux surface.

The safety factor :math:`q` is calculated as follows:

.. math::

  q = 2\pi B_0 \rho \left( \frac{\partial \psi}{\partial \rho} \right)^{-1}

With :math:`q(\rho=0)=\frac{2B_0}{\mu_0 j_{tot}(\rho=0) R_0}`.

To enable simulations of tokamak scenarios with dynamic equilibra, the TORAX roadmap
includes incorporating time-dependent geometry, e.g. by incorporating multiple equilibrium
files and interpolating the geometry variables at the times required by the solver.
Generalization to geometry data beyond CHEASE is also planned.

Plasma composition, initial and prescribed conditions
=====================================================

Presently, TORAX accommodates a single main ion species and single impurity species,
which can be comprised of time-dependent mixtures of ions with fractional abundances
summing to 1. This is useful for example for simulating isotope mixes. Based on the
ion symbols and fractional abundances, the average mass of each species is determined.
The average charge state of each ion in each mixture is determined by `Mavrin polynomials <https://doi.org/10.1080/10420150.2018.1462361>`_,
which are fitted to atomic data, and in the temperature ranges of interest in the tokamak core,
are well approximated as 1D functions of electron temperature. All ions with atomic numbers below
Carbon are assumed to be fully ionized.

The impurity and main ion densities are constrained by the plasma effective
charge, :math:`Z_\mathrm{eff}`, which is a user-provided 2D array in both time and space,
as well as quasineutrality.

:math:`n_i`, and :math:`n_{imp}`, are solved from the
following system of equations, where :math:`Z_\mathrm{eff}` and the electron density are
known, and :math:`Z_\mathrm{imp}` is the average impurity charge of the impurity mixture,
with the average charge state for each ion determined from the Mavrin polynomials.

.. math::

  n_\mathrm{i}Z_\mathrm{i}^2 + n_\mathrm{imp}Z_\mathrm{imp}^2 = n_\mathrm{e}Z_\mathrm{eff}

  n_\mathrm{i}Z_\mathrm{i} + n_\mathrm{imp}Z_\mathrm{imp} = n_\mathrm{e}

Initial conditions for the evolving profiles :math:`T_i`, :math:`T_e`, :math:`n_e`,
and :math:`\psi` are user-configurable. For :math:`T_{i,e}`, both the initial core
(:math:`r=0`) value and the boundary condition at :math:`\hat{\rho}=1` are provided.
Initial :math:`T_{i,e}` profiles are then a linear interpolation between these values.

For :math:`n_e`, the user specifies a line-averaged density, in either absolute units
or scaled to the Greenwald density :math:`n_\mathrm{GW}=\frac{I_p}{\pi a^2}~[10^{20} m^{-3}]`,
and the initial peaking factor. The initial :math:`n_e` profile, including the edge boundary
condition, is then a linear function scaled to match the desired line-averaged density and peaking.

If any of the :math:`T_{i,e}`, :math:`n_e` equations are set to be non-evolving (i.e., not evolved by the PDE solver),
then time-dependent prescribed profiles are user-configurable.

For the poloidal flux :math:`\psi(\hat{\rho})`, the user specifies if the initial condition
is based on a prescribed current profile, :math:`j_\mathrm{tor}=j_0(1-\hat{\rho}^2)^\current_profile_nu` (with :math:`j_0`
scaled to match :math:`I_p`, and :math:`\current_profile_nu` is user-configurable), or from the :math:`\psi` provided
in a CHEASE geometry file. The prescribed current profile option is always used for the circular geometry.
The total current :math:`I_p` can be user-configured or determined by the CHEASE geometry file.
In the latter case, the :math:`\psi` provided by CHEASE can still be used, but is scaled by the ratio
of the desired :math:`I_p` and the CHEASE :math:`I_p`.

In the development roadmap, more flexible initial condition setting is planned, such as from a wider
variety of formulas, from arbitrary arrays, or from arbitrary times from existing TORAX output files.

Transport models
================
Turbulent transport determines the values of the transport coefficients (:math:`\chi_i`, :math:`\chi_e`, :math:`D_e`, :math:`V_e`)
in the :ref:`equations`, and is a key ingredient in core transport simulations.
Theory-based turbulent transport models provide the largest source of nonlinearity in the PDE system.
TORAX currently offers three transport models:

  - **Constant:** This simple model sets all transport coefficients to constant, user-configurable values.
    While not physically realistic, it can be useful for testing purposes.

  - **CGM:** The critical gradient model (CGM) is a simple theory-based model, capturing the basic feature
    of tokamak turbulent transport, critical temperature gradient transport thresholds. The model is a simple
    way to introduce transport coefficient nonlinearity, and is mainly used for testing purposes.

    .. math::

      \chi_i = \begin{cases}
      \chi_{GB} \text{C} (\frac{R}{L_{Ti}} - \frac{R}{L_{Ti,\textit{crit}}})^{\fusion} & \text{if } \frac{R}{L_{Ti}} \ge \frac{R}{L_{Ti,\textit{crit}}} \\
      \chi_{min}  & \text{if } \frac{R}{L_{Ti}} < \frac{R}{L_{Ti,\textit{crit}}}
      \end{cases}

    with the GyroBohm scaling factor

    .. math::

      \chi_{GB} = \frac{(A_i m_p)^{1/2}}{(eB_0)^2}\frac{(T_i k_B)^{3/2}}{R_\textit{maj}}

    and the `Guo-Romanelli <https://doi.org/10.1063/1.860537>`_ ion-temperature-gradient (ITG)
    mode critical gradient formula.

    .. math::

      R/L_{Ti,crit} = \frac{4}{3}(1 + T_i/T_e)(1 + 2|\hat{s}|/q)

    where :math:`\chi_\textit{min}` is a user-configurable minimum allowed
    :math:`\chi`, :math:`L_{Ti}\equiv-\frac{T_i}{\nabla T_i}` is the ion temperature gradient length,
    :math:`A_i` is the main ion atomic mass number, :math:`m_p` the proton mass, :math:`e`
    the electron charge, :math:`B_0` the magnetic field on axis, and :math:`R_\mathrm{maj}` the major radius.
    The stiffness factor :math:`C` and the exponent :math:`\fusion` are user-configurable model parameters.

    Regarding additional transport coefficient outputs, the electron heat conductivity, :math:`\chi_e`,
    and particle diffusivity, :math:`D_e`, are scaled to :math:`\chi_i` using user-configurable model parameters.
    The particle convection velocity, :math:`V_e`, is user-defined.

  - **Bohm-GyroBohm:** A widely used semi-empirical model summing terms proportional to Bohm
    and gyro-Bohm scaling factors (`Erba et al, 1998 <doi.org/10.1088/0029-5515/38/7/305>`_).

    The heat diffusivities for electrons and ions are given by:

    .. math::

      \chi_e = \fusion_{e, \text{B}} \chi_{e, \text{B}} + \fusion_{e, \text{gB}}
      \chi_{e, \text{gB}}

    .. math::

      \chi_i = \fusion_{i, \text{B}} \chi_{i, \text{B}} + \fusion_{i, \text{gB}}
      \chi_{i, \text{gB}}

    where :math:`\fusion_{s, \text{B}}` and :math:`\fusion_{s, \text{gB}}` are the
    coefficients for the Bohm and gyro-Bohm contribution for species :math:`s`
    respectively. These are given by:

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

    The electron diffusivity is given by `Garzotti et al, 2003 <doi.org/10.1088/0029-5515/43/12/025>`_:

    .. math::
      D_e = \eta \frac{\chi_e \chi_i}{\chi_e + \chi_i}

    where :math:`\eta` is a weighting factor given by:

    .. math::

      \eta = c_1 + (c_2 - c_1) \rho_{\text{tor}}

    where :math:`c_1` and :math:`c_2` are user-defined parameters.

    There is little discussion in the literature about setting the electron convectivity from the Bohm/gyro-Bohm model.
    Following RAPTOR's `vpdn_chiescal` method, in TORAX, we set the electron convectivity proportional to the diffusivity,

    .. math::
      v_e = c_v D_e

    where :math:`c_v` is a user-defined parameter.

    The default values for the model parameters are as follows:

    :math:`\fusion_{e,i,\text{B}} = 8e^{-5}`

    :math:`\fusion_{e,i,\text{gB}} = 5e^{-6}`

    :math:`c_1 = 1.0`

    :math:`c_2 = 0.3`

    :math:`c_v = -0.1`

    Please note that the Bohm-GyroBohm model TORAX implementation is presently
    experimental and subject to ongoing verification against established simulations.

  - **QLKNN:** This is a ML-surrogate model trained on a large dataset of the `QuaLiKiz <https://gitlab.com/qualikiz-group/QuaLiKiz>`_
    quasilinear gyrokinetic code. Specifically, TORAX presently employs the QLKNN-hyper-10D model (`QLKNN10D <https://doi.org/10.1063/1.5134126>`_),
    which features a 10D input hypercube and separate NNs for ion-temperature-gradient (ITG),
    trapped-electron-mode (TEM), and electron-temperature-gradient (ETG) mode turbulent fluxes.
    The NNs take as input local plasma parameters, such as normalized gradients of temperature and density,
    temperature ratios, safety factor, magnetic shear, :math:`Z_{eff}`, and normalized collisionality,
    and outputs turbulent fluxes for ion and electron heat and particle transport.
    The QLKNN model is significantly faster than direct gyrokinetic simulations, enabling fast and accurate simulation
    within its range of validity. The ability to seamlessly couple ML-surrogate models is a key TORAX feature.
    TORAX depends only on the open source weights and biases of the QLKNN model, and includes dedicated
    JAX inference code written in `Flax <https://github.com/google/flax>`_.

  - **QuaLiKiz:** TORAX can be configured to use the `QuaLiKiz <https://gitlab.com/qualikiz-group/QuaLiKiz>`_
    quasilinear gyrokinetic transport model itself. Since QuaLiKiz is an external code (written in Fortran),
    both `QuaLiKiz <https://gitlab.com/qualikiz-group/QuaLiKiz>`_ and its associated
    `QuaLiKiz Pythontools <https://gitlab.com/qualikiz-group/QuaLiKiz-pythontools>`_ must be installed separately.
    (tag 1.4.1 or higher) The path to the QuaLiKiz executable must be set in the ``TORAX_QLK_EXEC_PATH`` environment variable.
    If this environment variable is not set, then the default is ``~/qualikiz/QuaLiKiz``.
    See above links for installation instructions. QuaLiKiz and TORAX exchange data via file I/O on
    a temporary directory. Since transport model calls are ostensibly carried out within JAX-compiled functions,
    running with QuaLiKiz requires disabling JAX compilation by setting ``TORAX_COMPILATION_ENABLED=False``.
    While other solutions exist, this is the simplest and most straightforward approach. In any case QuaLiKiz
    becomes the simulation bottleneck and limits the overall simulation speed regardless of JAX compilation in the
    rest of the stack. Furthermore, QuaLiKiz must be run with the ``linear`` solver, ideally with the
    ``use_predictor_corrector`` solver option, since ``newton_raphson`` requires autodiff which is not supported
    for QuaLiKiz. Running with QuaLiKiz is not a typical workflow due to its computational expense (2-3 orders of
    magnitude slower than with QLKNN). Its use-cases are:

      1. Evaluating ML-surrogates against their ground truth, i.e. for QLKNN, or as a template for how to carry this out for other ML-surrogates.

      2. Example of using TORAX as the PDE solver for a standard integrated modelling framework with higher fidelity models.

For all transport models, optional spatial smoothing of the transport coefficients using a Gaussian convolution kernel is
implemented, to improve solver convergence rates, an issue which can arise with stiff transport coefficients such
as from QLKNN. Furthermore, for all transport models, the user can set inner (towards the center) and/or outer
(towards the edge) radial zones where the transport coefficients are prescribed to fixed values.

An edge-transport-barrier, or pedestal, is set up in TORAX through an adaptive source
term which sets a desired value (pedestal height) of :math:`T_e`, :math:`T_i` and :math:`n_e`,
at a user-configurable location (pedestal width).

In the TORAX roadmap, coupling to additional transport models is envisaged, including
semi-empirical models such as Bohm/gyroBohm and H-mode confinement scaling law adaptive models,
as well as more ML-surrogates of theory-based models, both for core turbulence and pedestal
predictions. A more physically consistent approach for setting up the pedestal will be
implemented by incorporating adaptive transport coefficients in the pedestal region,
as opposed to an adaptive local source/sink term.

Neoclassical physics
====================
TORAX employs the `Sauter model <https://doi.org/10.1063/1.873240>`_ to calculate the
bootstrap current density, :math:`j_{bs}`, and the neoclassical conductivity,
:math:`\sigma_{||}`, used in the current diffusion equation. The Sauter model is
a widely-used analytical formulation that provides a relatively fast and differentiable
approximation for these neoclassical quantities.

Future work can incorporate more recent neoclassical physics parameterizations,
and also set neoclassical transport coefficients themselves. This can be of importance
for ion heat transport in the inner core. When extending TORAX to include impurity
transport, incorporating fast analytical neoclassical models for heavy impurity
transport will be of great importance.

Sources
=======
The source terms in the :ref:`equations` are comprised of a summation of individual
source/sink terms. Each of these terms can be configured to be either:

  - **Implicit:** Where needed in the theta-method, the source term is calculated based
    on the current guess for the state at :math:`t+\Delta t`.

  - **Explicit:**  The source term is always calculated based on the state of the system
    at the beginning of the timestep, even if the solver :math:`\theta>0`. This makes the
    PDE system less nonlinear, avoids the incorporation of the source in the residual
    Jacobian if solving with Newton-Raphson, and leads to a single source calculation per timestep.

Explicit treatment is less accurate, but can be justified and computationally beneficial for
sources with complex but slow-evolving physics. Furthermore, explicit source calculations do
not need to be JAX-compatible, since explicit sources are an input into the PDE solver,
and do not require JIT compilation. Conversely, implicit treatment can be important for accurately
resolving the impact of fast-evolving source terms.

All sources can optionally be set to zero, prescribed with explicit values or calculated with a dedicated physics-based model.
However, the code modular structure facilitates easy coupling of additional source models in future work. Specifics of source models
currently implemented in TORAX follow:

Ion-electron heat exchange
--------------------------
The collisional heat exchange power density is calculated as

.. math::

  Q_{ei} = \frac{1.5 n_e (T_i - T_e)}{A_i m_p \tau_e},

where :math:`A_i` is the atomic mass number of the main ion species,
:math:`m_p` is the proton mass, and :math:`\tau_e` is the electron collision time, given by:

.. math::

  \tau_e = \frac{12 \pi^{3/2} \epsilon_0^2 m_e^{1/2} (k_B T_e)^{3/2}}{n_e e^4 \ln \Lambda_{ei}},

where :math:`\epsilon_0` is the permittivity of free space, :math:`m_e` is the electron mass,
:math:`e` is the elementary charge, and :math:`\ln \Lambda_{ei}` is the Coulomb logarithm
for electron-ion collisions given by:

.. math::

  \ln \Lambda_{ei} = 15.2 - 0.5 \ln \left(\frac{n_e}{10^{20} \text{ m}^{-3}}\right) + \ln (T_e)

:math:`Q_{ei}` is added to the electron heat sources, meaning that positive :math:`Q_{ei}`
with :math:`T_i>T_e` heats the electrons. Conversely, :math:`-Q_{ei}` is added to the ion heat sources.

Fusion power
------------
TORAX optionally calculates the fusion power density assuming a 50-50 deuturium-tritium
(D-T) fuel mixture using the `Bosch-Hale <https://doi.org/10.1088/0029-5515/32/4/I07>`_ parameterization
for the D-T fusion reactivity :math:`\langle \sigma v \rangle`:

.. math::

  P_{fus} = E_{fus} \frac{1}{4} n_i^2 \langle \sigma v \rangle

where :math:`E_{fus} = 17.6` MeV is the energy released per fusion reaction,
:math:`n_i` is the ion density, and :math:`\langle \sigma v \rangle` is given by:

.. math::

  \langle \sigma v \rangle = C_1 \theta \sqrt{\frac{\xi}{m_rc^2 T_i^3}} \exp(-3\xi)

with

.. math::

  \theta = \frac{T_i}{1-\frac{T_i (C_2+T_i(C_4+T_iC_6))}{1+T_i(C_3+T_i(C_5+T_i C_7))}}

and

.. math::

  \xi = \left(\frac{B_G^2}{4\theta}\right)^{1/3}

where :math:`T_i` is the ion temperature in keV, :math:`m_rc^2` is the reduced mass of the D-T pair.
The values of :math:`m_rc^2`, the Gamov constant :math:`B_G`, and the constants :math:`C_1` through :math:`C_7`
are provided in the Bosch-Hale paper.

TORAX partitions the fusion power between ions and electrons using the parameterized
fusion particle slowing down model of Mikkelsen, which neglects the slowing down time itself.

Ohmic power
-----------
The Ohmic power density, :math:`P_\mathrm{ohm}`, arising from resistive dissipation of the plasma current, is calculated as:

.. math::

  P_\mathrm{ohm} = \frac{j_\mathrm{tor} }{2 \pi R_\mathrm{maj}}\frac{\partial \psi}{\partial t}

where :math:`j_\mathrm{tor}` is the toroidal current density, and :math:`R_\mathrm{maj}`
is the major radius. The loop voltage :math:`\frac{\partial \psi}{\partial t}` is calculated
according to the :math:`\psi` equation in the :ref:`equations`. :math:`P_\mathrm{ohm}`
is then included as a source term in the electron heat transport equation.

Auxiliary Heating and Current Drive
-----------------------------------
For prescribing generic non-physics-based auxiliary heating and current drive sources,
TORAX provides built-in Gaussian formulations of a generic ion and electron heat source,
and a generic current drive source, with time-dependent user configurable locations,
Gaussian width, amplitude, and fractional heating of ions and electrons.

More sophisticated physics-based models and/or ML-surrogates of specific auxiliary heating and current drive sources
can be coupled modularly to TORAX, enhancing the fidelity of the simulation. By setting these as explicit sources,
these can also come from external codes (not necessarily JAX compatible) coupled to TORAX in larger workflows.

Available physics-based models and/or ML-surrogates are listed below.

Electron-Cyclotron Heating and Current Drive
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The electron-cyclotron current drive can be calculated from the heating power density, :math:`Q_\mathrm{EC}(\rho) [Wm^{-3}]`,
and a dimensionless EC current drive efficiency profile, :math:`\zeta_\mathrm{EC}(\rho)` .
The current drive parallel to the magnetic field, in :math:`[Am^{-2}]`, is then given by:

.. math::

    \langle j_\mathrm{EC} \cdot B \rangle = \frac{2\pi\epsilon_0^2 F}{e^3 R_\mathrm{maj}} \frac{T_e}{n_e} \zeta_{EC} Q_\mathrm{EC}

where :math:`\epsilon_0` is the vacuum permittivity, :math:`F = B_\phi R` is the flux function, :math:`e` is the elementary charge,
:math:`R_\mathrm{maj}` is the device major radius, :math:`T_e` is the electron temperature in joules, and
:math:`n_e` is the electron density per cubic meter.
This relationship is based on the `Lin-Liu <https://doi.org/10.1063/1.1610472>`_ model. The derivation can be found :ref:`here <ec-derivation>`.


Particle Sources
----------------
Similar to auxiliary heating and current drive, particle sources can also be configured using either prescribed formulas.
Presently, TORAX provides three built-in formula-based particle sources for the :math:`n_e` equation:

  - **Gas Puff:** An exponential function with configurable parameters models the ionization
    of neutral gas injected from the plasma edge.

  - **Pellet Injection:** A Gaussian function approximates the deposition of particles from
    pellets injected into the plasma core. The time-dependent configuration parameter feature
    allows either a continuous approximation or discrete pellets to be modelled.

  - **Neutral Beam Injection (NBI):**  A Gaussian function models the ionization of neutral
    particles injected by a neutral beam.

Radiation
---------

Bremsstrahlung
^^^^^^^^^^^^^^

Uses the model from Wesson, John, and David J. Campbell. Tokamaks. Vol. 149.
An optional correction for relativistic effects from Stott PPCF 2005 can be enabled with the flag "use_relativistic_correction".

Cyclotron Radiation
^^^^^^^^^^^^^^^^^^^

Uses the total radiation power from `Albajar NF 2001 <https://doi.org/10.1088/0029-5515/41/6/301>`_
with a deposition profile from `Artaud NF 2018 <https://doi.org/10.1088/1741-4326/aad5b1>`_.
The Albajar model includes a parameterization of the temperature profile which in TORAX is fit by simple
grid search for computational efficiency.

Impurity Radiation
^^^^^^^^^^^^^^^^^^

The following models are available:

* Set the impurity radiation to be a constant fraction of the total external input power.

* Polynomial fits to ADAS data from `Mavrin, 2018. <https://doi.org/10.1080/10420150.2018.1462361>`_
  Provides radiative cooling rates for the following impurity species:
    - Helium
    - Lithium
    - Beryllium
    - Carbon
    - Nitrogen
    - Oxygen
    - Neon
    - Argon
    - Krypton
    - Xenon
    - Tungsten
  These cooling curves are multiplied by the electron density and impurity densities to obtain the impurity radiation power density.
  The valid temperature range of the fit is [0.1-100] keV.

Ion Cyclotron Resonance Heating
-------------------------------

Presently this source is implemented for a SPARC specific ICRH scenario.

A core Ion Cyclotron Range of Frequencies (ICRF) heating surrogate model trained
on TORIC ICRH spectrum solver simulation
https://meetings.aps.org/Meeting/DPP24/Session/NP12.106 is used to provide power
profiles for Helium-3, Tritium (via its second harmonic) and electrons.

A "Stix distribution" [Stix, Nuc. Fus. 1975] is used to model the non-thermal
Helium-3 distribution based on an analytic solution to the Fokker-Planck
equation to estimate the birth energy of Helium-3.

TORAX partitions the Helium-3 power between ions and electrons using the
parameterized model of Mikkelsen, as for Fusion Power.

It is assumed that all tritium heating goes to ions and all electron heating
goes to electrons.

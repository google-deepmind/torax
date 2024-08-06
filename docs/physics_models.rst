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

Presently, TORAX only accommodates a single main ion species,configured with its
atomic mass number (:math:`A_i`) and charge state (:math:`Z_i`). The plasma effective
charge, :math:`Z_\textit{eff}`, is assumed to be radially flat and is also
user-configurable. A single impurity with charge state :math:`Z_\textit{imp}` is
specified to accommodate :math:`Z_\textit{eff} > 1`. The main ion density dilution
is then calculated as follows:

.. math::

  n_i=(Z_\textit{imp}-Z_\textit{eff})/(Z_\textit{imp}-1)n_e

Initial conditions for the evolving profiles :math:`T_i`, :math:`T_e`, :math:`n_e`,
and :math:`\psi` are user-configurable. For :math:`T_{i,e}`, both the initial core
(:math:`r=0`) value and the boundary condition at :math:`\hat{\rho}=1` are provided.
Initial :math:`T_{i,e}` profiles are then a linear interpolation between these values.

For :math:`n_e`, the user specifies a line-averaged density, in either absolute units
or scaled to the Greenwald density :math:`n_\mathrm{GW}=\frac{I_p}{\pi a^2}~[10^{20} m^{-3}]`,
and the initial peaking factor. The initial :math:`n_e` profile, including the edge boundary
condition, is then a linear function scaled to match the desired line-averaged density and peaking.

If any of the :math:`T_{i,e}`, :math:`n_e` equations are set to be non-evolving (i.e., not evolved by the PDE stepper),
then time-dependent prescribed profiles are user-configurable.

For the poloidal flux :math:`\psi(\hat{\rho})`, the user specifies if the initial condition
is based on a prescribed current profile, :math:`j_\mathrm{tor}=j_0(1-\hat{\rho}^2)^\nu` (with :math:`j_0`
scaled to match :math:`I_p`, and :math:`\nu` is user-configurable), or from the :math:`\psi` provided
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
      \chi_{GB} \text{C} (\frac{R}{L_{Ti}} - \frac{R}{L_{Ti,\textit{crit}}})^{\alpha} & \text{if } \frac{R}{L_{Ti}} \ge \frac{R}{L_{Ti,\textit{crit}}} \\
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
    The stiffness factor :math:`C` and the exponent :math:`\alpha` are user-configurable model parameters.

    Regarding additional transport coefficient outputs, the electron heat conductivity, :math:`\chi_e`,
    and particle diffusivity, :math:`D_e`, are scaled to :math:`\chi_i` using user-configurable model parameters.
    The particle convection velocity, :math:`V_e`, is user-defined.

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
not need to be JAX-compatible, since explicit sources are an input into the PDE stepper,
and do not require JIT compilation. Conversely, implicit treatment can be important for accurately
resolving the impact of fast-evolving source terms.

All sources can optionally be set to zero, prescribed with non-physics-based formulas
(currently Gaussian or exponential) with user-configurable time-dependent parameters like
amplitude, width, and location, or calculated with a dedicated physics-based model. Not
all sources currently have a model implementation. However, the code modular structure
facilitates easy coupling of additional source models in future work. Specifics of source models
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
alpha particle slowing down model of Mikkelsen, which neglects the slowing down time itself.

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
While auxiliary heating such as neutral beam injection (NBI), ion cyclotron resonance heating (ICRH), etc,
and their associated non-inductive current drives, can all be prescribed with formulas, presently no
dedicated physics models are available within TORAX. Future work envisages incorporating more
sophisticated physics-based models or ML-surrogate models, enhancing the fidelity of the simulation.
For explicit sources, these can also come from external codes (not necessarily JAX compatible) coupled to TORAX in larger workflows.

Presently, a built-in non-physics-based Gaussian formulation of a generic ion and electron heat
source is available in TORAX, with user configurable location, Gaussian width, and fractional heating of ions and electrons.

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

Future work envisages coupling physics-based models and/or ML-surrogates.

Radiation
---------
Currently, TORAX only has a dedicated model for Bremsstrahlung. Models for cyclotron radiation,
recombination, and line radiation are still left for future work.

Bremsstrahlung
^^^^^^^^^^^^^^

Uses the model from Wesson, John, and David J. Campbell. Tokamaks. Vol. 149.
An optional correction for relativistic effects from Stott PPCF 2005 can be enabled with the flag "use_relativistic_correction".

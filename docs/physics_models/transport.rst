.. include:: ../links.rst

.. _physics_transport:

Transport models
################

Turbulent transport determines the values of the transport coefficients
(:math:`\chi_i`, :math:`\chi_e`, :math:`D_e`, :math:`V_e`) in the
:ref:`equations`, and is a key ingredient in core transport simulations.
Theory-based turbulent transport models provide the largest source of
nonlinearity in the PDE system. TORAX offers several transport models:

- **Prescribed:** This model allows setting transport coefficients to
  prescribed values (which can be constant or time/space-varying). While not
  physics-model-based, it is useful for testing, verification, and
  prescribing fixed inner or outer transport where models can be unreliable
  (e.g., deep core / sawtooth region or edge / pedestal).

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
    \frac{(T_i k_B)^{3/2}}{R_\mathrm{major}}

  and the |guo-romanelli| ion-temperature-gradient (ITG) mode critical
  gradient formula.

  .. math::

    R/L_{Ti,crit} = \frac{4}{3}(1 + T_i/T_e)(1 + 2|\hat{s}|/q)

  where :math:`\chi_\textit{min}` is a user-configurable minimum allowed
  :math:`\chi`, :math:`L_{Ti}\equiv-\frac{T_i}{\nabla T_i}` is the ion
  temperature gradient length, :math:`A_i` is the main ion atomic mass number,
  :math:`m_p` the proton mass, :math:`e` the electron charge, :math:`B_0` the
  vacuum toroidal magnetic field at the major radius :math:`R_\mathrm{major}`.
  The stiffness factor :math:`C` and the exponent :math:`\alpha` are
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
  vacuum toroidal magnetic field at :math:`R_\mathrm{major}`, :math:`n_e` is
  the electron density, :math:`\rho_{\text{tor}}` is the (unnormalized)
  toroidal flux coordinate, :math:`p_e` is the electron pressure, and
  :math:`T_e` is the electron temperature.

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
  ``use_fixed_point`` solver option, since ``newton_raphson`` requires
  autodiff which is not supported for QuaLiKiz. Running with QuaLiKiz is not a
  typical workflow due to its computational expense (2-3 orders of
  magnitude slower than with QLKNN). Its use-cases are:

  1. Evaluating ML-surrogates against their ground truth, i.e. for QLKNN.

  2. Carrying out higher-fidelity simulations to verify faster simulations
     carried out with ML-surrogates. It is a major advantage to be able to
     use TORAX as the same framework for both ML-surrogate and high-fidelity
     simulations.

In TORAX, individual transport models (such as QLKNN, Bohm-GyroBohm, Prescribed,
TGLF, TGLFNN-ukaea, QuaLiKiz, and CGM) represent specific physical transport
mechanisms and are implemented as component models (``ComponentTransportModel``,
configured via ``ComponentTransportConfigBase``). To simulate a full plasma discharge,
TORAX combines one or more core and pedestal component models within a top-level
``TransportModel``.

Numerical conditioning and post-processing of transport coefficients—such as
minimum/maximum clipping (``chi_min``, ``chi_max``, ``D_e_min/max``,
``V_e_min/max``) and spatial Gaussian smoothing (``smoothing_width``,
``pedestal_smoothing_width``, ``smoothing_zones``)—are configured on the top-level transport model rather than
on individual component models. Furthermore, to combine transport models or
prescribe transport coefficients within specific radial zones (e.g. inner core
or outer edge regions), users can configure dictionaries of named component
models (``core_transport_models`` and ``pedestal_transport_models``) with
explicit radial bounds (``rho_min``, ``rho_max``) and merge modes
(``merge_mode: 'add'`` or ``'overwrite'``).

An edge-transport-barrier, or pedestal, is set up in TORAX by a pedestal model
composed of three sub-models. A structure model sets the H-mode target values of
:math:`T_e`, :math:`T_i` and :math:`n_e` at a user-configurable pedestal-top
location (pedestal width). Two built-in variants are available, one setting the
pedestal pressure and temperature ratio, and the other setting the pedestal
temperatures directly; external models such as EPED surrogates can also be
registered (see :ref:`model-integration`). A formation model determines when
L-H and H-L transitions occur, either from a prescribed schedule or from a
:math:`P_{LH}` power-threshold scaling law with hysteresis. A saturation model
increases pedestal transport once the pedestal-top values exceed their targets.

The pedestal is imposed in one of two modes. In ``INTERNAL_BOUNDARY_CONDITION``
mode, internal boundary conditions set the pedestal-top values directly (either
at a single grid point or as an mtanh profile across the pedestal region), with
an optional linear ramp during L-H and H-L transitions. In
``ADAPTIVE_TRANSPORT`` mode, the turbulent transport coefficients in the
pedestal region are scaled down by the formation model and back up by the
saturation model, allowing the pedestal to evolve self-consistently.

In L-mode discharges, edge kinetic profiles (:math:`T_i, T_e, n_e`) in the edge
region :math:`\hat{\rho} \ge \hat{\rho}_{\text{edge}}` can optionally be
constrained via an internal boundary condition parameterized by a prescribed
normalized poloidal beta gradient :math:`\beta_{pol}' \equiv -\partial
\beta_{pol,\text{local}} / \partial \psi_N` (see e.g. |labombard2008|):

.. math::

  \beta_{pol,\text{local}}(\psi) =
  \frac{P_{\text{total}}(\psi)}{\langle B_p^2 \rangle / (2\mu_0)}

  P_{\text{total}}(\psi_N) = \frac{\langle B_p^2 \rangle}{2\mu_0}
  [\beta_{pol,\text{sep}} + \beta_{pol}' (1 - \psi_N)]

Combined with a prescribed edge electron density :math:`n_{e,\text{edge}}` and
temperature ratio :math:`T_i / T_e`, this model determines the edge temperature
profiles at each solver iteration.

In the TORAX roadmap, coupling to additional transport models is envisaged,
including to additional semi-empirical models and H-mode confinement
scaling law adaptive models, as well as more ML-surrogates of theory-based
models, both for core turbulence and pedestal predictions.

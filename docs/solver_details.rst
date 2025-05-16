.. include:: links.rst

.. _solver_details:

TORAX solver details
####################

TORAX employs a finite volume method (FVM) to discretize the governing PDEs in
space, and the theta-method to discretize in time. These methods are described
in further detail below.

Spatial discretization scheme
=============================
The TORAX JAX 1D FVM library is significantly influenced by
`FiPy <https://www.ctcms.nist.gov/fipy/>`_. See :ref:`fvm` for details on the
library API. This section summarizes 1D FVM numerics in general.

The 1D spatial domain, :math:`0 \leq \hat{\rho} \leq 1`, is divided into a
uniform grid of :math:`N` cells, each with a width of
:math:`d \hat{\rho} = 1/N`.  The cell centers are denoted by
:math:`\hat{\rho}_i`, where :math:`0 = 1, 2,..., N-1`, and the :math:`N+1` cell
faces are located at :math:`\hat{\rho}_{i\pm1/2}`. Both :math:`\hat{\rho}=0` and
:math:`\hat{\rho}=1` are on the face grid.

For a generic conservation law of the form:

.. math::

  \frac{\partial x}{\partial t} + \nabla \cdot \mathbf{\Gamma} = S

where :math:`x` is a conserved quantity, :math:`\mathbf{\Gamma}` is the flux,
and :math:`S` is a source term, the FVM approach involves integrating the PDE
over a control volume (named a "cell") and applying the divergence theorem to
convert volume integrals to surface integrals. For 1D systems, following
dividing by the cell volume, the finite volume method reduces to a special case
of finite differences:

.. math::

  \frac{\partial }{\partial t}(x_i) + \frac{1}{d \hat{\rho}}({\Gamma}_{i+1/2}
  - {\Gamma}_{i-1/2})  = S_i

where: :math:`x_i` is the cell-averaged value of :math:`x` in cell :math:`i`,
:math:`\Gamma_{i+1/2}` is the flux at face :math:`i+1/2`, and :math:`S_i` is the
cell-averaged source term in cell :math:`i`.

In general, the fluxes in TORAX are decomposed as

.. math::
  \Gamma = -D\frac{\partial x}{\partial \hat{\rho}} + Vx

where :math:`D` is a diffusion coefficient and :math:`V` now denotes a
convection coefficient, leading to:

.. math::

  \begin{aligned}
  \Gamma_{i+1/2} &= -D_{i+1/2}\frac{x_{i+1} - x_{i}}{d\hat{\rho}} +
   V_{i+1/2}x_{i+1/2} \\
  \Gamma_{i-1/2} &= -D_{i-1/2}\frac{x_{i} - x_{i-1}}{d\hat{\rho}} +
   V_{i-1/2}x_{i-1/2}
  \end{aligned}

The diffusion and convection coefficients are thus calculated on the face grid.
The value of :math:`x` on the face grid is approximated by implementing a
power-law scheme for Péclet weighting, which smoothly transitions between
central differencing and upwinding, as follows:

.. math::

  \begin{aligned}
  x_{i+1/2} &= \alpha_{i+1/2}x_i + (1 - \alpha_{i+1/2}) x_{i+1} \\
  x_{i-1/2} &= \alpha_{i-1/2}x_i + (1 - \alpha_{i-1/2}) x_{i-1} \\
  \end{aligned}

where the :math:`\alpha` weighting factor depends on the Péclet number,
defined as:

.. math::

  Pe = \frac{V d \hat{\rho}}{D}

where :math:`V` is convection and :math:`D` is diffusion. The power-law scheme
is as follows:

.. math::

  \alpha = \begin{cases}
  \frac{Pe - 1}{Pe}  & \text{if } Pe > 10, \\
  \frac{(Pe - 1) + (1 - Pe/10)^5}{Pe} & \text{if } 0 < Pe < 10, \\
  \frac{(1 + Pe/10)^5 - 1}{Pe} & \text{if } -10 < Pe < 0, \\
  -\frac{1}{Pe} & \text{if } Pe < -10.
  \end{cases}

The Péclet number quantifies the relative strength of convection and diffusion.
If the Péclet number is small and diffusion dominates, then the weighting scheme
converges to central differencing. If the absolute value of the Péclet number is
large, and convection dominates, then the scheme converges to upwinding.

Boundary conditions are taken into account by introducing ghost cells
:math:`x_{N}` and :math:`x_{-1}` whose values are determined by assuming linear
extrapolation through the edge cells and the face boundary conditions (for
Dirichlet boundary conditions), or by directly satisfying the derivative
(Neumann) boundary conditions.

The above equations, when combined, define the elements of the discretization
matrix and boundary condition vectors for the PDE diffusion term.

Time discretization and solver options
======================================

TORAX uses the theta method for time discretization, and employs several options
for solving the discretized PDE system. These are described below.

Theta method
------------

The theta method is a weighted average between the explicit and implicit Euler
methods. For a generic ODE of the form:

.. math::

  \frac{dx}{dt} = F(x, t)

where x is the state vector, the theta method approximates the solution at time
:math:`t + \Delta t` as:

.. math::
  :label: theta

  x_{t + \Delta t} - x_t = \Delta t \big[ \theta F(x_{t + \Delta t}, t +
  \Delta t) + (1 - \theta) F(x_t, t)\big]

where :math:`\theta` is a user-selected weighting parameter in the range
:math:`[0, 1]`. Different values of :math:`\theta` correspond to well-known
solution methods: explicit Euler (:math:`\theta = 0`), Crank-Nicolson
(:math:`\theta = 0.5`), and implicit Euler (:math:`\theta = 1`), which is
unconditionally stable.

TORAX equation composition
--------------------------

Upon inspection of the :ref:`equations`, we generalize equation :eq:`theta` and
write the TORAX state evolution equation as:

.. math::
  :label: state_evolution

  \begin{aligned}
  & \mathbf{\tilde{T}}(x_{t + \Delta t}, u_{t + \Delta t})\odot
  \mathbf{x}_{t + \Delta t} - \mathbf{\tilde{T}}(x_t, u_t)\odot\mathbf{x}_t =
  \\
  & \Delta t \big[ \theta \big( \mathbf{\bar{C}}(x_{t+\Delta t}, u_{t+\Delta t})
  \mathbf{x}_{t+\Delta t} + \mathbf{c}(x_{t+\Delta t}, u_{t+\Delta t}) \big) \\
  & \qquad + (1-\theta) \big( \mathbf{\bar{C}}(x_t, u_t)\mathbf{x}_t +
  \mathbf{c}(x_{t}, u_{t}) \big) \big]
  \end{aligned}

Starting from an initial condition :math:`\mathbf{x}_0`, equation
:eq:`state_evolution` solves for :math:`\mathbf{x}_{t+\Delta t}` at each
timestep. :math:`\mathbf{x}_t` is the evolving state vector at time :math:`t`,
including all variables being solved by the system, and is of length
:math:`\#N`, where :math:`\#` is the number of solved variables. For example,
consider a simulation with a gridsize of :math:`25` solving ion heat transport,
electron heat transport, and current diffusion. Then :math:`N=25`, :math:`\#=3`,
and :math:`\mathbf{x}_t` is comprised of :math:`T_i`, :math:`T_e`, and
:math:`\psi`, each with its own set of :math:`N` values, making a total vector
length of 75.

:math:`\mathbf{u}_t` corresponds to all known input parameters at time
:math:`t`. This includes boundary conditions, prescribed profiles
(e.g. :math:`n_e` in the example above), and input parameters such as heating
powers or locations.

:math:`\mathbf{\tilde{T}}` is the transient term (following
`FiPy <https://www.ctcms.nist.gov/fipy/>`_ nomenclature), where :math:`\odot`
signifies element-wise multiplication. For example, for the :math:`T_e`
equation, :math:`\mathbf{\tilde{T}}=\mathbf{n_e}`, which makes the system
nonlinear if :math:`\mathbf{n_e}` itself is an evolving variable.

:math:`\mathbf{\bar{C}}(x_t, u_t)` and
:math:`\mathbf{\bar{C}}(x_{t+\Delta t}, u_{t+\Delta t})` are the discretization
matrices, of size :math:`\#N\times\#N`. In general, depending on the physics
models used, :math:`\mathbf{\bar{C}}` depends on state variables
:math:`\mathbf{x}`, for example through state-variable dependencies of transport
coefficients :math:`\chi`, :math:`D`, :math:`V`, plasma conductivity, and
ion-electron heat exchange, making the system nonlinear due to the
:math:`x_{t+\Delta t}` dependence. :math:`\mathbf{c}` is a vector, containing
source terms and boundary condition terms.

Solver options
--------------

TORAX provides three solver options for solving the TORAX nonlinear evolution
system of equations, summarized next.

Linear solver
^^^^^^^^^^^^^

This solver addresses the nonlinearity of the PDE system with fixed-point
iteration, also known as the predictor-corrector method. For :math:`K`
iterations (user-configurable), an approximation for
:math:`\mathbf{x}_{t+\Delta t}` is obtained by solving the following equation
iteratively with :math:`k=1,2,..,K`:

.. math::

  \begin{aligned}
  & \mathbf{\tilde{T}}(x_{t + \Delta t}^{k-1}, u_{t + \Delta t})\odot
  \mathbf{x}_{t + \Delta t}^k - \mathbf{\tilde{T}}(x_t, u_t)\odot\mathbf{x}_t =
  \\
  & \Delta t \big[ \theta \big( \mathbf{\bar{C}}(x_{t+\Delta t}^{k-1},
  u_{t+\Delta t})\mathbf{x}_{t+\Delta t}^k + \mathbf{c}(x_{t+\Delta t}^{k-1},
  u_{t+\Delta t}) \big) \\
  & \qquad + (1-\theta) \big( \mathbf{\bar{C}}(x_t, u_t)\mathbf{x}_t +
  \mathbf{c}(x_{t}, u_{t}) \big) \big]
  \end{aligned}

and where :math:`\mathbf{x}_{t+\Delta t}^{0} = \mathbf{x}_t`.

By replacing :math:`\mathbf{x}_{t+\Delta t}` with
:math:`\mathbf{x}_{t+\Delta t}^{k-1}` within the coefficients
:math:`\mathbf{\tilde{T}}`, :math:`\mathbf{\bar{C}}` and :math:`\mathbf{c}`,
these coefficients become known at every iteration step, describing a `linear`
system of equations. :math:`\mathbf{x}_{t+\Delta t}^k` can then be solved using
standard linear algebra methods implemented in JAX.

To further enhance the stability of the linear solver, particularly in the
presence of stiff transport coefficients (e.g., when using the QLKNN turbulent
transport model, see :ref:`physics_models`), the |pereverzev-corrigan-method|
is implemented as an option. This method adds a large (user-configurable)
artificial diffusion term to the transport equations, balanced by a large inward
convection term such that zero extra transport is added at time :math:`t`. These
terms stabilize the solution, at the cost of accuracy over short transient
phenomena, demanding care in the choice of :math:`\Delta t` and the value of the
artificial diffusion term.

Newton-Raphson Solver
^^^^^^^^^^^^^^^^^^^^^

This solver solves the nonlinear PDE system, using a gradient-based iterative
Newton-Raphson root-finding method for finding the value of
:math:`\mathbf{x}_{t+\Delta t}` that renders the residual vector zero:

.. math::
  :label: residual

  \mathbf{R}(\mathbf{x}_{t+\Delta t},\mathbf{x}_t,\mathbf{u}_{t+\Delta t},
  \mathbf{u}_t, \theta, \Delta t) = 0

where :math:`\mathbf{R}` is the LHS-RHS of equation :eq:`state_evolution`.

Starting from an initial guess
:math:`\mathbf{x}_{t+\Delta t}=\mathbf{x}_{t+\Delta t}^0`, the Newton-Raphson
method linearizes equation :eq:`residual` about iteration
:math:`\mathbf{x}_{t+\Delta t}^k` and solves the linear system for a step
:math:`\delta\mathbf{x}`:

.. math::

  \mathbf{\bar{J}}(\mathbf{x}_{t+\Delta t}^k) \delta\mathbf{x} =
  -\mathbf{R}(\mathbf{x}_{t+\Delta t}^k)

where :math:`\mathbf{\bar{J}}` is the Jacobian of :math:`\mathbf{R}` with
respect to :math:`\mathbf{x}_{t+\Delta t}`. Crucially, JAX automatically
calculates :math:`\mathbf{\bar{J}}` using auto-differentiation.

With
:math:`\delta\mathbf{x} = \mathbf{x}_{t+\Delta t}^{k+1} - \mathbf{x}_{t+\Delta t}^{k}`,
:math:`\mathbf{x}_{t+\Delta t}^{k+1}` is solved using standard linear algebra
methods implemented in JAX such as LU decomposition. This process iterates until
the residual falls below a user-configurable tolerance
:math:`\varepsilon`,  i.e:
:math:`\| \mathbf{R}(\mathbf{x}_{t+\Delta t}^{k+1}) \|_2 < \varepsilon`, where
:math:`\|\cdot\|_2` is the vector two-norm.

Solver robustness is obtained with a combination of :math:`\delta \mathbf{x}`
line search and :math:`\Delta t` backtracking. :math:`\delta \mathbf{x}` line
search reduces the step size within a given Newton iteration step, while
:math:`\Delta t` backtracking reduces the overall time step and restarts the
entire Newton-Raphson solver for the present timestep, as follows:

  - If a Newton step leads to an increasing residual,
    i.e.
    :math:`\mathbf{R}(\mathbf{x}_{t+\Delta t}^{k+1}) > \mathbf{R}(\mathbf{x}_{t+\Delta t}^k)`,
    or if :math:`\mathbf{x}_{t+\Delta t}^{k+1}` is unphysical, e.g. negative
    temperature, then :math:`\delta \mathbf{x}` is reduced by a
    user-configurable factor, and the line-search checks are repeated. The total
    accumulative reduction factor in a Newton step is denoted :math:`\tau`.

  - If during the line-search phase, :math:`\tau` becomes too low, as determined
    by a user-configurable variable, then the solve is abandoned and
    :math:`\Delta t` backtracking is invoked. A new solve attempt is made at a
    reduced :math:`\Delta t`, reduced by a user-configurable factor, which
    results in a less nonlinear system.

For the initial guess :math:`\mathbf{x}_{t+\Delta t}^0`, two options are
available. The user can start from :math:`\mathbf{x}_t`, or use the result of
the predictor-corrector linear solver as a warm-start.

Optimizer solver
^^^^^^^^^^^^^^^^

An alternative nonlinear solver using the JAX-compatible
`jaxopt library <https://github.com/google/jaxopt>`_ is also available. This
method recasts the PDE residual as a loss function, which is minimized using an
iterative optimization algorithm. Similar to the Newton-Raphson solver, adaptive
timestepping is implemented, where the timestep is reduced if the loss remains
above a tolerance at exit. While offering flexibility with different
optimization algorithms, this option is relatively untested for TORAX to date.

Timestep (:math:`\Delta t`) calculation
=======================================

TORAX provides two methods for calculating the timestep :math:`\Delta t`,
as follows.

  - Fixed :math:`\Delta t`: This method uses a user-configurable constant
    timestep throughout the simulation. If a nonlinear solver is employed, and
    adaptive timestepping is enabled, then in practice, some steps may have a
    lower :math:`\Delta t` following backtracking.

  - Adaptive :math:`\Delta t`:  This method adapts :math:`\Delta t` based on the
    maximum heat conductivity :math:`\chi_{\max}=\max(\chi_i, \chi_e)`.
    :math:`\Delta t` is a multiple of a base timestep inspired by the explicit
    stability limit for parabolic PDEs:

  .. math::

    \Delta t_{ \mathrm{base}}=\frac{(d\hat{\rho})^2}{2\chi_{\max}}

  where
  :math:`\Delta t = c_{ \mathrm{mult}}^{dt} \Delta t_{ \mathrm{base}}`. :math:`c_{ \mathrm{mult}}^{dt}`
  is a user-configurable prefactor.  In practice,
  :math:`c_{ \mathrm{mult}}^{dt}` can be significantly larger than unity for
  implicit solution methods.

The adaptive timestep method protects against traversing through fast transients
in the simulation, by enforcing :math:`\Delta t \propto \chi`.

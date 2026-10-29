.. include:: ../links.rst

.. _profile_conditions:

profile_conditions
##################

Configures boundary conditions, initial conditions, and (optionally) prescribed
time-dependence of temperature, density, and current.

``I_p`` (**time-varying-scalar** [default = None])
  Total plasma current in A. Required when ``I_p_source = 'profile_conditions'``;
  ignored otherwise. If ``I_p_source = 'geometry'``, then the total plasma
  current is read from the geometry data instead. Boundary condition for the
  :math:`\psi` equation if ``use_v_loop_lcfs_boundary_condition`` is False. If
  ``use_v_loop_lcfs_boundary_condition`` is True, only used as an initial
  condition.

``I_p_source`` (str [default = ``'profile_conditions'``])
  Source for the total plasma current :math:`I_p`.
  Options are ``'profile_conditions'`` (uses the ``I_p`` scalar attribute) or
  ``'geometry'`` (reads :math:`I_p` from numerical geometry; not supported
  for ``'circular'`` geometry).

``use_v_loop_lcfs_boundary_condition`` (bool = False)
  Boundary condition at LCFS for Vloop ( = dpsi_lcfs/dt ). If
  ``use_v_loop_lcfs_boundary_condition`` is True, then the specified Vloop at
  the LCFS is used to construct the boundary condition for the psi equation;
  otherwise, I_p is used to construct the boundary condition.

``v_loop_lcfs`` (**time-varying-scalar** [default = 0.0])
  Boundary condition at LCFS for Vloop ( = dpsi_lcfs/dt ). This sets the psi at
  the next timestep. This is ignored if ``use_v_loop_lcfs_boundary_condition``
  is False.

``T_i_right_bc`` (**time-varying-scalar** [default = None])
  Temperature boundary condition at r=a_minor. If this is ``None`` the boundary
  condition will instead be taken from ``T_i`` at :math:`\hat{\rho}=1`.
  In units of :math:`keV`.

``T_e_right_bc`` (**time-varying-scalar** [default = None])
  Temperature boundary condition at r=a_minor. If this is ``None`` the boundary
  condition will instead be taken from ``T_e`` at :math:`\hat{\rho}=1`.
  In units of :math:`keV`.

``T_i`` (**time-varying-array** [default = {0: {0: 15.0, 1: 1.0}}])
  If ``numerics.evolve_ion_heat`` is True, then the initial value of ion
  temperature is taken from here. If ``numerics.evolve_ion_heat`` is False,
  then the prescribed ion temperature for all times is taken from here. In units
  of :math:`keV`.

``T_e`` (**time-varying-array** [default = {0: {0: 15.0, 1: 1.0}}])
  If ``numerics.evolve_electron_heat`` is True, then the initial value of
  electron temperature is taken from here. If ``numerics.evolve_electron_heat``
  is False, then the prescribed electron temperature for all times is taken from
  here. In units of :math:`keV`.

``psi`` (**time-varying-array** | None [default = None])
  Poloidal flux :math:`\psi` profile in units of :math:`\text{Wb}`. Used as the
  initial ``psi`` when ``initial_psi_source = 'profile_conditions'`` (required
  in that mode unless restarting from a file). If ``numerics.evolve_current`` is
  False and ``psi`` is provided, then the prescribed ``psi`` for
  :math:`t > t_{\text{initial}}` is taken from here. If ``psi`` is provided when
  ``initial_psi_source`` is not ``'profile_conditions'`` and
  ``numerics.evolve_current`` is True, it is ignored and a warning is logged.

``psidot`` (**time-varying-array** | None [default = None])
  Prescribed values for the time derivative of poloidal flux (loop voltage).
  If provided, and if ``numerics.evolve_current`` is False, this prescribed
  ``psidot`` will be used instead of the internally calculated one. The
  motivation for this feature is that sometimes the initial ``psi`` condition
  leads to unphysical transient ``psidot`` and thus transiently unphysical
  (e.g. too high) ohmic power. For such cases, it is useful to override a static
  non-physical ``psidot`` with a more physical value, e.g., one obtained from
  another similar simulation with current evolution enabled.

``n_e`` (**time-varying-array** [default = {0: {0: 1.2e20, 1: 0.8e20}}])
  If ``numerics.evolve_density`` is True, then the initial value of electron
  density is taken from here. If ``numerics.evolve_density`` is False, then the
  prescribed electron density for all times is taken from here. In units of
  :math:`m^{-3}`.

``normalize_n_e_to_nbar`` (bool = False)
  Whether to renormalize the density profile to have the desired line averaged
  density ``nbar``.

``nbar`` (**time-varying-scalar** [default = 0.85e20])
  Line averaged density. In units of :math:`m^{-3}` if
  ``n_e_nbar_is_fGW = False``. In Greenwald fraction if
  ``n_e_nbar_is_fGW = True``. :math:`n_{GW} = I_p/(\pi a^2)` with a in m,
  :math:`n_{GW}` in :math:`10^{20} m^{-3}`, I_p in MA.

``n_e_nbar_is_fGW`` (bool = False)
  Toggle units of ``nbar``.

``n_e_right_bc`` (**time-varying-scalar** | None [default = None])
  Density boundary condition for r=a_minor. In units of m^-3 if
  ``n_e_right_bc_is_fGW = False``. In Greenwald fraction if
  ``n_e_right_bc_is_fGW = True``. If ``n_e_right_bc`` is ``None`` then the
  boundary condition will instead be taken from ``n_e`` at :math:`\hat{\rho}=1`.

``n_e_right_bc_is_fGW`` (bool [default = False])
  Toggle units of ``n_e_right_bc``.

``n_e_right_bc_mode`` (str [default = "prescribed"])
  Mode for setting the electron density boundary condition at :math:`r = a`.
  Options:

  - ``"prescribed"``: Use the value of ``n_e_right_bc`` directly (or the edge
    value of ``n_e`` if ``n_e_right_bc`` is ``None``).
  - ``"density_fraction"``: Compute ``n_e_right_bc`` from the evolved electron
    density profile at
    :math:`n_e(\hat{\rho}_{\text{ref}}, t) \times \text{multiplier}`, where
    :math:`t` refers to the beginning of each time step,
    :math:`\hat{\rho}_{\text{ref}}` is set by ``n_e_right_bc_reference_rho`` and
    the multiplier by ``n_e_right_bc_multiplier``. At initialization (before any
    evolved profile exists), the prescribed ``n_e`` profile is used as a
    fallback. When using this mode, ``n_e_right_bc`` and ``n_e_right_bc_is_fGW``
    are ignored.

``n_e_right_bc_reference_rho`` (float | None [default = None])
  Reference normalized toroidal flux coordinate :math:`\hat{\rho}` at which to
  evaluate the electron density when using the ``"density_fraction"`` boundary
  condition mode. Required when ``n_e_right_bc_mode = "density_fraction"``.

``n_e_right_bc_multiplier`` (float | None [default = None])
  Multiplicative factor applied to the electron density at the reference
  :math:`\hat{\rho}` when using the ``"density_fraction"`` boundary condition
  mode. Required when ``n_e_right_bc_mode = "density_fraction"``.

``current_profile_nu`` (float [default = 1.0])
  Peaking factor of initial current, either total or "Ohmic":
  :math:`j = j_0(1 - r^2/a^2)^\nu` where :math:`\nu` is ``current_profile_nu``. Used if
  ``initial_psi_source`` is ``'j'``. In that case, then this sets the
  peaking factor of either the total or Ohmic initial current profile, depending
  on the ``initial_j_is_total_current`` flag.

``initial_j_is_total_current`` (bool [default = False])
  Toggle if the initial current formula set by ``current_profile_nu`` is the
  total current, or the Ohmic current. If Ohmic current, then the magnitude of
  the Ohmic current is set such that the initial total non-inductive current +
  total Ohmic current equals ``I_p``

``initial_psi_source`` (str)
  Required method for determining initial poloidal flux :math:`\psi`.
  Options are ``'profile_conditions'`` (uses the ``psi`` array attribute, which
  must be provided unless restarting from a file), ``'geometry'`` (uses
  :math:`\psi` from numerical geometry; not supported for ``'circular'``
  geometry), or ``'j'`` (calculates from the ``current_profile_nu`` current
  formula).

``toroidal_angular_velocity`` (**time-varying-array** | None [default = None])
  Toroidal angular velocity profile. If not provided, the velocity will be set
  to zero.

``toroidal_angular_velocity_right_bc`` (**time-varying-scalar** | None [default = None])
  Toroidal angular velocity boundary condition for r=a_minor. If this is
  ``None`` the boundary condition will instead be taken from
  ``toroidal_angular_velocity`` at :math:`\hat{\rho}=1`. If ``toroidal_angular_velocity`` is
  also ``None``, then the boundary condition will be set to zero.

``internal_boundary_conditions`` (dict [default = ``{'model_name': 'no_ibc'}``])
  Internal boundary conditions for :math:`T_i`, :math:`T_e`, and :math:`n_e`.
  These conditions are enforced via matrix row replacement in the PDE solver.
  Defaults to ``'no_ibc'`` (no internal boundary conditions active). Active
  models are selected via the ``'model_name'`` key:

  1. ``'model_name': 'no_ibc'`` (default):
     No internal boundary conditions are active.

  2. ``'model_name': 'prescribed'``:
     Explicitly prescribe time-varying values for any subset of ``T_i``,
     ``T_e``, and ``n_e`` using **sparse time-varying-array** types. Any values
     of 0.0 will be treated as no IBC in that location. It is therefore possible
     to only have an IBC for a subset of the channels (e.g. only setting
     ``T_e``). Not including a channel in the config means no IBC for that
     channel. Values are specified as ``{time: {rho_norm: value, ...}, ...}``.
     For example:

     .. code-block:: python

       'profile_conditions': {
           'internal_boundary_conditions': {
               'model_name': 'prescribed',
               'T_e': {
                   0.0: {0.85: 1.0},
                   1.0: {0.85: 1.5}
               }
           }
       }

     This sets the electron temperature to 1.0 keV at :math:`\hat{\rho}=0.85`
     at t=0, and 1.5 keV at :math:`\hat{\rho}=0.85` at t=1, with linear
     interpolation in time in between.

     Time-varying values can also be specified for a **range** of
     :math:`\hat{\rho}` values:

     .. code-block:: python

       'profile_conditions': {
           'internal_boundary_conditions': {
               'model_name': 'prescribed',
               'T_i': {
                   0.0: {
                       (0.85, 1.0): {0.85: 1.5, 1.0: 1.0}
                   }
               }
           }
       }

     Note that radial locations (:math:`\hat{\rho}` keys) must be the same at
     all times.

  3. ``'model_name': 'beta_poloidal_prime'``:
     Dynamically computes L-mode edge kinetic profiles (:math:`T_i, T_e, n_e`)
     in the edge region :math:`\hat{\rho} \ge \hat{\rho}_{\text{edge}}` from a
     prescribed gradient of the local poloidal beta with respect to normalized
     poloidal flux, :math:`\beta_{pol}' \equiv -\partial
     \beta_{pol,\text{local}} / \partial \psi_N`, motivated by empirical
     observations of critical edge pressure gradients in L-mode plasmas (see
     e.g. |labombard2008|). Parameters:

     - ``mode`` (str [default = ``'constant'``]): Controls how the effective
       edge poloidal beta gradient is determined:

       - ``'constant'``: Uses ``beta_poloidal_prime`` directly.
       - ``'power_dependent'``: Scales the effective poloidal beta gradient
         smoothly with the net heating power crossing the separatrix
         :math:`P_{\text{SOL}}` according to
         :math:`\beta_{pol,\text{eff}}' = \beta_{pol,\text{min}}' +
         (\beta_{pol}' - \beta_{pol,\text{min}}') y(x)`, where
         :math:`x = \max(P_{\text{SOL}}, 0) / P_{\text{SOL,scaling}}`,
         :math:`z_1 = (x^2 + 0.3 x) / 1.3`, and
         :math:`y = (z_1^{\alpha} + 1)^{1/\alpha}` with :math:`\alpha = -15`.
     - ``rho_norm_edge`` (**time-varying-scalar**): Edge normalized toroidal
       flux coordinate :math:`\hat{\rho}_{\text{edge}}` bounding the constrained
       edge region.
     - ``n_e_edge`` (**time-varying-scalar** | None [default = None]):
       Prescribed electron density at the edge boundary
       :math:`\hat{\rho}_{\text{edge}}` [:math:`\text{m}^{-3}` if
       ``n_e_is_fGW = False``, Greenwald fraction if ``n_e_is_fGW = True``].
       Mutually exclusive with ``n_e_edge_multiplier`` (exactly one must be
       provided).
     - ``n_e_edge_multiplier`` (**time-varying-scalar** | None [default =
       None]): Multiplier applied to the right boundary condition density
       (``n_e_right_bc``) to set the electron density at
       :math:`\hat{\rho}_{\text{edge}}` [dimensionless]. Mutually exclusive
       with ``n_e_edge`` (exactly one must be provided).
     - ``n_e_is_fGW`` (bool [default = False]): If True, ``n_e_edge`` is
       interpreted as a Greenwald fraction (dimensionless) instead of absolute
       density (:math:`\text{m}^{-3}`).
     - ``beta_poloidal_prime`` (**time-varying-scalar**): Prescribed normalized
       poloidal beta gradient :math:`\beta_{pol}' > 0` [dimensionless] (serves
       as the high-power asymptotic value when ``mode = 'power_dependent'``).
     - ``beta_poloidal_prime_min`` (**time-varying-scalar** [default = 0.2]):
       Minimum normalized poloidal beta gradient :math:`\beta_{pol,\text{min}}'
       \ge 0` at :math:`P_{\text{SOL}} \le 0` when
       ``mode = 'power_dependent'`` [dimensionless].
     - ``P_SOL_scaling`` (**time-varying-scalar** | None [default = None]):
       Characteristic power crossing the separatrix :math:`P_{\text{SOL,scaling}}
       > 0` [W] for the power transition when ``mode = 'power_dependent'``
       (required when ``mode = 'power_dependent'``). ``P_SOL_scaling`` sets the
       knee of the power scaling curve, so that a scaled :math:`P_{\text{SOL}}`
       of 1 is approximately where ``beta_poloidal_prime`` saturates.
     - ``Ti_Te_ratio`` (**time-varying-scalar**): Prescribed ratio
       :math:`T_i / T_e` in the edge region [dimensionless].

     Example configuration:

     .. code-block:: python

       'profile_conditions': {
           'internal_boundary_conditions': {
               'model_name': 'beta_poloidal_prime',
               'rho_norm_edge': 0.8,
               'n_e_edge': 0.2e20,
               'beta_poloidal_prime': 1.5,
               'Ti_Te_ratio': 1.0,
           }
       }

``fast_ions`` (list[dict] | None [default = None])
  Prescribed fast ion density and temperature profiles. Each entry prescribes
  a fast ion species for a specific source, overriding any model-computed fast
  ion values for that (source, species) pair. Requires
  ``numerics.enable_fast_ions`` to be ``True``.

  Each entry in the list is a dict with the following keys:

  * ``source`` (str): The source name (e.g., ``'icrh'``).
  * ``species`` (str): The species name (must be one of ``'H'``, ``'D'``,
    ``'T'``, ``'He3'``, ``'He4'``).
  * ``n`` (**time-varying-array**): Prescribed density profile [m\ :sup:`-3`].
  * ``n_right_bc`` (**time-varying-scalar**): Right boundary condition for
    density [m\ :sup:`-3`].
  * ``T`` (**time-varying-array**): Prescribed temperature profile [keV].
  * ``T_right_bc`` (**time-varying-scalar**): Right boundary condition for
    temperature [keV].

  Example:

  .. code-block:: python

    'profile_conditions': {
        'fast_ions': [
            {
                'source': 'icrh',
                'species': 'He3',
                'n': {0.0: 1e19},
                'n_right_bc': 0.0,
                'T': {0.0: 100.0},
                'T_right_bc': 0.0,
            },
        ],
    }

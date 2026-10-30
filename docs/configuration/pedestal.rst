.. _pedestal_config:

pedestal
########

The ``pedestal`` dictionary configures the pedestal model, which is composed of
three sub-models:

* ``structure_model``: computes the H-mode target pedestal width
  (``rho_norm_ped_top``) and pedestal-top electron density, ion temperature,
  and electron temperature.
* ``formation_model``: determines when L-H and H-L transitions occur (and in
  ``'adaptive_transport'`` mode, the transport decrease multiplier).
* ``saturation_model``: determines how transport scales in the pedestal region
  relative to the H-mode target values in ``'adaptive_transport'`` mode.

``mode`` (str [default = 'internal_boundary_condition'])
  Defines how the pedestal is generated. Options:

  * ``'internal_boundary_condition'``: Sets the pedestal top value by directly
    modifying the state equations to set Dirichlet internal boundary
    conditions at a cell grid point corresponding to the pedestal top, or for a
    profile (see ``pedestal_profile_form``). This is the default mode.
  * ``'adaptive_transport'``: Sets the pedestal by modifying the transport
    coefficients in the pedestal region, allowing the pedestal to
    self-consistently evolve. Transport coefficients are scaled to allow the
    temperature and density to evolve towards the prescribed pedestal values.

``transition_time_width`` (**time-varying-scalar** [default = 0.0])
  Duration of the L-H or H-L transition ramp in seconds for
  ``'internal_boundary_condition'`` mode. During a transition, the confinement
  mode enters ``TRANSITIONING_TO_H_MODE`` or ``TRANSITIONING_TO_L_MODE`` and
  pedestal values are linearly interpolated between the captured L-mode baseline
  values and the H-mode target values (e.g. ``T_i_ped(t)``, ``T_e_ped(t)``,
  ``n_e_ped(t)``) over this time window. Must be non-negative. Defaults to
  ``0.0`` (immediate step transition between ``l_mode`` and ``h_mode``) for
  ``'prescribed'`` formation, and must be explicitly specified when using a
  power-scaling formation model (``'martin_scaling'`` or ``'delabie_scaling'``)
  in ``'internal_boundary_condition'`` mode. Note that if
  ``transition_time_width > 0.0`` is used with time-varying pedestal targets,
  those targets should represent the H-mode trajectory rather than manually
  encoding the L-H ramp, as the ramp scaling is applied on top of the target
  values.

``explicit_pedestal`` (**bool** [default = True])
  Controls whether the pedestal model is evaluated explicitly (once per
  timestep) or implicitly (every internal iteration). When ``True`` (default),
  the pedestal model is evaluated once in the pre-step phase and its output is
  frozen during the solver loop. When ``False``, the pedestal model is
  re-evaluated every Newton iteration, coupling pedestal physics to the
  implicit solve, but increasing the non-linearity of the system. Note: for
  ``adaptive_transport`` mode, transport multipliers are always re-evaluated
  implicitly with current profiles regardless of this setting, since the
  saturation model's profile-based feedback loop requires implicit coupling.

``initial_L_mode_bc_multiplier`` (**float** [default = 2.0])
  Multiplier applied to the LCFS right-boundary values (:math:`T_i`, :math:`T_e`,
  :math:`n_e`) at :math:`t_{\text{initial}}` to seed the L-mode pedestal-top
  baselines when the simulation starts directly in H-mode without prior L-mode
  profile data. Must be positive.

``structure_model`` (dict [default = ``{'model_name': 'no_pedestal'}``])
  Configuration for the pedestal structure model, which computes the H-mode
  target pedestal-top temperature, density, radial location, and profile form.
  Custom models (such as
  `epednn-mit <https://github.com/aaronkho/epednn_mit>`_) can also be registered
  via ``torax.pedestal.register_pedestal_structure_model`` (see
  :ref:`model-integration`). All structure models share the following parameter:

  * ``pedestal_profile_form`` (**str** [default = ``"set_at_ped_top"``]):
    Controls the shape of internal boundary conditions in the pedestal region.
    Options:

    * ``"set_at_ped_top"``: Pedestal values are pinned at a single grid cell
      nearest to ``rho_norm_ped_top``.
    * ``"mtanh"``: A smooth modified-tanh (mtanh) profile is applied across the
      pedestal region from the pedestal top to the separatrix, following the mtanh
      parameterization of `Snyder et al., Nucl. Fusion 44 (2004) 320
      <https://doi.org/10.1088/0029-5515/44/2/014>`_.
      For simplicity, the core polynomial part of the full mtanh is neglected;
      only the tanh component is used in the pedestal region.
      The mtanh width :math:`\Delta` is derived on-the-fly from
      ``rho_norm_ped_top`` via the :math:`\psi_N(\rho)` mapping in core profiles:

      .. math::

         f(\psi) = f_{\text{sep}} + a_0 \left[\tanh(1) - \tanh\!\left(\frac{2(\psi - \psi_{\text{mid}})}{\Delta}\right)\right]

      where :math:`\psi_{\text{top}} = \psi_N(\rho_{\text{ped,top}})`,
      :math:`\Delta = (1 - \psi_{\text{top}}) / 1.5`,
      :math:`\psi_{\text{mid}} = 1 - \Delta/2`, and
      :math:`a_0 = (f_{\text{top}} - f_{\text{sep}}) / (\tanh(1) + \tanh(2))`.

  The ``model_name`` key inside ``structure_model`` selects the built-in or
  registered model:

  * ``'no_pedestal'`` (default): Disables the pedestal across the entire
    ``PedestalModel`` regardless of ``formation_model`` or ``saturation_model``.
  * ``'set_T_ped_n_ped'``: Directly specify the pedestal width, electron
    density, ion temperature, and electron temperature (see below).
  * ``'set_P_ped_n_ped'``: Specify the pedestal width, electron density, total
    pedestal pressure, and ion-to-electron temperature ratio (see below).

``formation_model`` (dict)
  Configuration for the pedestal formation model, which determines when L-H
  and H-L transitions occur. Defaults to
  ``{'model_name': 'prescribed',  'pedestal_active': True}`` in
  ``'internal_boundary_condition'`` mode, and must be explicitly
  specified when ``mode`` is ``'adaptive_transport'``.

  .. note::
     Transition conditions are evaluated in the pre-step phase at the **start**
     of each timestep :math:`[t, t + dt]`. A transition condition met at
     :math:`t_{\text{trans}}` therefore takes effect on the first timestep
     starting at :math:`t \ge t_{\text{trans}}` (and not during an earlier step
     that merely crosses :math:`t_{\text{trans}}`).

  All formation models share the following parameters:

  * ``initial_confinement_mode`` (**str | None** [default = ``None``]):
    Optional explicit initial confinement mode (``'l_mode'`` or ``'h_mode'``)
    at :math:`t_{\text{initial}}`. When ``None`` (default), the initial mode is
    inferred automatically from the formation model's transition conditions at
    :math:`t_{\text{initial}}` (cannot be ``'h_mode'`` for ``'no_pedestal'``).
  * ``base_multiplier`` (float [default = 1e-6]): The base value of the
    transport decrease multiplier in ``'adaptive_transport'`` mode when the
    pedestal is active.

  The ``model_name`` key selects the model:

  * ``'prescribed'``: Controls pedestal activation via a prescribed boolean
    schedule rather than a power-threshold scaling law. When
    ``pedestal_active`` is ``True``, activates the internal boundary condition
    in ``'internal_boundary_condition'`` mode (stepping directly between
    ``l_mode`` and ``h_mode`` when ``transition_time_width == 0.0``, or ramping
    through ``TRANSITIONING_TO_H_MODE`` / ``TRANSITIONING_TO_L_MODE`` over
    ``transition_time_width`` when ``transition_time_width > 0.0``), or applies
    a constant transport suppression multiplier (``base_multiplier``) in
    ``'adaptive_transport'`` mode (while the saturation model regulates
    transport once pedestal targets are reached). Additional parameters:

    * ``pedestal_active`` (**time-varying-scalar** [default = True]): Whether
      the pedestal is active (default interpolation is ``STEP``; evaluated at
      step start :math:`t`).

  * ``'martin_scaling'``: Uses the Martin scaling law to determine the L-H
    power threshold. Additional parameters:

    * ``sharpness`` (float [default = 100.0]): Controls the sharpness of the
      transition sigmoid function.
    * ``offset`` (float [default = 0.0]): Dimensionless offset applied to the
      argument of the transition sigmoid function.
    * ``P_LH_prefactor`` (float [default = 1.0]): Dimensionless multiplier
      applied to :math:`P_{LH}`.
    * ``P_LH_hysteresis_factor`` (**time-varying-scalar** [default = 0.8]):
      Hysteresis factor for H-to-L back transitions. When checking for an H-L
      transition, the L-H threshold power :math:`P_{LH}` is multiplied by this
      factor, i.e. the back transition occurs when
      :math:`P_{SOL} < P_{LH} \times` ``P_LH_hysteresis_factor``. A value less
      than 1 means the plasma must lose more power to transition back to L-mode
      than was required to enter H-mode, consistent with experimentally observed
      hysteresis. Must be in [0, 1].
    * ``include_dW_dt_in_P_SOL`` (**bool** [default = False]): Whether to
      include the :math:`dW/dt` term in the :math:`P_{SOL}` calculation used for
      comparing against :math:`P_{LH}`. When ``False`` (default), uses
      :math:`P_{heat}` (total auxiliary heating + Ohmic power - sinks) instead
      of :math:`P_{SOL} = P_{heat} - dW/dt`, avoiding spurious dithering during
      L-mode to H-mode transitions.

  * ``'delabie_scaling'``: Uses the Delabie scaling law for the L-H power
    threshold. Accepts the same parameters as ``'martin_scaling'``, plus:

    * ``divertor_configuration`` (str [default = ``'horizontal_target'``]): Divertor
      configuration (``'horizontal_target'`` for horizontal target or ``'vertical_target'`` for vertical
      target).

``saturation_model`` (dict)
  Configuration for the pedestal saturation model, which determines how the
  pedestal values scale relative to H-mode target values. The ``model_name``
  key selects the model:

  * ``'profile_value'``: Uses the current profile value at the pedestal top
    for saturation. Additional parameters:

    * ``steepness`` (float [default = 100.0]): Controls the steepness of the
      saturation function.
    * ``offset`` (float [default = 0.1]): Offset for the saturation function.
    * ``base_multiplier`` (float [default = 1e6]): Base multiplier for the
      saturation function.

``pedestal_top_smoothing_width`` (**time-varying-scalar** [default = 0.02])
  Width of the smoothing region at the pedestal top boundary, in units of
  :math:`\hat{\rho}`.

``chi_max`` (**time-varying-scalar** [default = 1.0])
  Maximum effective thermal diffusion coefficient from the core transport model
  allowed in the pedestal region [m^2/s].

``D_e_max`` (**time-varying-scalar** [default = 1.0])
  Maximum effective particle diffusion coefficient from the core transport model
  allowed in the pedestal region [m^2/s].

``V_e_max`` (**time-varying-scalar** [default = 1.0])
  Maximum effective particle pinch velocity from the core transport model
  allowed in the pedestal region [m/s].

``V_e_min`` (**time-varying-scalar** [default = -1.0])
  Minimum effective particle pinch velocity from the core transport model
  allowed in the pedestal region [m/s].

The following ``structure_model`` ``model_name`` options are built into
TORAX:

no_pedestal
===========
No pedestal profile is set. This is the default option.

set_T_ped_n_ped
===============
Directly specify the pedestal width, electron density, ion temperature and
electron temperature.

``n_e_ped`` (**time-varying-scalar** [default = 0.7e20])
  Electron density at the pedestal top.
  In units of reference density if ``n_e_ped_is_f_gw==False``. In units of
  Greenwald fraction if ``n_e_ped_is_f_gw==True``.

``n_e_ped_is_f_gw`` (**bool** [default = False])
  Toggles units of ``n_e_ped``.

``T_i_ped`` (**time-varying-scalar** [default = 5.0])
  Ion temperature at the pedestal top in units of keV.

``T_e_ped`` (**time-varying-scalar** [default = 5.0])
  Electron temperature at the pedestal top in units of keV.

``rho_norm_ped_top`` (**time-varying-scalar** [default = 0.91])
  Location of pedestal top, in units of :math:`\hat{\rho}`.

set_P_ped_n_ped
===============
Set the pedestal width, electron density and ion temperature by providing the
total pressure at the pedestal and the ratio of ion to electron temperature.

``P_ped`` (**time-varying-scalar** [default = 1e5])
  The plasma pressure at the pedestal in units of :math:`Pa`.

``P_ped_multiplier`` (**time-varying-scalar** [default = 1.0])
  Multiplier for the pedestal pressure.

``n_e_ped`` (**time-varying-scalar** [default = 0.7e20])
  Electron density at the pedestal top. In units of reference density if
  ``n_e_ped_is_f_gw==False``. In units of  Greenwald fraction if
  ``n_e_ped_is_f_gw==True``.

``n_e_ped_is_f_gw`` (**bool** [default = False])
  Toggles units of ``n_e_ped``.

``T_i_T_e_ratio`` (**time-varying-scalar** [default = 1.0])
  Ratio of the ion and electron temperature at the pedestal.

``rho_norm_ped_top`` (**time-varying-scalar** [default = 0.91])
  Location of pedestal top, in units of :math:`\hat{\rho}`.

.. include:: ../links.rst

.. _rotation_physics:

Rotation Physics
################

The radial electric field (:math:`E_r`), which drives :math:`E \times B` plasma
rotation, is a crucial factor in turbulent transport, as :math:`E \times B`
shear can suppress turbulence.

The radial electric field :math:`E_r` is determined from the radial force
balance equation for the main ions:

.. math::
  E_r = \frac{1}{Z_i e n_i} \frac{d P_i}{dr} - v_{\phi} B_{\theta} + v_{\theta} B_{\phi}

where :math:`P_i` is the main ion pressure, :math:`n_i` is the main ion
density, :math:`Z_i` is the main ion charge number, :math:`e` is the elementary
charge, :math:`v_{\phi}` is the toroidal rotation velocity, :math:`v_{\theta}`
is the poloidal rotation velocity, :math:`B_{\theta}` is the poloidal magnetic
field, and :math:`B_{\phi}` is the toroidal magnetic field. The derivatives
are with respect to a midplane-averaged radial coordinate.

The poloidal velocity :math:`v_{\theta}` is provided by the neoclassical
``poloidal_velocity`` sub-model. By default (``'kim'``), it implements
Equation 33 from |kim1991|:

.. math::
  v_{\theta} = k_{neo} \frac{1}{Z_i e} \frac{dT_i}{dr} \frac{B_{tor}}{B_{total}^2}

where :math:`k_{neo}` is a neoclassical coefficient, :math:`dT_i/dr` is the
radial gradient of the ion temperature, and :math:`B_{total}` is the total
magnetic field.

The neoclassical coefficient :math:`k_{neo}` is based on Equation (6.135) from
|hinton1976|. This coefficient depends on the normalized ion collisionality
(:math:`\nu_{i}^{*}`) and the inverse aspect ratio (:math:`\epsilon`). The
limits of this formula are approximately 1.17 in the banana regime
(:math:`\nu_{i}^{*} \rightarrow 0`) and -2.1 in the Pfirsch-Schluter regime
(:math:`\nu_{i}^{*} \rightarrow \infty`).

The normalized ion collisionality :math:`\nu_{i}^{*}` is calculated based on
|sauter99|, and depends on the safety factor (:math:`q`), geometry, ion
density (:math:`n_i`), ion temperature (:math:`T_i`), effective charge number
(:math:`Z_{eff}`), and the ion-ion Coulomb logarithm (:math:`\log \Lambda_{ii}`).

The :math:`E \times B` velocity (:math:`v_{E \times B}`) is derived from the
radial electric field :math:`E_r` and the total magnetic field as follows:

.. math::
  v_{E \times B} = \frac{E_r}{B_{total}}

Rotation effects are currently disabled by default in transport models. They can
be enabled through the transport model configuration.

**Rotation in Transport Models:**

*   **TGLFNN-ukaea:** The TGLFNN-ukaea model includes the :math:`E \times B`
    shearing rate as an input feature, directly informing the model's turbulent
    transport predictions. To enable this, set the `use_rotation` parameter to
    `True` in the transport model configuration.

*   **QLKNN:** The QLKNN transport model
    incorporates a "rotation rule" that modifies turbulent fluxes (specifically
    for ITG and TEM modes) based on E×B shear. The modification combines two
    physical effects:

    * **Waltz rule** applied to poloidal/pressure contributions (from
      :math:`\nabla P_i` and :math:`v_\theta B_\phi`): Simple suppression
      :math:`f_{waltz} = -\alpha`, where :math:`\alpha` is set by the
      ``shear_suppression_alpha`` parameter (default 1.0). See |waltz1998|.

    * **Victor rule** applied to toroidal contributions (from
      :math:`-v_\phi B_\theta`): A fitted model that can produce both
      suppression and enhancement depending on local plasma parameters,
      due to competing stabilizing (E×B shear) and destabilizing (parallel
      velocity shear) effects. See |qlknn10d|.

    The combined scaling factor is:

    .. math::
      f_{rot} = \text{clip}\left(1 + f_{waltz}\frac{|\gamma_{pp}|}{\gamma_{max}}
      + f_{victor}\frac{|\gamma_{tor}|}{\gamma_{max}}, 0\right)

    where :math:`\gamma_{pp}` and :math:`\gamma_{tor}` are the E×B shearing
    rates from poloidal/pressure and toroidal contributions respectively.

    The application of the rotation rule is controlled by the
    ``rotation_mode`` configuration parameter. Options for ``rotation_mode`` are:

    * ``off``: No rotation correction is applied.

    * ``half_radius``: The rotation correction is only applied to the outer
      half of the radius (:math:`\hat{\rho} > 0.5`).

    * ``full_radius``: The rotation correction is applied everywhere.

**Tuning Rotation Effects:**

Two parameters are available to fine-tune the impact of rotation, both of them
default to 1.0:

*   ``rotation_multiplier``: Located in the transport model configs, this
    parameter scales the :math:`E \times B` shear term.
*   ``poloidal_velocity_multiplier``: Found under the
    ``neoclassical.poloidal_velocity`` configuration, this parameter directly
    scales the poloidal velocity term.

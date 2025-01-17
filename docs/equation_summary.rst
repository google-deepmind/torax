.. _equations:

TORAX equation summary
######################

TORAX simulates the time evolution of core plasma profiles using a coupled set
of 1D transport PDEs, discretized in space and time, and solved numerically.
The PDEs arise from flux-surfaced averaged moments of the underlying kinetic
equations, and from Faraday's law.

Governing equations
===================

TORAX solves coupled 1D PDEs in normalized toroidal flux coordinates,
:math:`\hat{\rho}`. The set of 1D PDEs being solved are as follows:

  Ion heat transport, governing the evolution of the ion temperature :math:`T_i`.

  .. math::

    \begin{multline}
    \frac{3}{2} V'^{-5/3} \left(\frac{\partial }{\partial t}-
    \frac{\dot{\Phi}_b}{2\Phi_b}\frac{\partial}{\partial\hat{\rho}}\hat{\rho}\right)\left[V'^{5/3} n_i T_i\right] = \\
    \frac{1}{V'} \frac{\partial}{\partial \hat{\rho}} \left[
      \chi_i n_i \frac{g_1}{V'} \frac{\partial T_i}{\partial \hat{\rho}} -
      g_0q_i^{\mathrm{conv}}T_i\right] + Q_i
    \end{multline}

  If multiple main ion species are present (e.g., a D-T mix), then :math:`n_i` represents the sum of all
  main ions, and ion attributes like charge and mass are averaged values for the mixture, weighted by fractional abundance.

  Electron heat transport, governing the evolution of the electron temperature :math:`T_e`.

  .. math::

    \begin{multline}
    \frac{3}{2} V'^{-5/3} \left(\frac{\partial }{\partial t}-
    \frac{\dot{\Phi}_b}{2\Phi_b}\frac{\partial}{\partial\hat{\rho}}\hat{\rho}\right)\left[V'^{5/3} n_e T_e\right] = \\
    \frac{1}{V'} \frac{\partial}{\partial \hat{\rho}} \left[
      \chi_e n_e \frac{g_1}{V'} \frac{\partial T_e}{\partial \hat{\rho}} -
      g_0q_e^{\mathrm{conv}}T_e \right] + Q_e
    \end{multline}

  Electron particle transport, governing the evolution of the electron density :math:`n_e`.

  .. math::

    \begin{multline}
    \left(\frac{\partial}{\partial t}-
    \frac{\dot{\Phi}_b}{2\Phi_b}\frac{\partial}{\partial\hat{\rho}}\hat{\rho}\right)\left[ n_e V' \right] = \\
    \frac{\partial}{\partial \hat{\rho}} \left[D_e n_e \frac{g_1}{V'} \frac{\partial n_e}{\partial \hat{\rho}}
    - g_0V_e n_e \right] + V'S_n
    \end{multline}

  Current diffusion, governing the evolution of the poloidal flux :math:`\psi`.

  .. math::

    \begin{multline}
    \frac{16 \pi^2 \sigma_{||}\mu_0 \hat{\rho} \Phi_b^2}{F^2}\left(\frac{\partial \psi}{\partial t}-
    \frac{\hat{\rho}\dot{\Phi}_b}{2\Phi_b}\frac{\partial \psi}{\partial \hat{\rho}}\right)  = \\
    \frac{\partial}{\partial \hat{\rho}} \left( \frac{g_2 g_3}{\hat{\rho}} \frac{\partial \psi}{\partial \hat{\rho}} \right) -
    \frac{8\pi^2 V' \mu_0 \Phi_b}{F^2} \langle \mathbf{B} \cdot \mathbf{j}_{ni} \rangle
    \end{multline}

where :math:`T_{i,e}` are ion and electron temperatures, :math:`n_{i,e}` are ion
and electron densities, and :math:`\psi` is poloidal flux. :math:`\chi_{i,e}` are
ion and electron heat conductivities, :math:`q_{i,e}^{\mathrm{conv}}` are ion
and electron heat convection terms, :math:`D_e` is electron particle diffusivity,
and :math:`V_e` is electron particle convection. :math:`Q_{i,e}` are the total
ion and electron heat sources, and :math:`S_n` is the total electron particle source.
:math:`V' \equiv dV/d\hat{\rho}`, i.e. the flux surface volume derivative. :math:`\sigma_{||}`
is the plasma neoclassical conductivity, and :math:`\langle \mathbf{B} \cdot \mathbf{j}_{ni} \rangle` is the
flux-surface-averaged non-inductive current (comprised of the bootstrap current
and external current drive) projected onto and multiplied by the magnetic field.
:math:`F \equiv RB_\varphi` is the toroidal field flux function, where :math:`R` is major radius, and
:math:`B_\varphi` is the toroidal magnetic field. :math:`\Phi_b` is the toroidal flux enclosed by the
last closed flux surface, and :math:`\dot{\Phi}_b` is its time derivative, non-zero with time-varying toroidal
magnetic field and/or last closed flux surface shape. :math:`\mu_0` is the permeability of free space.
The geometric quantities :math:`g_0`, :math:`g_1`, :math:`g_2` and :math:`g_3` are defined as follows.

.. math::

  g_0 = \left< \left( \nabla V \right) \right>

.. math::

  g_1 = \left< \left( \nabla V \right)^2 \right>

where :math:`\nabla V` is the radial derivative of the plasma volume, and
:math:`\langle \rangle` denotes flux surface averaging.

.. math::

  g_2 = \left< \frac{\left( \nabla V \right)^2}{R^2}\right>

.. math::

  g_3 = \left< \frac{1}{R^2}\right>

where :math:`R` is the major radius along the flux surface being averaged.

The geometry terms :math:`V'`, :math:`g_0`, :math:`g_1`, :math:`g_2` and :math:`g_3`
are calculated from flux-surface-averaged outputs of a Grad-Shafranov equilibrium
code (see :ref:`physics_models`), either pre-calculated or coupled to TORAX
within a larger workflow.

Currently, TORAX assumes a static magnetic equilibrium, apart from varying plasma current,
and does not consider the time variation of :math:`V'` or a non-zero :math:`\dot{\Phi}_b` term.
Upcoming development plans include incorporating geometry time-dependence.

The boundary conditions are as follows. All equations have a zero-derivative
boundary condition at :math:`\hat{\rho}=0`. The :math:`T_i`, :math:`T_e`, :math:`n_e`
equations have fixed boundary conditions at :math:`\hat{\rho}=1`, which are
user-defined. The :math:`\psi` equation has a Neumann (derivative) boundary
condition at :math:`\hat{\rho}=1`, which sets the total plasma current through the relation:

.. math::

  I_p = \left[\frac{\partial \psi}{\partial \rho} \frac{g_2 g_3}{\rho}\frac{R_0 J}{16\pi^4\mu_0}\right]_{LCFS}

Future work can extend the governing equations to include momentum transport,
and multiple ion species including impurities. Details of the physics models
underlying the PDE coefficients is provided in :ref:`physics_models`.

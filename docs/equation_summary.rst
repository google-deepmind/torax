.. _equations:

TORAX equation summary
######################

TORAX simulates the time evolution of core plasma profiles using a coupled set of 1D transport PDEs, discretized in space and time, and solved numerically. The PDEs arise from flux-surfaced averaged moments of the underlying kinetic equations, and from Faraday's law.

Governing equations
===================

TORAX solves coupled 1D PDEs in normalized toroidal flux coordinates,  :math:`\hat{\rho}`. The set of 1D PDEs being solved are as follows:

  Ion heat transport, governing the evolution of the ion temperature :math:`T_i`.

  .. math::

    \frac{3}{2} \frac{\partial (n_i T_i)}{\partial t} = \frac{1}{V'} \frac{\partial}{\partial \hat{\rho}}\left[ V' \left(  \chi_i n_i \frac{g_1}{V'} \frac{\partial T_i}{\partial \hat{\rho}} \right) \right] + Q_i
    
  Electron heat transport, governing the evolution of the electron temperature :math:`T_e`.

  .. math::

    \frac{3}{2} \frac{\partial (n_e T_e)}{\partial t} =  \frac{1}{V'} \frac{\partial}{\partial \hat{\rho}} \left[ V' \left(  \chi_e n_e \frac{g_1}{V'} \frac{\partial T_e}{\partial \hat{\rho}} \right) \right] + Q_e

  Electron particle transport, governing the evolution of the electron density :math:`n_e`.

  .. math::

    \frac{\partial n_e}{\partial t} = \frac{1}{V'} \frac{\partial}{\partial \hat{\rho}} \left[ V' \left(  D_e n_e \frac{g_1}{V'} \frac{\partial n_e}{\partial \hat{\rho}} - g_0V_e n_e  \right) \right] + S_n

  Current diffusion, governing the evolution of the poloidal flux $\psi$.

  .. math::

    \frac{\sigma_{||}\mu_0 \hat{\rho}}{R_0 J^2}\frac{\partial \psi}{\partial t}  = \frac{\partial}{\partial \hat{\rho}} \left( \frac{G_2}{J} \frac{\partial \psi}{\partial \hat{\rho}} \right) - \frac{V' \mu_0}{2 \pi R_0 J^2} \langle j_{ni} \rangle 

where :math:`T_{i,e}` are ion and electron temperatures, :math:`n_{i,e}` are ion and electron densities, and :math:`\psi` is poloidal flux. :math:`\chi_{i,e}` are ion and electron heat conductivities, :math:`D_e` is electron particle diffusivity, and :math:`V_e` is electron particle convection. :math:`Q_{i,e}` are the total ion and electron heat sources, and :math:`S_n` is the total electron particle source. :math:`V' \equiv dV/d\hat{\rho}`, i.e. the flux surface volume derivative, :math:`R_0` is the major radius, :math:`\sigma_{||}` is the plasma neoclassical conductivity, and :math:`\langle j_{ni} \rangle` is the flux-surface-averaged non-inductive current, being comprised of the bootstrap current and external current drive. :math:`J \equiv \frac{RB_\phi}{R_0B_0}` is the normalized toroidal field flux function, where :math:`B_\phi` the toroidal magnetic field. :math:`\mu_0` is the permeability of free space. The geometric quantities :math:`g_0`, :math:`g_1`, and :math:`G_2` are defined as follows. 

.. math::
  
  g_0 = \left< \left( \nabla V \right) \right> 

.. math::

  g_1 = \left< \left( \nabla V \right)^2 \right> 

where :math:`\nabla V` is the radial derivative of the plasma volume, and :math:`\langle \rangle` denotes flux surface averaging.

.. math::
  
  G_2 = \frac{V'}{4\pi^2}\left< \left( \frac{(\nabla \rho)^2}{R^2} \right) \right> 

where :math:`\nabla \rho` is the radial derivative of the toroidal flux coordinate.

The geometry terms :math:`V'`, :math:`g_0`, :math:`g_1`, and :math:`G_2` are calculated from flux-surface-averaged outputs of a Grad-Shafranov equilibrium code (see :ref:`physics_models`), either pre-calculated or coupled to TORAX within a larger workflow. Currently, TORAX assumes a static magnetic equilibrium. Upcoming development plans include incorporating time-dependent geometry terms..

The boundary conditions are as follows. All equations have a zero-derivative boundary condition at :math:`\hat{\rho}=0`. The :math:`T_i`, :math:`T_e`, :math:`n_e` equations have fixed boundary conditions at :math:`\hat{\rho}=1`, which are user-defined. The :math:`\psi` equation has a Neumann (derivative) boundary condition at :math:`\hat{\rho}=1`, which sets the total plasma current through the relation:

.. math::

  I_p = \left[\frac{\partial \psi}{\partial \rho} \frac{G_2}{\mu_0}\right]_{LCFS}

Future work can extend the governing equations to include momentum transport, and multiple ion species including impurities. Details of the physics models underlying the PDE coefficients is provided in :ref:`physics_models`.

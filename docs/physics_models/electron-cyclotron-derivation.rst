.. _ec-derivation:

Derivation of electron-cyclotron current drive efficiency
=========================================================

*(based on notes from Emmi Tholerus, UKAEA)*

The local dimensionless electron-cyclotron (EC) current drive efficiency is
defined in |lin-liu|, |luce1999| as

.. math::

   \zeta = \frac{e^3 \ln \Lambda}{16\pi\varepsilon_0^2} \frac{n_e}{T_e}
   \frac{j_\mathrm{tor}}{Q_\mathrm{ec}},

where :math:`e` is the electron charge in :math:`C`, :math:`\ln \Lambda` is the
electron-electron Coulomb logarithm, :math:`\varepsilon_0` the vacuum
permittivity, :math:`n_e` the electron density in :math:`m^{-3}`, :math:`T_e`
the electron temperature in :math:`J`, :math:`j_\mathrm{tor} = \frac{dI_p}{dA}`
with :math:`A` as the cross sectional area inside the flux surface, and
:math:`Q_\mathrm{ec}` is the absorbed EC power density in :math:`Wm^{-3}`.

Assume that the current drive and power absorption are localised in a region
:math:`\delta\rho` around a flux surface at :math:`\rho`, meaning that the total
current driven is :math:`I = j_\mathrm{tor}\delta A` and the total absorbed power
is :math:`P = Q \delta V`.
Then,

.. math::

     \frac{j_\mathrm{tor}}{Q_\mathrm{ec}} \approx
     \frac{I_\mathrm{ec}}{P_\mathrm{ec}} \frac{\delta V}{\delta A} \approx
     \frac{I_\mathrm{ec}}{P_\mathrm{ec}} \frac{V'}{A'} =
     \frac{I_\mathrm{ec}}{P_\mathrm{ec}} \frac{2\pi}{\langle R^{-1} \rangle}

For conventional tokamaks, it is often acceptable to take
:math:`\langle R^{-1} \rangle \approx R^{-1}`; however, this is not valid for
high-beta strongly shaped plasmas like those found in STs.

Substituting in the above gives

.. math::

   \zeta = \frac{e^3 \ln \Lambda}{16\pi\varepsilon_0^2}
   \frac{2\pi}{\langle R^{-1} \rangle} \frac{n_e}{T_e} \frac{I}{P} ,

with all variables in SI units.

As per |lin-liu|, the EC-driven current is parallel to the magnetic field and
:math:`\langle J \cdot B \rangle` is a flux function that can be written as

.. math::

    \langle J \cdot B \rangle = \frac{J}{B} \langle B^2 \rangle.

We can therefore write

.. math::

    \langle J \cdot B \rangle &= \langle J_\phi B_\phi
    \rangle + \langle \frac{J_\phi B_\mathrm{pol}^2}{B_\phi} \rangle =
    F \langle \frac{J_\phi}{R}\rangle +
    \frac{J}{B} \langle B_\mathrm{pol}^2 \rangle \\
    &= F \langle \frac{J_\phi}{R}\rangle +
    \frac{\langle J \cdot B \rangle}{\langle B^2 \rangle}
    \langle B_\mathrm{pol}^2 \rangle \\
    &= F \langle \frac{J_\phi}{R}\rangle
    \left( 1 - \frac{ \langle B_\mathrm{pol}^2  \rangle}
    {\langle B^2 \rangle} \right)^{-1} \\
    &= F \langle \frac{J_\phi}{R}\rangle
    \left( \frac{\langle B^2 \rangle- \langle B_\mathrm{pol}^2
    \rangle}{\langle B^2 \rangle} \right)^{-1} \\
    &= F \langle \frac{J_\phi}{R}\rangle
    \left( \frac{\langle B^2 \rangle}{\langle B^2 \rangle-
    \langle B_\mathrm{pol}^2 \rangle} \right) \\
    &= F \langle \frac{J_\phi}{R}\rangle
    \left( \frac{\langle B_\phi^2 \rangle +
    \langle B_\mathrm{pol}^2 \rangle}{\langle B_\phi^2 \rangle} \right) \\

    &= F \langle \frac{J_\phi}{R}\rangle\left( 1 +
    \frac{\langle B_\mathrm{pol}^2\rangle}{\langle B_\phi^2 \rangle} \right)

We have :math:`\langle B_\phi^2 \rangle = F^2 \langle R^{-2} \rangle`, and

.. math::

    \langle B_\mathrm{pol}^2 \rangle = \frac{1}{4\pi^2} \left\langle
    \frac{|\nabla \Psi_\mathrm{pol}|^2}{R^2} \right\rangle &= \frac{1}{4\pi^2}
    \left(\frac{\Psi_\mathrm{pol}'}{V'}\right)^2\left\langle
    \frac{|\nabla \Psi_\mathrm{pol}|^2}{R^2} \right\rangle\\
    &= \frac{F^2 \langle R^{-2} \rangle^2
    \langle \frac{|\nabla V|^2}{R^2} \rangle}{16\pi^4q^2}


where we have used
:math:`q = \frac{F \langle R^{-2} \rangle V'}{2\pi \Psi'_\mathrm{pol}}`
(derived from the definition of the safety factor
:math:`q = \frac{\partial \Psi_\mathrm{tor}}{\partial\Psi_\mathrm{pol}}`).

Hence,

.. math::

    \langle J \cdot B \rangle = F \left\langle\frac{J_\phi}{R}\right\rangle
    \left(1+ \frac{g_2 g_3}{16\pi^4 q^2}\right)

where :math:`F = R B_\phi` is the toroidal field flux function,
:math:`g_2 = \left\langle \frac{|\nabla V|^2}{R^2} \right\rangle`, and
:math:`g_3 = \langle R^{-2} \rangle`.

Using :math:`\left\langle\frac{J_\phi}{R}\right\rangle = \langle R^{-1} \rangle
j_\mathrm{tor}` and dividing by the vacuum toroidal magnetic field on axis
:math:`B_0` gives the parallel current density required by the :math:`\psi`
source term in TORAX:

.. math::
    :label: j_parallel_ec

    j_{\parallel,\mathrm{EC}} = \frac{\langle J \cdot B \rangle}{B_0}
    = \frac{F \langle R^{-1} \rangle}{B_0}
    \left(1+ \frac{g_2 g_3}{16\pi^4 q^2}\right) j_\mathrm{tor}
    = \frac{16\pi\varepsilon_0^2}{e^3 \ln\Lambda}
    \frac{F \langle R^{-1} \rangle}{B_0}
    \left(1+ \frac{g_2 g_3}{16\pi^4 q^2}\right)
    \frac{T_e}{n_e} \zeta Q_\mathrm{ec}.

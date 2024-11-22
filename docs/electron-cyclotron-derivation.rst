.. _ec-derivation:

Derivation of electron-cyclotron current drive efficiency
=========================================================

The local dimensionless electron-cyclotron (EC) current drive efficiency is given as [1]_,

.. math::

   \zeta = \frac{e^3}{\varepsilon_0^2} \frac{n_e}{T_e} R_0  \frac{dI^\mathrm{ec}_\mathrm{tor}}{dP^\mathrm{ec}_\mathrm{absorbed}},

where :math:`e` is the electron charge, :math:`\varepsilon_0` the vacuum permittivity, :math:`n_e` the electron density in :math:`m^{-3}`, :math:`T_e` the electron temperature in :math:`J`, and :math:`R_0` the device major radius in :math:`m`.
:math:`dI^\mathrm{ec}_\mathrm{tor}` is defined as the toroidal EC current driven in the elemental area between two flux surfaces, :math:`dA`, and :math:`dP^\mathrm{ec}_\mathrm{absorbed}` is the EC power absorbed in the elemental volume between two flux surfaces, :math:`dV`.

Defining the flux-surface averaged toroidal current density as:

.. math::

   j^\mathrm{ec}_\mathrm{tor} = \frac{\partial I^\mathrm{ec}_\mathrm{tor}}{\partial A},

and setting :math:`dP = Q^\mathrm{ec} dV = 2\pi R_0 Q^\mathrm{ec} dA` gives:

.. math::

   \zeta = \frac{e^3}{\varepsilon_0^2} \frac{n_e}{T_e} \frac{j^\mathrm{ec}_\mathrm{tor}}{2\pi Q^\mathrm{ec}}.

From the ASTRA manual [2]_,

.. math::

   \langle \boldsymbol{j}^\mathrm{ec} \cdot \boldsymbol{B} \rangle &= 2\pi R_0 B_0 J^2 \frac{\partial}{\partial V} \left[\frac{I^\mathrm{ec}_\mathrm{tor}}{J}\right], \\
   &= 2\pi F^2 \frac{\partial}{\partial V} \left[\frac{I_p}{F}\right], \\
   &= 2\pi \left( F \frac{\partial I^\mathrm{ec}_\mathrm{tor}}{\partial V} - I^\mathrm{ec}_\mathrm{tor} \frac{\partial F}{\partial V} \right), \\
   \langle \boldsymbol{j}^\mathrm{ec} \cdot \boldsymbol{B} \rangle &= 2\pi \left( F \frac{j^\mathrm{ec}_\mathrm{tor}}{2\pi R_0} - I^\mathrm{ec}_\mathrm{tor} \frac{\frac{\partial F}{\partial \rho}}{\frac{\partial V}{\partial \rho}} \right),

where :math:`J = \frac{F}{R_0 B_0} = \frac{RB_\phi}{R_0 B_0}`.

**Assume the second term is small,** i.e.

.. math::

   \frac{F j^\mathrm{ec}_\mathrm{tor}}{2\pi R_0} \gg \frac{I^\mathrm{ec}_\mathrm{tor} \frac{\partial F}{\partial \rho}}{\frac{\partial V}{\partial \rho}}.

In practice, testing for various devices, we found that typically
:math:`\frac{I^\mathrm{ec}_\mathrm{tor} \frac{\partial F}{\partial \rho}}{\frac{\partial V}{\partial \rho}} \propto 1e^{-3}-1e^{-5} \times \frac{F j^\mathrm{ec}_\mathrm{tor}}{2\pi R_0}`.

Then:

.. math::

   \langle \boldsymbol{j}^\mathrm{ec} \cdot \boldsymbol{B} \rangle = \frac{F}{R_0} j^\mathrm{ec}_\mathrm{tor},

and so:

.. math::

   \zeta = \frac{e^3}{\varepsilon_0^2} \frac{R_0}{2\pi F} \frac{n_e}{T_e} \frac{\langle \boldsymbol{j}^\mathrm{ec} \cdot \boldsymbol{B} \rangle}{Q^\mathrm{ec}}.


.. rubric:: References

.. [1] Equation 44 in: Lin-Liu, Y. R., Chan, V. S., & Prater, R. (2003). `Electron cyclotron current drive efficiency in general tokamak geometry <https://doi.org/10.1063/1.1610472>`_. Physics of Plasmas, 10(10), 4064-4071.
.. [2] Equation 34 in: Pereverzev, G., & Yushmanov, P. N. (2002). `ASTRA: Automated System for TRansport Analysis <https://w3.pppl.gov/~hammett/work/2009/Astra_ocr.pdf>`_. IPP 5/98. Germany.

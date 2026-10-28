.. include:: ../links.rst

.. _physics_geometry:

Magnetic geometry
#################

TORAX presently supports four geometry models for determining the metric
coefficients and flux surface averaged geometry variables required for the
transport equations.

- **CHEASE:** This model utilizes equilibrium data from the |chease| fixed
  boundary Grad-Shafranov equilibrium code. Users provide a CHEASE output
  file, and TORAX extracts the relevant geometric quantities.

- **FBT:** This model utilizes equilibrium data from the |fbt| free boundary
  Grad-Shafranov equilibrium code. Users provide FBT output files, and TORAX
  extracts the relevant geometric quantities.

- **EQDSK:** This model utilizes equilibrium data provided in the EQDSK
  format. Users provide EQDSK formatted files. TORAX extracts the relevant
  geometric quantities and carries out the necessary flux surface averaging
  based on traced contours of the 2D poloidal flux data.

- **Circular:** For testing and demonstration purposes, TORAX includes a
  simplified circular geometry model. This model assumes a circular plasma
  cross-section and corrects for elongation to approximate the metric
  coefficients.

Using :math:`\psi` and the magnetic geometry terms, the toroidal current density
is calculated as follows:

.. math::

  j_\mathrm{tor} = \frac{R_0^2}{8\pi^3\mu_0}\frac{\partial }{\partial V}
  \left(\frac{g_2 g_3 J}{\rho} \frac{\partial \psi}{\partial \rho}\right)

where :math:`V` is the volume enclosed by a flux surface.

The safety factor :math:`q` is calculated as follows:

.. math::

  q = 2\pi B_0 \rho \left( \frac{\partial \psi}{\partial \rho} \right)^{-1}

With :math:`q(\rho=0)=\frac{2B_0}{\mu_0 j_{tot}(\rho=0) R_0}`.

For simulations of tokamak scenarios with dynamic equilibra, sequences of
geometry data can be provided. TORAX interpolates geometry data at each end
of each timestep interval, using both sets of values in the theta solver method.
Time-derivative terms are calculated based on finite differences. Generalization
to enable self-consistent geometry calculations by models within the solver loop
is planned.

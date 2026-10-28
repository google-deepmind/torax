.. include:: ../links.rst

.. _physics_fast_ions:

Fast Ion Physics
################

Fast ions are particles with kinetic energies well above the thermal equilibrium
distribution of the bulk plasma. They can originate from auxiliary heating
(e.g., ICRH, NBI) or from fusion reactions. Fast ions affect the overall plasma
equilibrium and performance through three main mechanisms:

1. **Dilution:** Fast ions are part of the total ion population and displace
   thermal ions to maintain quasineutrality.

2. **Pressure:** Fast ions contribute significantly to the total energy density
   but have a non-thermal energy distribution, characterized by an effective
   temperature :math:`T_{tail} \gg T_{thermal}`.

3. **ITG stabilization:** Fast ions can stabilize Ion Temperature Gradient
   turbulence, increasing confinement. Currently we only treat electrostatic
   stabilization through modification of ITG resonances.

TORAX provides a modular framework for tracking fast ion populations and
consistently incorporating their effects into pressure and density accounting.

Fast Ion Representation
=======================
Each fast ion population is represented by a ``FastIon`` dataclass, which
stores:

- **species**: The species name (e.g., ``'He3'``).
- **source**: The name of the generating source (e.g., ``'icrh'``).
- **n**: Density profile :math:`n_{tail}` [:math:`m^{-3}`].
- **T**: Effective temperature profile :math:`T_{tail}` [keV].

The ``CoreProfiles`` state stores a tuple of ``FastIon`` objects, enabling
multiple fast ion populations from different sources or species to coexist.

Bimaxwellian Split
==================
Presently we do not treat the full distribution function of fast ion
populations and characterize fast ions with an effective Maxwellian around the
average energy. For sources such as ICRH minority heating, the total minority
species population is split into a thermal "bulk" component and a non-thermal
"tail" component using a power balance closure.

The model proceeds as follows:

1. **Tail temperature** :math:`T_{tail}` is computed from the Stix distribution
   |stix1975| using the Spitzer slowing-down time |stix1972|:

   .. math::

     \tau_s = \frac{6.27 \times 10^8 \, A_f \, T_e^{3/2}}{Z_f^2 \, n_e \, \ln \Lambda_{ei}}

   where :math:`A_f` and :math:`Z_f` are the mass and charge numbers of the fast
   species, :math:`T_e` is the electron temperature in eV, :math:`n_e` is in
   :math:`cm^{-3}`, and :math:`\ln \Lambda_{ei}` is the electron-ion Coulomb
   logarithm.

   The Stix parameter :math:`\xi` is then:

   .. math::

     \xi = \frac{P_{abs} \cdot \tau_s / 2}{\frac{3}{2} n_{total} T_e}

   giving the tail temperature:

   .. math::

     T_{tail} = T_e \, (1 + \xi)

2. **Tail density** :math:`n_{tail}` is determined from power balance. The
   absorbed power per unit volume is balanced by collisional energy transfer
   from the tail to all bulk species (electrons, main ions, and impurities):

   .. math::

     P_{abs} = \frac{3}{2} n_{tail} \sum_b \nu_\epsilon^{ab} \left( T_{tail} - T_b \right)

   where :math:`\nu_\epsilon^{ab}` is the NRL Formulary energy exchange rate
   between the fast species :math:`a` and bulk species :math:`b`:

   .. math::

     \nu_\epsilon^{ab} = \frac{1.8 \times 10^{-19} \sqrt{m_a m_b} \, Z_a^2 Z_b^2 \, n_b \ln \Lambda}
     {(m_b T_a + m_a T_b)^{3/2}}

   Solving for :math:`n_{tail}`:

   .. math::

     n_{tail} = \frac{P_{abs}}
     {\frac{3}{2} \sum_b \nu_\epsilon^{ab} (T_{tail} - T_b)}

   The tail density is clipped to :math:`[0, 0.99 \, n_{total}]` for numerical
   stability.

3. **Conservation**: The bulk density is obtained by subtraction:

   .. math::

     n_{bulk} = n_{total} - n_{tail}

Pressure Accounting
===================
Fast ions are excluded from the thermal pressure calculation to avoid
double-counting. The pressure split in ``CoreProfiles`` is:

- **Thermal ion pressure:**

  .. math::

    p_{th,i} = (n_i + n_{imp,thermal}) \, T_i \, k_B

  where :math:`n_{imp,thermal}` excludes fast ion densities for species that are
  also in the impurity mixture.

- **Fast ion pressure:**

  .. math::

    p_{fast} = \sum_j n_{tail,j} \, T_{tail,j} \, k_B

  summed over all fast ion populations :math:`j`.

- **Total pressure:**

  .. math::

    p_{total} = p_{th,e} + p_{th,i} + p_{fast}

The fast ion pressure is important for equilibrium calculations since it
contributes to the Shafranov shift.


Dilution
========
Because fast ions are typically a subset of an existing ion species (e.g., He3
minority), dilution is naturally handled by the existing plasma composition
logic. By flagging them as "fast", the system shifts their pressure contribution
from the thermal calculation to the fast calculation while maintaining the
correct density displacement for quasineutrality.

Specifically, for fast ion species that are present in the impurity mixture,
their density is subtracted from the thermal impurity density when computing
:math:`p_{th,i}`. This ensures that the total number of particles is conserved
and that quasineutrality is maintained:

.. math::

  n_i Z_i + n_{imp,thermal} Z_{imp} + n_{tail} Z_{tail} = n_e

The framework currently supports fast ions from ICRH minority heating.
Extension to NBI and fusion-born fast ion populations is planned.

ITG Critical Gradient Stabilization
===================================
Fast ions can electrostatically stabilize Ion Temperature Gradient (ITG)
turbulence, raising the critical gradient threshold above which turbulent
transport is triggered. This effect can improve confinement |disiena2021|.

In TORAX, this stabilization is captured by modifying the normalized ion
temperature gradient :math:`R/L_{T_i}` *before* it is passed to the turbulent
transport surrogate (QLKNN or TGLFNN). This approach avoids the need to
retrain the transport model itself.

**Surrogate Model**

A neural network surrogate predicts the ITG threshold modification factor
:math:`h` as a function of local plasma and fast ion parameters. The model
architecture is:

.. math::

  h = 1 + \frac{n_{fi}}{n_e} \cdot \mathrm{MLP}\!\left(\hat{\mathbf{x}}\right)

where :math:`\hat{\mathbf{x}}` is the normalized input vector and MLP is a
multi-layer perceptron. The multiplicative structure guarantees :math:`h = 1`
(no correction) when the fast ion density is zero.

The input features are:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Feature
     - Description
   * - :math:`\hat{s}`
     - Magnetic shear
   * - :math:`q`
     - Safety factor
   * - :math:`n_{fi}/n_e`
     - Fast ion to electron density ratio
   * - :math:`T_{fi}/T_e`
     - Fast ion to electron temperature ratio
   * - :math:`R/L_{T_{fi}}`
     - Normalized logarithmic gradient of fast ion temperature

The training data are generated from QuaLiKiz gyrokinetic simulations across a
wide range of plasma conditions, with and without fast ions. Separate models are
trained for hydrogenic (H, D, T) and helium (He3, He4) fast ion species.
Future work can extend this calculation to non-Maxwellian distribution
functions and higher fidelity turbulence models.

**Application to Transport**

When enabled, the stabilization factor modifies the :math:`R/L_{T_i}` input
supplied to the transport surrogate:

.. math::

  \left(\frac{R}{L_{T_i}}\right)_{\mathrm{eff}} =
  \frac{R/L_{T_i}}{h}

Since :math:`h \geq 1` (stabilization raises the threshold), the effective
gradient seen by the transport model is reduced, leading to lower predicted
turbulent fluxes.

When multiple fast ion species are present, the total stabilization factor is:

.. math::

  h_{\mathrm{total}} = \prod_j h_j

where the product runs over all fast ion species.

**Multiplier**

A configurable multiplier :math:`m` scales the correction part of the
stabilization factor:

.. math::

  h_{\mathrm{adj}} = (h - 1) \cdot m + 1

This allows users to explore the sensitivity of results to the strength of the
ITG stabilization. Defaults to 2.0.

**Configuration**

The fast ion stabilization is controlled by three transport model parameters:

- ``fast_ion_stabilization`` (bool, time-varying): Enable/disable the
  correction. Default: ``False``.
- ``fast_ion_stabilization_model`` (dict): Mapping from species name to model
  name or path. If empty, default models are loaded from the model registry.
- ``fast_ion_stabilization_multiplier`` (float): Multiplier on the correction
  part of the stabilization factor. Default: ``2.0``. Default is currently 2.0
  since QuaLiKiz tends to underpredict the strength of this effect compared to
  higher fidelity gyrokinetics.

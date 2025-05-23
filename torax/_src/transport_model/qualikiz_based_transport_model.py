# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Base class and utils for Qualikiz-based models."""

import chex
from jax import numpy as jnp
from torax._src import constants as constants_module
from torax._src import state
from torax._src.geometry import geometry
from torax._src.physics import collisions
from torax._src.physics import psi_calculations
from torax._src.transport_model import quasilinear_transport_model


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(quasilinear_transport_model.DynamicRuntimeParams):
  """Shared parameters for Qualikiz-based models."""

  collisionality_multiplier: float
  avoid_big_negative_s: bool
  smag_alpha_correction: bool
  q_sawtooth_proxy: bool


# pylint: disable=invalid-name
@chex.dataclass(frozen=True)
class QualikizInputs(quasilinear_transport_model.QuasilinearInputs):
  """Inputs to Qualikiz-based models."""

  Z_eff_face: chex.Array
  q: chex.Array
  smag: chex.Array
  x: chex.Array
  Ti_Te: chex.Array
  log_nu_star_face: chex.Array
  normni: chex.Array
  alpha: chex.Array
  epsilon_lcfs: chex.Array
  ator: chex.Array
  exb_shear_rate: chex.Array

  # Also define the logarithmic gradients using standard QuaLiKiz notation.
  @property
  def Ati(self) -> chex.Array:
    return self.lref_over_lti

  @property
  def Ate(self) -> chex.Array:
    return self.lref_over_lte

  @property
  def Ane(self) -> chex.Array:
    return self.lref_over_lne

  @property
  def Ani0(self) -> chex.Array:
    return self.lref_over_lni0

  @property
  def Ani1(self) -> chex.Array:
    return self.lref_over_lni1


class QualikizBasedTransportModel(
    quasilinear_transport_model.QuasilinearTransportModel
):
  """Base class for Qualikiz-based transport models."""

  def _prepare_qualikiz_inputs(
      self,
      Z_eff_face: chex.Array,
      density_reference: chex.Numeric,
      transport: DynamicRuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> QualikizInputs:
    """Prepare Qualikiz inputs."""
    constants = constants_module.CONSTANTS

    # define radial coordinate as midplane average r
    # (typical assumption for transport models developed in circular geo)
    rmid = (geo.R_out - geo.R_in) * 0.5
    rmid_face = (geo.R_out_face - geo.R_in_face) * 0.5

    # gyrobohm diffusivity
    # (defined here with Lref=a_minor due to QLKNN training set normalization)
    chiGB = quasilinear_transport_model.calculate_chiGB(
        reference_temperature=core_profiles.T_i.face_value(),
        reference_magnetic_field=geo.B_0,
        reference_mass=core_profiles.A_i,
        reference_length=geo.a_minor,
    )

    # transport coefficients from the qlknn-hyper-10D model
    # (K.L. van de Plassche PoP 2020)

    # set up input vectors (all as jax.numpy arrays on face grid)

    # Calculate normalized logarithmic gradients
    normalized_logarithmic_gradients = quasilinear_transport_model.NormalizedLogarithmicGradients.from_profiles(
        core_profiles=core_profiles,
        radial_coordinate=rmid,
        reference_length=geo.R_major,
    )

    q = core_profiles.q_face

    # Due to QuaLikiz geometry assumptions, we need to calculate s with respect
    # to the midplane average, and not use the standard s_face from CoreProfiles
    smag = psi_calculations.calc_s_rmid(
        geo,
        core_profiles.psi,
    )

    # Inverse aspect ratio at LCFS.
    epsilon_lcfs = rmid_face[-1] / geo.R_major
    # Local normalized radius.
    x = rmid_face / rmid_face[-1]
    x = jnp.where(jnp.abs(x) < constants.eps, constants.eps, x)

    # Ion to electron temperature ratio
    Ti_Te = (
        core_profiles.T_i.face_value() / core_profiles.T_e.face_value()
    )

    # logarithm of normalized collisionality
    nu_star = collisions.calc_nu_star(
        geo=geo,
        core_profiles=core_profiles,
        density_reference=density_reference,
        Z_eff_face=Z_eff_face,
        collisionality_multiplier=transport.collisionality_multiplier,
    )
    log_nu_star_face = jnp.log10(nu_star)

    # calculate alpha for magnetic shear correction (see S. van Mulders NF 2021)
    alpha = quasilinear_transport_model.calculate_alpha(
        core_profiles=core_profiles,
        density_reference=density_reference,
        q=q,
        reference_magnetic_field=geo.B_0,
        normalized_logarithmic_gradients=normalized_logarithmic_gradients,
    )

    # to approximate impact of Shafranov shift. From van Mulders Nucl. Fusion
    # 2021.
    smag = jnp.where(
        transport.smag_alpha_correction,
        smag - alpha / 2,
        smag,
    )

    # very basic ad-hoc sawtooth model
    smag = jnp.where(
        jnp.logical_and(
            transport.q_sawtooth_proxy,
            q < 1,
        ),
        0.1,
        smag,
    )

    q = jnp.where(
        jnp.logical_and(
            transport.q_sawtooth_proxy,
            q < 1,
        ),
        1,
        q,
    )

    smag = jnp.where(
        jnp.logical_and(
            transport.avoid_big_negative_s,
            smag - alpha < -0.2,
        ),
        alpha - 0.2,
        smag,
    )
    normni = core_profiles.n_i.face_value() / core_profiles.n_e.face_value()

    # Step 1: Retrieve omega_tor
    omega_tor_face = core_profiles.omega_tor.face_value()
    domega_tor_drhon = core_profiles.omega_tor.face_grad()

    # Step 2: Calculate ator (normalized rotation gradient R_major/L_omega_tor)
    # L_omega_tor = - omega_tor_face / (domega_tor_drhon) (this is wrt rho_norm)
    # To make L_omega_tor have units of length, domega_tor_drhon needs to be domega_tor_dr
    # domega_tor_dr = domega_tor_drhon / geo.dr_drho_norm_face
    # L_omega_tor_inv = - (domega_tor_drhon / geo.dr_drho_norm_face) / (omega_tor_face + constants.eps)
    # ator = geo.R_major * L_omega_tor_inv
    # Using NormalizedLogarithmicGradients.from_one_profile is more robust as discussed
    # and directly gives R_major/L_omega if reference_length=R_major and radial_coordinate=rmid.
    # This calculates (R_major / omega_tor_cell) * (d_omega_tor / d_rmid_cell) on cell grid, then interpolated to face.
    omega_tor_norm_log_grad = quasilinear_transport_model.NormalizedLogarithmicGradients.from_one_profile(
        profile=core_profiles.omega_tor, # CellVariable
        radial_coordinate=rmid, # cell-centered radial coordinate
        reference_length=geo.R_major, # L_ref
    )
    ator = omega_tor_norm_log_grad.lref_over_lx_face # (L_ref/profile_face) * (d_profile/d_rad_coord)_face

    # Step 3: Calculate Ion Pressure Gradient (dPi/drho_norm)
    n_i_face = core_profiles.n_i.face_value()
    n_impurity_face = core_profiles.n_impurity.face_value() # Assuming single impurity for total ion pressure
    T_i_face = core_profiles.T_i.face_value()
    dn_i_drhon = core_profiles.n_i.face_grad()
    dn_impurity_drhon = core_profiles.n_impurity.face_grad()
    dT_i_drhon = core_profiles.T_i.face_grad()
    # prefactor converts n*T[keV] to pressure in SI (Pascals)
    # n is in [density_reference m^-3], T is in [keV]
    # So P = n * density_reference * T * keV2J
    prefactor = constants.keV2J * density_reference # core_profiles.density_reference is already included in n_i.value etc.
                                                  # No, n_i.value is normalized to density_reference. So multiply by it.
    # Correct prefactor: n_i (normalized) * density_reference (abs units) * T_i (keV) * constants.keV2J
    # However, core_profiles.n_i already has .value in units of [density_reference]
    # So n_i_face * T_i_face is in [density_reference * keV].
    # Multiply by constants.keV2J to get energy density units (J/m^3 if density_reference = 1 m^-3)
    # And multiply by density_reference to get actual pressure gradient.
    # Oh, dynamic_runtime_params_slice.numerics.density_reference is the scalar value.
    # core_profiles.n_i.value is n_i / density_reference. So n_i_abs = core_profiles.n_i.value * density_reference.
    # P_i_pascal = (n_i_abs + n_imp_abs) * T_i_kev * constants.keV2J
    # dp_i_drhon_face is d((n_i_abs + n_imp_abs) * T_i_kev * constants.keV2J) / drho_norm
    dp_i_drhon_face = (
        ( (n_i_face + n_impurity_face) * dT_i_drhon + (dn_i_drhon + dn_impurity_drhon) * T_i_face )
        * density_reference * constants.keV2J
    )

    # Step 4: Calculate Radial Electric Field (E_r)
    # dr/drho_norm on face grid
    dr_drho_norm_val = geo.dr_drho_norm_face
    # dp_i/dr = (dp_i/drho_norm) / (dr/drho_norm)
    dp_i_dr_face = dp_i_drhon_face / (dr_drho_norm_val + constants.eps)

    # B_theta approximation
    # Using rmid_face as the local minor radius 'r' in B_theta formula
    q_face_safe = jnp.where(jnp.abs(core_profiles.q_face) < constants.eps, constants.eps, core_profiles.q_face)
    # Ensure q_face_safe is not zero if core_profiles.q_face was exactly zero.
    q_face_safe = jnp.where(jnp.abs(q_face_safe) < constants.eps, constants.eps, q_face_safe)

    B_theta_face = geo.B_0 * rmid_face / (q_face_safe * geo.R_major + constants.eps)

    # V_phi_i term in Er equation: V_diamagnetic_tor + omega_tor * R
    # Here, the formula is given as (omega_tor * R_major) * B_theta
    # This implies the V_phi in the Er equation is omega_tor * R_major.
    Vphi_Btheta_term = (omega_tor_face * geo.R_major) * B_theta_face

    # Z_i should be on face. core_profiles.Z_i is on cell. core_profiles.Z_i_face exists.
    # n_i_face is already n_i_abs / density_reference. Need n_i_abs for Er.
    n_i_abs_face = n_i_face * density_reference

    # Avoid division by zero for Z_i_face * n_i_abs_face
    safe_denom_Er = core_profiles.Z_i_face * n_i_abs_face * constants.elementary_charge + constants.eps

    # Er = (1 / (Z_i * e * n_i)) * (dP_i / dr) - V_phi * B_theta (assuming V_theta B_phi = 0)
    Er_face = (1 / safe_denom_Er) * dp_i_dr_face - Vphi_Btheta_term

    # Step 5: Calculate ExB Shearing Rate (omega_ExB)
    # omega_ExB = | (R * B_p / B_t) * d/dr (Er / (R * B_p)) |
    # Let X = Er / (rmid_face * B_theta_face) - using rmid_face as local radius R in this term
    # Denominator for X
    X_denom = rmid_face * B_theta_face + constants.eps
    X = Er_face / X_denom

    # dX/drho_norm
    # Using jnp.gradient to get dX/d(rho_face_norm)
    # rho_face_norm are the coordinates for X which is on face grid.
    dX_drho_norm = jnp.gradient(X, geo.rho_face_norm) # geo.rho_face_norm are grid cell centers for face values.

    # dX/dr = (dX/drho_norm) / (dr/drho_norm)
    dX_dr = dX_drho_norm / (dr_drho_norm_val + constants.eps)

    # Prefactor for omega_ExB: (R_major * B_theta_face / B_0)
    # Using R_major as the 'R' in (R B_p / B_t)
    omega_ExB_prefactor = (geo.R_major * B_theta_face) / (geo.B_0 + constants.eps)
    exb_shear_rate = jnp.abs(omega_ExB_prefactor * dX_dr)

    return QualikizInputs(
        Z_eff_face=Z_eff_face,
        lref_over_lti=normalized_logarithmic_gradients.lref_over_lti,
        lref_over_lte=normalized_logarithmic_gradients.lref_over_lte,
        lref_over_lne=normalized_logarithmic_gradients.lref_over_lne,
        lref_over_lni0=normalized_logarithmic_gradients.lref_over_lni0,
        lref_over_lni1=normalized_logarithmic_gradients.lref_over_lni1,
        q=q, # This is the q used for other calculations, potentially modified by sawtooth proxy
        smag=smag, # This is smag used for other calculations
        x=x,
        Ti_Te=Ti_Te,
        log_nu_star_face=log_nu_star_face,
        normni=normni,
        chiGB=chiGB,
        Rmaj=geo.R_major,
        Rmin=geo.a_minor, # geo.a_minor is Lref for chiGB, also Rmin in QLK
        alpha=alpha,
        epsilon_lcfs=epsilon_lcfs,
        ator=ator,
        exb_shear_rate=exb_shear_rate,
    )

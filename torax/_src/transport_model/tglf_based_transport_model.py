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
"""Base class and utils for TGLF-based models."""

import dataclasses
from typing import Annotated

import chex
import jax
from jax import numpy as jnp
import pydantic
from torax._src import array_typing
from torax._src import constants
from torax._src import jax_utils
from torax._src import math_utils
from torax._src import state
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.physics import psi_calculations
from torax._src.physics import rotation
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import component
from torax._src.transport_model import config_base
from torax._src.transport_model import quasilinear_utils
from torax._src.transport_model import runtime_params as transport_runtime_params
from torax._src.transport_model import transport_coeffs


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(transport_runtime_params.ComponentRuntimeParams):
  """Shared parameters for TGLF-based models."""

  DV_effective: bool = jax.tree.static()
  An_min: float
  DV_effective_smooth_width: float
  use_rotation: bool = jax.tree.static()
  rotation_multiplier: float
  collisionality_multiplier: float
  max_normalized_collisionality: float


class TGLFBasedTransportModelConfig(
    config_base.ComponentTransportConfigBase
):
  """Base config for TGLF-family transport models."""

  DV_effective: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  An_min: pydantic.PositiveFloat = 0.05
  DV_effective_smooth_width: pydantic.NonNegativeFloat = 0.001
  use_rotation: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  rotation_multiplier: pydantic.NonNegativeFloat = 1.0
  collisionality_multiplier: float = 1.0
  max_normalized_collisionality: pydantic.PositiveFloat = float("inf")

  def build_runtime_params(self, t: chex.Numeric) -> RuntimeParams:
    base_params = super().build_runtime_params(t)
    return RuntimeParams(
        DV_effective=self.DV_effective,
        An_min=self.An_min,
        DV_effective_smooth_width=self.DV_effective_smooth_width,
        use_rotation=self.use_rotation,
        rotation_multiplier=self.rotation_multiplier,
        collisionality_multiplier=self.collisionality_multiplier,
        max_normalized_collisionality=self.max_normalized_collisionality,
        **vars(base_params),
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TGLFInputs:
  r"""Dimensionless inputs to TGLF-based models.

  See https://gacode.io/tglf/tglf_table.html for definitions.

  In this interface we use the following species numbering for TGLF: species 1 =
  electrons, species 2 = main ion, species 3 = impurity.
  For example, TGLF variables RLTS_1, RLTS_2, and RLTS_3 are:
    RLTS_1 = lref / lte,
    RLTS_2 = lref / lti,
    RLTS_3 = lref / ltimp.

  Attributes:
    ZS_1: Charge of electrons (species 1).
    MASS_1: Ratio of electron mass to deuterium mass.
    TAUS_1: Ratio of electron temperature to electron temperature (species 1).
    AS_1: Ratio of electron density to electron density (species 1).
    ZS_2: Charge of main ion (species 2).
    MASS_2: Ratio of main ion mass to deuterium mass.
    TAUS_2: Ratio of ion temperature to electron temperature (species 2).
    AS_2: Ratio of main ion density to electron density.
    ZS_3: Charge of impurity (species 3).
    MASS_3: Ratio of impurity mass to deuterium mass.
    TAUS_3: Ratio of impurity temperature to electron temperature (species 3).
      As TORAX does not track impurity temp separately, set to T_i / T_e.
    AS_3: Ratio of impurity density to electron density.
    RLNS_1: Electron density gradient normalisation.
    RLNS_2: Main ion density gradient normalisation.
    RLNS_3: Impurity density gradient normalisation.
    RLTS_1: Electron temperature gradient normalisation.
    RLTS_2: Main ion temperature gradient normalisation.
    RLTS_3: Impurity temperature gradient normalisation.
    RMIN_LOC: Flux surface centroid minor radius normalized by minor radius.
    RMAJ_LOC: Flux surface centroid major radius normalized by minor radius.
    DRMAJDX_LOC: Gradient of the major radius centroid w.r.t minor radius.
    Q_LOC: Safety factor.
    Q_PRIME_LOC: Safety factor gradient.
    XNUE: Collision frequency.
    DEBYE: Debye length length.
    KAPPA_LOC: Plasma elongation.
    S_KAPPA_LOC: Elongation shear.
    DELTA_LOC: Plasma triangularity.
    S_DELTA_LOC: Triangularity shear.
    BETAE: Electron beta.
    P_PRIME_LOC: Pressure gradient.
    ZEFF: Effective charge Z_eff.
    VEXB_SHEAR: ExB shear.
    Q_GB: TGLF heat flux normalisation factor.
    GAMMA_GB: TGLF particle flux normalisation factor.
  """

  ZS_1: array_typing.FloatVectorFace
  MASS_1: array_typing.FloatVectorFace
  TAUS_1: array_typing.FloatVectorFace
  AS_1: array_typing.FloatVectorFace

  ZS_2: array_typing.FloatVectorFace
  MASS_2: array_typing.FloatVectorFace
  TAUS_2: array_typing.FloatVectorFace
  AS_2: array_typing.FloatVectorFace

  ZS_3: array_typing.FloatVectorFace
  MASS_3: array_typing.FloatVectorFace
  TAUS_3: array_typing.FloatVectorFace
  AS_3: array_typing.FloatVectorFace

  RLNS_1: array_typing.FloatVectorFace
  RLNS_2: array_typing.FloatVectorFace
  RLNS_3: array_typing.FloatVectorFace
  RLTS_1: array_typing.FloatVectorFace
  RLTS_2: array_typing.FloatVectorFace
  RLTS_3: array_typing.FloatVectorFace

  RMIN_LOC: array_typing.FloatVectorFace
  RMAJ_LOC: array_typing.FloatVectorFace
  DRMAJDX_LOC: array_typing.FloatVectorFace
  Q_LOC: array_typing.FloatVectorFace
  Q_PRIME_LOC: array_typing.FloatVectorFace
  XNUE: array_typing.FloatVectorFace
  DEBYE: array_typing.FloatVectorFace
  KAPPA_LOC: array_typing.FloatVectorFace
  S_KAPPA_LOC: array_typing.FloatVectorFace
  DELTA_LOC: array_typing.FloatVectorFace
  S_DELTA_LOC: array_typing.FloatVectorFace
  BETAE: array_typing.FloatVectorFace
  P_PRIME_LOC: array_typing.FloatVectorFace
  ZEFF: array_typing.FloatVectorFace
  VEXB_SHEAR: array_typing.FloatVectorFace

  Q_GB: array_typing.FloatVectorFace
  GAMMA_GB: array_typing.FloatVectorFace


class TGLFBasedTransportModel(component.ComponentTransportModel):
  """Base class for TGLF-based transport models."""

  def _prepare_tglf_inputs(
      self,
      transport: RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      two_point_mask: array_typing.BoolVectorFace,
  ) -> TGLFInputs:
    """Construct a TGLFInputs object from the TORAX state.

    Normalisation and coordinate conventions:
      TGLF values are normalised with respect to:
        - Minor radius at the LCFS (a) in m
        - B_unit in T
          (defined at https://gacode.io/geometry.html#effective-field)
        - The ion sound speed in m/s
          (defined at https://gacode.io/cgyro.html#id10)
      The radial coordinate is the minor radius of the flux surface (r) in m.
      Psi_TGLF is Psi_TORAX / 2π.

    Shortcomings:
      Currently, geometry parameters are taken directly from the global
      profiles. For complete accuracy, a Miller (or s-α) geometry fit routine
      should instead be implemented to calculate *local* geometry parameters.

      At present, only a subset of TGLF inputs have been implemented.

      This model is under construction.

    References:
      The main points of reference for this implementation are:
        - The TGLF documentation (https://gacode.io/tglf.html)
        - General GACode documentation (https://gacode.io/input_gacode.html)
        - The CGYRO documentation (https://gacode.io/cgyro.html)

      If something isn't documented in TGLF, it will use the same definitions as
      CGYRO and other GACode libraries. The implementation in this function uses
      SI units and definitions throughout unless specified otherwise; be careful
      when comparing with the TGLF docs, as they use a combination of SI, keV,
      1e19m^-3, and Gaussian CGS.

    Args:
      transport: Runtime parameters for the transport model.
      geo: Geometric parameters of the tokamak.
      core_profiles: Core plasma profiles (e.g., temperatures, densities, q).
      two_point_mask: Boolean face mask indicating where face gradients are
        calculated with 2-point central differences instead of 3-point.

    Returns:
      A `TGLFInputs` dataclass containing dimensionless inputs required by
      TGLF-based models.
    """
    T_e_J = core_profiles.T_e.face_value() * constants.CONSTANTS.keV_to_J
    n_e = core_profiles.n_e.face_value()

    # Reference values used for TGLF-specific normalisation
    # - 'a' in TGLF means the minor radius at the LCFS
    # - 'r' in TGLF means the flux surface centroid minor radius. Gradients are
    #   taken w.r.t. r
    #   https://gacode.io/tglf/tglf_list.html#rmin-loc
    # - B_unit = 1/r d(psi_tor/2π)/dr = q/2πr dpsi/dr, noting that psi_tor is
    #   psi/2π
    #   https://gacode.io/geometry.html#effective-field
    #   https://gacode.io/cgyro.html#faq
    # - c_s (ion sound speed)
    #   https://gacode.io/cgyro.html#id10
    m_D = (
        constants.ION_PROPERTIES_DICT["D"].A * constants.CONSTANTS.m_amu
    )  # Mass of deuterium [kg]
    c_s = (T_e_J / m_D) ** 0.5  # T_e [J], m_D [kg], gives c_s in [m/s]
    a = geo.a_minor  # Device minor radius at LCFS [m]
    r = geo.r_mid_face  # Flux surface centroid minor radius [m]
    # On the magnetic axis (index 0), both r and dpsi/dr vanish (0/0), so use
    # the first off-axis face ratio
    # Use face_grad to correctly handle constraints on the psi CellVariable.
    dpsi_dr = core_profiles.psi.face_grad(
        x=geo.r_mid,
        x_left=r[0],
        x_right=r[-1],
        two_point_mask=two_point_mask,
    )
    dpsi_dr_over_r = jnp.concatenate([
        dpsi_dr[1:2] / r[1:2],
        dpsi_dr[1:] / r[1:],
    ])
    # Note: psi_TGLF is psi_TORAX/2π
    B_unit = core_profiles.q_face * dpsi_dr_over_r / (2 * jnp.pi)

    # Mass profiles.
    n_faces = len(geo.rho_face_norm)
    m_i_over_m_D = (
        core_profiles.A_i / constants.ION_PROPERTIES_DICT["D"].A
    ) * jnp.ones(n_faces)
    m_imp_over_m_D = (
        core_profiles.A_impurity_face / constants.ION_PROPERTIES_DICT["D"].A
    )

    # Ion gyroradius
    # TODO(b/502473098): Currently, q_e has to be outside of the safe_divide to
    # avoid being swamped by the eps in the denominator.
    rho_s = (
        math_utils.safe_divide(
            num=m_D * c_s,
            denom=B_unit,
            eps=1e-7,
        )
        / constants.CONSTANTS.q_e
    )

    # Debye length
    # https://gacode.io/tglf/tglf_list.html#debye
    # - In the TGLF docs, the prefactor of 743.0 comes from a combination of the
    #   constants below plus being in CGS units. Below is the SI version.
    normalized_debye = math_utils.safe_divide(
        num=(
            (constants.CONSTANTS.epsilon_0 / constants.CONSTANTS.q_e)
            * (core_profiles.T_e.face_value() * 1e3)  # keV -> eV
            / n_e
        )
        ** 0.5,
        denom=rho_s,
        eps=1e-7,
    )

    # Temperature ratio
    T_i_over_T_e = (
        core_profiles.T_i.face_value() / core_profiles.T_e.face_value()
    )

    # Ion dilution
    n_i_over_n_e = (
        core_profiles.n_i.face_value() / core_profiles.n_e.face_value()
    )
    n_impurity_over_n_e = (
        core_profiles.n_impurity.face_value() / core_profiles.n_e.face_value()
    )

    # Dimensionless gradients
    normalized_log_gradients = (
        quasilinear_utils.NormalizedLogarithmicGradients.from_profiles(
            core_profiles=core_profiles,
            radial_coordinate=geo.r_mid,  # On the cell grid
            radial_face_coordinate=geo.r_mid_face,
            reference_length=a,
            two_point_mask=two_point_mask,
        )
    )

    # Dimensionless electron-electron collision frequency = nu_ee / (c_s/a)
    # https://gacode.io/tglf/tglf_list.html#xnue
    # https://gacode.io/cgyro/cgyro_list.html#nu-ee
    # - In the TGLF docs, XNUE is labelled as electron-ion collision
    #   frequency. It is believed that it should actually be the electron-
    #   electron collision frequency, see
    #   https://pyrokinetics.readthedocs.io/en/latest/user_guide/collisions.html#tglf
    # - In the TGLF docs, nu_ee is shown in CGS units. Below is the SI version.
    # - TGLF uses a slightly different calculation of log_Lambda to those in
    #   Wesson, namely
    #       log_Lambda = (
    #         24.0 - 0.5 * jnp.log(n_e_1e19 * 1e13) + jnp.log(T_e_keV * 1e3)
    #       )
    #   https://github.com/gafusion/gacode/blob/740bb2dc811bc1ad38be65ea4b87330995931305/cgyro/src/cgyro_make_profiles.F90#L145
    #   This is different to Wesson by about ~0.5%. Below, we use the TLGF
    #   version, but with n_e [m^-3] and T_e [J].
    log_Lambda = 74.2 - 0.5 * jnp.log(n_e) + jnp.log(T_e_J)

    # ν_ee = (sqrt(2) n_e q_e^4 lnΛ) / (16 π ε_0^2 m_e^0.5 T_e^1.5)
    # Compute via the log for stability
    log_nu_ee = (
        0.5 * jnp.log(2)
        + jnp.log(n_e)
        + 4 * jnp.log(constants.CONSTANTS.q_e)
        + jnp.log(log_Lambda)
        - jnp.log(16)
        - jnp.log(jnp.pi)
        - 2 * jnp.log(constants.CONSTANTS.epsilon_0)
        - 0.5 * jnp.log(constants.CONSTANTS.m_e)
        - 1.5 * jnp.log(T_e_J)
    )
    normalized_nu_ee = (
        jnp.exp(log_nu_ee) / (c_s / a) * transport.collisionality_multiplier
    )
    # Cap collision frequency to mitigate unreliable transport predictions
    # at high collisionality.
    normalized_nu_ee = jnp.minimum(
        normalized_nu_ee, transport.max_normalized_collisionality
    )

    # Dimensionless safety factor shear
    # https://gacode.io/tglf/tglf_list.html#tglf-q-prime-loc
    # - TGLF definition is q^2 a^2 s / r^2
    #   where s = r/q dq/dr
    #   (https://gacode.io/cgyro/cgyro_list.html#s)
    # - r_mid is zero on axis, so use safe_divide to avoid division by zero.
    q_prime = math_utils.safe_divide(
        num=psi_calculations.calc_s_rmid(geo, core_profiles.psi)
        * core_profiles.q_face**2
        * a**2,
        denom=r**2,
        eps=1e-7,
    )

    # Dimensionless pressure gradient
    # https://gacode.io/tglf/tglf_list.html#tglf-p-prime-loc
    # - In the TGLF docs, p_prime equation is shown in CGS units, this is the SI
    #   version
    # - 8 * pi factor missing since TGLF internally operates on it using
    #   beta/(8*pi)
    p_prime = math_utils.safe_divide(
        num=1.0e-7
        * core_profiles.pressure_thermal_total.face_grad(
            x=geo.r_mid,
            x_left=r[0],
            x_right=r[-1],
            two_point_mask=two_point_mask,
        )
        * core_profiles.q_face
        * a**2,
        denom=r * B_unit**2,
        eps=1e-7,
    )

    # Electron beta
    # https://gacode.io/tglf/tglf_list.html#tglf-betae
    # https://gacode.io/cgyro.html#faq
    # https://gacode.io/cgyro/cgyro_list.html#betae-unit
    # - In the TGLF docs, beta_e equation shown in CGS units, this is the SI
    #   version
    beta_e = math_utils.safe_divide(
        num=2 * constants.CONSTANTS.mu_0 * n_e * T_e_J,
        denom=B_unit**2,
        eps=1e-7,
    )

    # Major radius shear = drmaj/drmin, where 'rmaj' is the flux surface
    # centroid major radius and 'rmin' the flux surface centroid minor radius
    # https://gacode.io/tglf/tglf_list.html#tglf-drmajdx-loc
    r_major = (geo.R_in_face + geo.R_out_face) / 2
    dr_major = jnp.asarray(jnp.gradient(r_major, r))

    # Dimensionless elongation shear = r/kappa dkappa/dr
    # https://gacode.io/tglf/tglf_list.html#tglf-s-kappa-loc
    kappa = geo.elongation_face
    kappa_shear = r / kappa * jnp.asarray(jnp.gradient(kappa, r))

    # Dimensionless triangularity shear = r ddelta/dr
    # https://gacode.io/tglf/tglf_list.html#tglf-s-delta-loc
    delta_shear = r * jnp.asarray(jnp.gradient(geo.delta_face, r))

    # Output normalisations
    # https://gacode.io/tglf/tglf_table.html#id7
    # https://gacode.io/cgyro.html#id10
    GB = c_s * (rho_s / a) ** 2
    Q_GB = n_e * T_e_J * GB  # [W/m^2]
    Gamma_GB = n_e * GB

    # Normalized ExB shear rate, on the face grid.
    # https://gacode.io/tglf/tglf_list.html#vexb-shear
    # VEXB_SHEAR = -sign(I_p) (r / |q|) d(omega_0)/dr (a / c_s), where GACODE
    # normalizes rotation to B_pol (https://gacode.io/rotation.html):
    #   omega_0 = -dPhi/dpsi = E_r / (R B_pol).
    # For pure toroidal rotation, omega_0 is the toroidal angular velocity.
    def _get_v_ExB_shear(
        core_profiles: state.CoreProfiles,
        geo: geometry.Geometry,
        two_point_mask: array_typing.BoolVectorFace,
    ):
      rotation_output = rotation.calculate_rotation(
          psi=core_profiles.psi,
          n_i=core_profiles.n_i,
          Z_i_face=core_profiles.Z_i_face,
          toroidal_angular_velocity=core_profiles.toroidal_angular_velocity,
          poloidal_velocity=core_profiles.poloidal_velocity,
          pressure_total_i=core_profiles.pressure_total_i,
          geo=geo,
      )
      # On the magnetic axis (index 0), both E_r and B_pol vanish (0/0), so
      # evaluate omega_0 on the off-axis faces [1:] and reuse the first
      # off-axis face value on axis.
      B_pol_off_axis = jnp.sqrt(
          psi_calculations.calc_bpol_squared(geo, core_profiles.psi)[1:]
      )
      R_B_pol_off_axis = jnp.maximum(
          geo.R_major_profile_face[1:] * B_pol_off_axis,
          constants.CONSTANTS.eps,
      )
      omega_0_off_axis = rotation_output.Er[1:] / R_B_pol_off_axis
      omega_0_face = jnp.concatenate([omega_0_off_axis[:1], omega_0_off_axis])
      cv = cell_variable.CellVariable(
          value=geometry.face_to_cell(omega_0_face),
          face_centers=geo.rho_face_norm,
          right_face_constraint=omega_0_face[-1],
          right_face_grad_constraint=None,
          left_face_constraint=None,
          left_face_grad_constraint=jnp.array(0.0, dtype=jax_utils.get_dtype()),
      )
      grad = cv.face_grad(
          x=geo.r_mid,
          x_left=r[0],
          x_right=r[-1],
          two_point_mask=two_point_mask,
      )
      v_ExB_shear = -(
          jnp.sign(core_profiles.Ip_profile_face)
          * (r / jnp.abs(core_profiles.q_face))
          * grad
          * (a / c_s)
      )
      v_ExB_shear = v_ExB_shear * transport.rotation_multiplier
      return v_ExB_shear

    # TODO(b/381199010): Validate against existing frameworks.
    if transport.use_rotation:
      v_ExB_shear = _get_v_ExB_shear(
          core_profiles,
          geo,
          two_point_mask=two_point_mask,
      )
    else:
      v_ExB_shear = jnp.zeros_like(core_profiles.q_face)

    smag = psi_calculations.calc_s_rmid(geo, core_profiles.psi)
    lref_over_lti = quasilinear_utils.apply_fast_ion_stabilization(
        core_profiles=core_profiles,
        smag=smag,
        q=jnp.asarray(core_profiles.q_face),
        normalized_logarithmic_gradients=normalized_log_gradients,
        transport=transport,
    )

    Z_e_face = -1.0 * jnp.ones(n_faces)
    m_e_over_m_D = (
        constants.CONSTANTS.m_e
        / (constants.CONSTANTS.m_amu * constants.ION_PROPERTIES_DICT["D"].A)
    ) * jnp.ones(n_faces)
    T_e_over_T_e = 1.0 * jnp.ones(n_faces)
    n_e_over_n_e = 1.0 * jnp.ones(n_faces)
    T_imp_over_T_e = T_i_over_T_e

    return TGLFInputs(
        ZS_1=Z_e_face,
        MASS_1=m_e_over_m_D,
        TAUS_1=T_e_over_T_e,
        AS_1=n_e_over_n_e,
        ZS_2=core_profiles.Z_i_face,
        MASS_2=m_i_over_m_D,
        TAUS_2=T_i_over_T_e,
        AS_2=n_i_over_n_e,
        ZS_3=core_profiles.Z_impurity_face,
        MASS_3=m_imp_over_m_D,
        TAUS_3=T_imp_over_T_e,
        AS_3=n_impurity_over_n_e,
        RLNS_1=normalized_log_gradients.lref_over_lne,
        RLNS_2=normalized_log_gradients.lref_over_lni0,
        RLNS_3=normalized_log_gradients.lref_over_lni1,
        RLTS_1=normalized_log_gradients.lref_over_lte,
        RLTS_2=lref_over_lti,
        RLTS_3=lref_over_lti,
        RMIN_LOC=r / a,
        RMAJ_LOC=r_major / a,
        DRMAJDX_LOC=dr_major,
        Q_LOC=core_profiles.q_face,
        Q_PRIME_LOC=q_prime,
        XNUE=normalized_nu_ee,
        DEBYE=normalized_debye,
        KAPPA_LOC=kappa,
        S_KAPPA_LOC=kappa_shear,
        DELTA_LOC=geo.delta_face,
        S_DELTA_LOC=delta_shear,
        BETAE=beta_e,
        P_PRIME_LOC=p_prime,
        ZEFF=core_profiles.Z_eff_face,
        Q_GB=Q_GB,
        GAMMA_GB=Gamma_GB,
        VEXB_SHEAR=v_ExB_shear,
    )

  def _make_transport_coeffs(
      self,
      electron_heat_flux_GB: jax.Array,
      ion_heat_flux_GB: jax.Array,
      electron_particle_flux_GB: jax.Array,
      tglf_inputs: TGLFInputs,
      transport: RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      two_point_mask: array_typing.BoolVectorFace,
  ) -> transport_coeffs.TransportCoeffs:
    # Denormalised TGLF output fluxes.
    Q_e = electron_heat_flux_GB * tglf_inputs.Q_GB  # [W/m^2]
    Q_i = ion_heat_flux_GB * tglf_inputs.Q_GB  # [W/m^2]
    Gamma_e = electron_particle_flux_GB * tglf_inputs.GAMMA_GB  # [s^-1/m^2]

    # Total thermal power.
    dV_drho = geo.vpr_face / geo.rho_b
    P_e = Q_e * dV_drho  # [W]
    P_i = Q_i * dV_drho  # [W]

    # Convert from power to chi.
    # Note: g1/vpr = ⟨(∇ρₙ)²⟩ ∂V/∂ρₙ, and has units [m].
    dT_e_drhon = (
        core_profiles.T_e.face_grad(two_point_mask=two_point_mask)
        * constants.CONSTANTS.keV_to_J
    )
    dT_i_drhon = (
        core_profiles.T_i.face_grad(two_point_mask=two_point_mask)
        * constants.CONSTANTS.keV_to_J
    )
    chi_e = math_utils.safe_divide(
        num=-P_e,
        denom=core_profiles.n_e.face_value()
        * dT_e_drhon
        * geo.g1_over_vpr_face,
        eps=1e-7,
    )
    chi_i = math_utils.safe_divide(
        num=-P_i,
        denom=core_profiles.n_i.face_value()
        * dT_i_drhon
        * geo.g1_over_vpr_face,
        eps=1e-7,
    )

    if transport.DV_effective:
      d_face_el, v_face_el = quasilinear_utils.calculate_dv_effective(
          particle_flux_SI=Gamma_e,
          normalized_particle_flux=electron_particle_flux_GB,
          n_e=core_profiles.n_e,
          geo=geo,
          gradient_reference_length=geo.R_major,
          An_min=transport.An_min,
          DV_effective_smooth_width=transport.DV_effective_smooth_width,
          two_point_mask=two_point_mask,
      )
    else:
      # Scaled D approach. Scale electron diffusivity to electron heat
      # conductivity (this has some physical motivations),
      # and set convection to then match total particle transport.
      d_face_el, v_face_el = quasilinear_utils.calculate_d_scaled(
          particle_flux_SI=Gamma_e,
          chi_face_el=chi_e,
          n_e=core_profiles.n_e,
          geo=geo,
          two_point_mask=two_point_mask,
      )

    return transport_coeffs.TransportCoeffs(
        chi_face_ion=chi_i,
        chi_face_el=chi_e,
        d_face_el=d_face_el,
        v_face_el=v_face_el,
    )

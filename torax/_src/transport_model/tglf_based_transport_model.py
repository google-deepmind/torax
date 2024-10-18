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

import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import state
from torax._src.geometry import geometry
from torax._src.physics import psi_calculations
from torax._src.transport_model import quasilinear_transport_model


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(quasilinear_transport_model.RuntimeParams):
  """Shared parameters for TGLF-based models.

  Left here as a placeholder for future TGLF settings."""

  pass


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TGLFInputs(quasilinear_transport_model.QuasilinearInputs):
  r"""Dimensionless inputs to TGLF-based models.
  See https://gacode.io/tglf/tglf_table.html for definitions.

  Attributes:
    Ti_over_Te: Ratio of ion temperature to electron temperature.
    r_minor: Flux surface centroid minor radius.
    dr_major: Gradient of the flux surface centroid major radius with respect to
      the minor radius (:math:`dr_{major}/dr_{minor}`).
    q: The safety factor.
    q_prime: Magnetic shear, defined as :math:`\frac{q^2 a^2 s}{r^2}`.
    nu_ee: The electron-electron collision frequency.
    kappa: Plasma elongation.
    kappa_shear: Shear in elongation, defined as
      :math:`\frac{r}{\kappa} \frac{d\kappa}{dr}`.
    delta: Plasma triangularity.
    delta_shear: Shear in triangularity, defined as :math:`r\frac{d\delta}{dr}`.
    beta_e: Electron pressure normalized by TGLF's :math:`B_\mathrm{unit}`.
    Zeff: Effective charge.
  """

  Ti_over_Te: array_typing.FloatVectorFace
  r_minor: array_typing.FloatVectorFace
  dr_major: array_typing.FloatVectorFace
  q: array_typing.FloatVectorFace
  q_prime: array_typing.FloatVectorFace
  nu_ee: array_typing.FloatVectorFace
  kappa: array_typing.FloatVectorFace
  kappa_shear: array_typing.FloatVectorFace
  delta: array_typing.FloatVectorFace
  delta_shear: array_typing.FloatVectorFace
  beta_e: array_typing.FloatVectorFace
  Zeff: array_typing.FloatVectorFace

  # Also define all the TGLF notations for the variables
  @property
  def TAUS_2(self) -> array_typing.FloatVectorFace:
    return self.Ti_over_Te

  @property
  def DRMAJDX_LOC(self) -> array_typing.FloatVectorFace:
    return self.dr_major

  @property
  def Q_LOC(self) -> array_typing.FloatVectorFace:
    return self.q

  @property
  def Q_PRIME_LOC(self) -> array_typing.FloatVectorFace:
    return self.q_prime

  @property
  def XNUE(self) -> array_typing.FloatVectorFace:
    return self.nu_ee

  @property
  def KAPPA_LOC(self) -> array_typing.FloatVectorFace:
    return self.kappa

  @property
  def S_KAPPA_LOC(self) -> array_typing.FloatVectorFace:
    return self.kappa_shear

  @property
  def DELTA_LOC(self) -> array_typing.FloatVectorFace:
    return self.delta

  @property
  def S_DELTA_LOC(self) -> array_typing.FloatVectorFace:
    return self.delta_shear

  @property
  def BETAE(self) -> array_typing.FloatVectorFace:
    return self.beta_e

  @property
  def ZEFF(self) -> array_typing.FloatVectorFace:
    return self.Zeff

  @property
  def RLNS_1(self) -> array_typing.FloatVectorFace:
    return self.lref_over_lne

  @property
  def RLTS_1(self) -> array_typing.FloatVectorFace:
    return self.lref_over_lte

  @property
  def RLTS_2(self) -> array_typing.FloatVectorFace:
    return self.lref_over_lti

  @property
  def RMIN_LOC(self) -> array_typing.FloatVectorFace:
    return self.r_minor

  @property
  def VEXB_SHEAR(self) -> array_typing.FloatVectorFace:
    # TODO: Replace with real values once rotation is added to TORAX
    return jnp.zeros_like(self.r_minor)


class TGLFBasedTransportModel(
    quasilinear_transport_model.QuasilinearTransportModel
):
  """Base class for TGLF-based transport models."""

  def _prepare_tglf_inputs(
      self,
      transport: RuntimeParams,  # pylint: disable=unused-argument
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> TGLFInputs:
    """Construct a TGLFInputs object from the TORAX state.

    Normalisation and coordinate conventions:
      TGLF values are normalised with respect to:
        - Minor radius at the LCFS (a) in m
        - B_unit in T (defined at https://gacode.io/geometry.html#effective-field)
        - The ion sound speed in m/s (defined at https://gacode.io/cgyro.html#id10)
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
    """
    T_e = core_profiles.T_e.face_value() * constants.CONSTANTS.keV2J
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
    m_D_amu = constants.ION_PROPERTIES_DICT['D'].A  # Mass of deuterium [amu]
    m_D = m_D_amu * constants.CONSTANTS.mp  # Mass of deuterium [kg]
    c_s = (T_e / m_D) ** 0.5  # T_e [J], m_D [kg], gives c_s in [m/s]
    a = geo.a_minor  # Device minor radius at LCFS [m]
    r = geo.r_mid_face  # Flux surface centroid minor radius [m]
    B_unit = (
        core_profiles.q_face
        / (2 * jnp.pi * r)  # Note: psi_TGLF is psi_TORAX/2π
        * jnp.gradient(core_profiles.psi.face_value(), r)
    )

    # Temperature ratio
    Ti_over_Te = core_profiles.T_i.face_value() / core_profiles.T_e.face_value()

    # Dimensionless gradients
    normalized_log_gradients = quasilinear_transport_model.NormalizedLogarithmicGradients.from_profiles(
        core_profiles=core_profiles,
        radial_coordinate=geo.r_mid,  # On the cell grid
        reference_length=a,
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
    log_Lambda = 74.2 - 0.5 * jnp.log(n_e) + jnp.log(T_e)

    # ν_ee = (sqrt(2) n_e q_e^4 lnΛ) / (16 π ε_0^2 m_e^0.5 T_e^1.5)
    # Compute via the log for stability
    log_nu_ee = (
        0.5 * jnp.log(2)
        + jnp.log(n_e)
        + 4 * jnp.log(constants.CONSTANTS.qe)
        + jnp.log(log_Lambda)
        - jnp.log(16)
        - jnp.log(jnp.pi)
        - 2 * jnp.log(constants.CONSTANTS.epsilon0)
        - 0.5 * jnp.log(constants.CONSTANTS.me)
        - 1.5 * jnp.log(T_e)
    )

    normalized_nu_ee = jnp.exp(log_nu_ee) / (c_s / a)

    # Dimensionless safety factor shear
    # https://gacode.io/tglf/tglf_list.html#tglf-q-prime-loc
    # - TGLF definition is q^2 a^2 s / r^2
    #   where s = r/q dq/dr
    #   (https://gacode.io/cgyro/cgyro_list.html#s)
    q_prime = (
        psi_calculations.calc_s_rmid(geo, core_profiles.psi)
        * core_profiles.q_face**2
        * a**2
        / r**2
    )

    # Electron beta
    # https://gacode.io/tglf/tglf_list.html#tglf-betae
    # https://gacode.io/cgyro.html#faq
    # https://gacode.io/cgyro/cgyro_list.html#betae-unit
    # - In the TGLF docs, beta_e equation shown in CGS units, this is the SI
    #   version
    beta_e = 2 * constants.CONSTANTS.mu0 * n_e * T_e / B_unit**2

    # Major radius shear = drmaj/drmin, where 'rmaj' is the flux surface
    # centroid major radius and 'rmin' the flux surface centroid minor radius
    # https://gacode.io/tglf/tglf_list.html#tglf-drmajdx-loc
    r_major = (geo.R_in_face + geo.R_out_face) / 2
    dr_major = jnp.gradient(r_major, r)

    # Dimensionless elongation shear = r/kappa dkappa/dr
    # https://gacode.io/tglf/tglf_list.html#tglf-s-kappa-loc
    kappa = geo.elongation_face
    kappa_shear = r / kappa * jnp.gradient(kappa, r)

    # Dimensionless triangularity shear = r ddelta/dr
    # https://gacode.io/tglf/tglf_list.html#tglf-s-delta-loc
    delta_shear = r * jnp.gradient(geo.delta_face, r)

    # Gyrobohm diffusivity
    # https://gacode.io/tglf/tglf_table.html#id7
    # https://gacode.io/cgyro.html#id10
    # - TGLF uses the same normalisation as CGYRO.
    # - The extra c^2 in the docs comes from CGS Gaussian units when
    #   calculating \rho_s
    chiGB = quasilinear_transport_model.calculate_chiGB(
        reference_temperature=core_profiles.T_e.face_value(),  # [keV]
        reference_magnetic_field=B_unit,
        reference_mass=m_D_amu,  # [amu]
        reference_length=a,
    )

    return TGLFInputs(
        # From QuasilinearInputs
        chiGB=chiGB,
        Rmin=geo.a_minor,
        Rmaj=geo.R_major,
        lref_over_lti=normalized_log_gradients.lref_over_lti,
        lref_over_lte=normalized_log_gradients.lref_over_lte,
        lref_over_lne=normalized_log_gradients.lref_over_lne,
        lref_over_lni0=normalized_log_gradients.lref_over_lni0,
        lref_over_lni1=normalized_log_gradients.lref_over_lni1,
        # From TGLFInputs
        Ti_over_Te=Ti_over_Te,
        r_minor=r / a,
        dr_major=dr_major,
        q=core_profiles.q_face,
        q_prime=q_prime,
        nu_ee=normalized_nu_ee,
        kappa=kappa,
        kappa_shear=kappa_shear,
        delta=geo.delta_face,
        delta_shear=delta_shear,
        beta_e=beta_e,
        Zeff=core_profiles.Z_eff_face,
    )

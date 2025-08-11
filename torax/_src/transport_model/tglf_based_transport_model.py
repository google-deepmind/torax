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

import chex
import jax
from jax import numpy as jnp
from torax._src import constants as constants_module
from torax._src import state
from torax._src.geometry import geometry
from torax._src.physics import psi_calculations
from torax._src.transport_model import quasilinear_transport_model


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DynamicRuntimeParams(quasilinear_transport_model.DynamicRuntimeParams):
  """Shared parameters for TGLF-based models."""

  pass


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TGLFInputs(quasilinear_transport_model.QuasilinearInputs):
  r"""Dimensionless inputs to TGLF-based models.

  See https://gafusion.github.io/doc/tglf/tglf_table.html for definitions.
  """

  # Ti/Te
  Ti_over_Te: chex.Array
  # Flux surface centroid minor radius
  r_minor: chex.Array
  # Flux surface centroid major radius gradient, drmajor/dr
  dr_major: chex.Array
  # q
  q: chex.Array
  # q^2 a^2 s / r^2
  q_prime: chex.Array
  # nu_ee (see note in prepare_tglf_inputs)
  nu_ee: chex.Array
  # Elongation, kappa
  kappa: chex.Array
  # Shear in elongation, r/kappa dkappa/dr
  kappa_shear: chex.Array
  # Triangularity, delta
  delta: chex.Array
  # Shear in triangularity, r ddelta/dr
  delta_shear: chex.Array
  # Electron pressure defined w.r.t B_unit
  beta_e: chex.Array
  # Effective charge
  Zeff: chex.Array

  # Also define all the TGLF notations for the variables
  @property
  def TAUS_2(self):
    return self.Ti_over_Te

  @property
  def DRMAJDX_LOC(self):
    return self.dr_major

  @property
  def Q_LOC(self):
    return self.q

  @property
  def Q_PRIME_LOC(self):
    return self.q_prime

  @property
  def XNUE(self):
    return self.nu_ee

  @property
  def KAPPA_LOC(self):
    return self.kappa

  @property
  def S_KAPPA_LOC(self):
    return self.kappa_shear

  @property
  def DELTA_LOC(self):
    return self.delta

  @property
  def S_DELTA_LOC(self):
    return self.delta_shear

  @property
  def BETAE(self):
    return self.beta_e

  @property
  def ZEFF(self):
    return self.Zeff

  @property
  def RLNS_1(self):
    return self.lref_over_lne

  @property
  def RLTS_1(self):
    return self.lref_over_lte

  @property
  def RLTS_2(self):
    return self.lref_over_lti

  @property
  def RMIN_LOC(self):
    return self.r_minor


class TGLFBasedTransportModel(
    quasilinear_transport_model.QuasilinearTransportModel
):
  """Base class for TGLF-based transport models."""

  def _prepare_tglf_inputs(
      self,
      transport: DynamicRuntimeParams,  # pylint: disable=unused-argument
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> TGLFInputs:
    """Construct a TGLFInputs object from the TORAX state.

    Normalisation and coordinate conventions:
      TGLF values are normalised with respect to:
        - Minor radius at the LCFS (a) in m
        - B_unit in T (defined at https://gafusion.github.io/doc/geometry.html#effective-field)
        - The ion sound speed in m/s (defined at https://gafusion.github.io/doc/cgyro/outputs.html#output-normalization)
      The radial coordinate is the minor radius of the flux surface (r) in m.
      Psi_TGLF is Psi_TORAX / 2π.

    Shortcomings:
      Currently, geometry parameters are taken directly from the global
      profiles. For complete accuracy, a Miller (or s-α) geometry fit routine
      should instead be implemented to calculate *local* geometry parameters.

      At present, only a subset of TGLF inputs have been implemented.

    References:
      The main points of reference for this implementation are:
        - The TGLF documentation (https://gafusion.github.io/doc/tglf)
        - General GACode documentation (https://gafusion.github.io/doc/input_gacode.html)
        - The CGYRO documentation (https://gafusion.github.io/doc/cgyro.html)

      If something isn't documented in TGLF, it will use the same definitions as
      CGYRO and other GACode libraries. The implementation in this function uses
      SI units and definitions throughout unless specified otherwise; be careful
      when comparing with the TGLF docs, as they use a combination of SI, keV,
      1e19m^-3, and Gaussian CGS.
    """
    constants = constants_module.CONSTANTS
    T_e = core_profiles.T_e.face_value() * constants.keV2J
    n_e = core_profiles.n_e.face_value()

    # Reference values used for TGLF-specific normalisation
    # - 'a' in TGLF means the minor radius at the LCFS
    # - 'r' in TGLF means the flux surface centroid minor radius. Gradients are
    #   taken w.r.t. r
    #   https://gafusion.github.io/doc/tglf/tglf_list.html#rmin-loc
    # - B_unit = 1/r d(psi_tor/2π)/dr = q/2πr dpsi/dr, noting that psi_tor is
    #   psi/2π
    #   https://gafusion.github.io/doc/geometry.html#effective-field
    #   https://gafusion.github.io/doc/cgyro.html#faq
    # - c_s (ion sound speed)
    #   https://gafusion.github.io/doc/cgyro/outputs.html#output-normalization
    m_D_amu = 2.014  # Mass of deuterium [amu] TODO: load from lookup table
    m_D = m_D_amu * constants.mp  # Mass of deuterium [kg]
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
    # TODO: check a is the correct reference length
    normalized_log_gradients = quasilinear_transport_model.NormalizedLogarithmicGradients.from_profiles(
        core_profiles=core_profiles,
        radial_coordinate=geo.r_mid,  # On the cell grid
        reference_length=a,
    )

    # Dimensionless electron-electron collision frequency = nu_ee / (c_s/a)
    # https://gafusion.github.io/doc/tglf/tglf_list.html#xnue
    # https://gafusion.github.io/doc/cgyro/cgyro_list.html#cgyro-nu-ee
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
    nu_ee = (jnp.sqrt(2) * n_e * constants.qe**4 * log_Lambda) / (
        16 * jnp.pi * constants.epsilon0**2 * constants.me**0.5 * T_e**1.5
    )
    normalized_nu_ee = nu_ee / (c_s / a)

    # Dimensionless safety factor shear
    # https://gafusion.github.io/doc/tglf/tglf_list.html#tglf-q-prime-loc
    # - TGLF definition is q^2 a^2 s / r^2
    #   where s = r/q dq/dr
    #   (https://gafusion.github.io/doc/cgyro/cgyro_list.html#cgyro-s)
    q_prime = (
        psi_calculations.calc_s_face(geo, core_profiles.psi)
        * core_profiles.q_face**2
        * a**2
        / r**2
    )

    # Electron beta
    # https://gafusion.github.io/doc/tglf/tglf_list.html#tglf-betae
    # https://gafusion.github.io/doc/cgyro.html#faq
    # https://gafusion.github.io/doc/cgyro/cgyro_list.html#betae-unit
    # - In the TGLF docs, beta_e equation shown in CGS units, this is the SI
    #   version
    beta_e = 2 * constants.mu0 * n_e * T_e / B_unit**2

    # Major radius shear = drmaj/drmin, where 'rmaj' is the flux surface
    # centroid major radius and 'rmin' the flux surface centroid minor radius
    # https://gafusion.github.io/doc/tglf/tglf_list.html#tglf-drmajdx-loc
    r_major = (geo.R_in_face + geo.R_out_face) / 2
    dr_major = jnp.gradient(r_major, r)

    # Dimensionless elongation shear = r/kappa dkappa/dr
    # https://gafusion.github.io/doc/tglf/tglf_list.html#tglf-s-kappa-loc
    kappa = geo.elongation_face
    kappa_shear = r / kappa * jnp.gradient(kappa, r)

    # Dimensionless triangularity shear = r ddelta/dr
    # https://gafusion.github.io/doc/tglf/tglf_list.html#tglf-s-delta-loc
    delta_shear = r * jnp.gradient(geo.delta_face, r)

    # Gyrobohm diffusivity
    # https://gafusion.github.io/doc/tglf/tglf_table.html#id7
    # https://gafusion.github.io/doc/cgyro/outputs.html#output-normalization
    # - TGLF uses the same normalisation as CGYRO.
    # - The extra c^2 comes from Gaussian units when calculating \rho_s
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

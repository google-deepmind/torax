# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Angioni-Sauter neoclassical transport model.

This module implements the neoclassical transport model described in:
C. Angioni and O. Sauter, Phys. Plasmas 7, 1224 (2000).
https://doi.org/10.1063/1.873933

The implementation was facilitated by and verified against the NEOS code:
https://gitlab.epfl.ch/spc/public/neos [O. Sauter et al]
"""

from typing import Annotated, Literal

from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical import formulas
from torax._src.neoclassical.transport import base
from torax._src.neoclassical.transport import runtime_params as transport_runtime_params
from torax._src.physics import collisions
from torax._src.torax_pydantic import torax_pydantic
from typing_extensions import override

# pylint: disable=invalid-name


class AngioniSauterModelConfig(base.NeoclassicalTransportModelConfig):
  """Pydantic model for the Angioni-Sauter neoclassical transport model."""

  model_name: Annotated[
      Literal['angioni_sauter'], torax_pydantic.JAX_STATIC
  ] = 'angioni_sauter'

  @override
  def build_model(self) -> 'AngioniSauterModel':
    return AngioniSauterModel()

  @override
  def build_runtime_params(self) -> transport_runtime_params.RuntimeParams:
    return super().build_runtime_params()


class AngioniSauterModel(base.NeoclassicalTransportModel):
  """Implements the Angioni-Sauter neoclassical transport model."""

  @override
  def _call_implementation(
      self,
      runtime_params: runtime_params_slice.RuntimeParams,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> base.NeoclassicalTransport:
    """Calculates neoclassical transport coefficients."""
    return _calculate_angioni_sauter_transport(
        runtime_params=runtime_params,
        geometry=geometry,
        core_profiles=core_profiles,
    )

  def __hash__(self) -> int:
    return hash(self.__class__.__name__)

  def __eq__(self, other) -> bool:
    return isinstance(other, self.__class__)


def _calculate_angioni_sauter_transport(
    runtime_params: runtime_params_slice.RuntimeParams,
    geometry: geometry_lib.Geometry,
    core_profiles: state.CoreProfiles,
) -> base.NeoclassicalTransport:
  """JIT-compatible implementation of the Angioni-Sauter transport model.

  Args:
    runtime_params: Runtime parameters.
    geometry: Geometry object.
    core_profiles: Core profiles object.

  Returns:
    Neoclassical transport coefficients.

  All internally assigned profiles are on the face grid. The face suffix is
  omitted for brevity.
  """

  del runtime_params  # Unused.

  # --- Step 1: Calculate intermediate physics quantities ---

  # Calculate trapped fractions ft and ftd from paper Eq. (17)
  # TODO(b/426291465): Implement a more accurate calculation of <1/B^2>.
  # Analytical expressions for circular geometry.
  B2_avg = geometry.B_0**2 / jnp.sqrt(1.0 - geometry.epsilon_face**2)
  Bm2_avg = geometry.B_0**-2 * (1.0 + 1.5 * geometry.epsilon_face**2)

  B2_avg_Bm2_avg = B2_avg * Bm2_avg

  # Use the Sauter model's effective trapped fraction logic
  aa = (1.0 - geometry.epsilon_face) / (1.0 + geometry.epsilon_face)
  epseff = (
      0.67
      * (1.0 - 1.4 * jnp.abs(geometry.delta_face) * geometry.delta_face)
      * geometry.epsilon_face
  )
  ftrap = 1.0 - jnp.sqrt(aa) * (1.0 - epseff) / (1.0 + 2.0 * jnp.sqrt(epseff))

  # Equation (17)
  ftrap_d = 1.0 - (1.0 - ftrap) / B2_avg_Bm2_avg

  # Collisionalities
  log_lambda_ei = collisions.calculate_log_lambda_ei(
      core_profiles.T_e.face_value(), core_profiles.n_e.face_value()
  )
  nu_e_star = formulas.calculate_nu_e_star(
      q=core_profiles.q_face,
      geo=geometry,
      n_e=core_profiles.n_e.face_value(),
      T_e=core_profiles.T_e.face_value(),
      Z_eff=core_profiles.Z_eff_face,
      log_lambda_ei=log_lambda_ei,
  )

  log_lambda_ii = collisions.calculate_log_lambda_ii(
      core_profiles.T_i.face_value(),
      core_profiles.n_i.face_value(),
      core_profiles.Z_i_face,
  )

  # Equation 18c from Sauter PoP 1999
  nu_i_star = (
      4.9e-18
      * core_profiles.q_face
      * geometry.R_major
      * core_profiles.n_i.face_value()
      * core_profiles.Z_i_face**4
      * log_lambda_ii
      / (
          (core_profiles.T_i.face_value() * 1e3) ** 2
          * (geometry.epsilon_face + constants.CONSTANTS.eps) ** 1.5
      )
  )

  # Impurity strength parameter
  # Using single impurity definition: alpha_I = nimpZimp/niZi = Zeff - 1
  alpha_I = core_profiles.Z_eff_face - 1.0

  # --- Step 2: Calculate dimensionless transport matrix K_mn ---
  Kmn_e, Kmn_i = _calculate_Kmn(
      ftrap=ftrap,
      ftrap_d=ftrap_d,
      Z_eff=core_profiles.Z_eff_face,
      B2_avg_Bm2_avg=B2_avg_Bm2_avg,
      nu_e_star=nu_e_star,
      nu_i_star=nu_i_star,
      alpha_I=alpha_I,
  )

  # --- Step 3: Calculate dimensional transport matrix L_mn ---
  Lmn_e, Lmn_i = _calculate_Lmn(
      Kmn_e=Kmn_e,
      Kmn_i=Kmn_i,
      geo=geometry,
      core_profiles=core_profiles,
      epsilon=geometry.epsilon_face,
      nu_e_star=nu_e_star,
      nu_i_star=nu_i_star,
      B2_avg=B2_avg,
      Bm2_avg=Bm2_avg,
  )

  # --- Step 4: Calculate thermodynamic forces ---
  dpsi_drhon = core_profiles.psi.face_grad() + constants.CONSTANTS.eps
  dlnne_dpsi = (
      core_profiles.n_e.face_grad() / core_profiles.n_e.face_value()
  ) / (dpsi_drhon + constants.CONSTANTS.eps)
  dlnte_dpsi = (
      core_profiles.T_e.face_grad() / core_profiles.T_e.face_value()
  ) / (dpsi_drhon + constants.CONSTANTS.eps)
  dlnni_dpsi = (
      core_profiles.n_i.face_grad() / core_profiles.n_i.face_value()
  ) / (dpsi_drhon + constants.CONSTANTS.eps)
  dlnti_dpsi = (
      core_profiles.T_i.face_grad() / core_profiles.T_i.face_value()
  ) / (dpsi_drhon + constants.CONSTANTS.eps)

  # --- Step 5: Calculate neoclassical fluxes ---
  pe = (
      core_profiles.n_e.face_value()
      * core_profiles.T_e.face_value()
      * constants.CONSTANTS.keV2J
  )
  pi = (
      core_profiles.n_i.face_value()
      * core_profiles.T_i.face_value()
      * constants.CONSTANTS.keV2J
  )
  Rpe = pe / (pe + pi)
  alpha = -Kmn_i[:, 0, 1]
  E_parallel = core_profiles.psidot.face_value() / (
      2 * jnp.pi * geometry.R_major
  )

  # Total electron heat flux Q_e = B_e2 * T_e / (dpsi/drho) (see Angioni Sec 5)
  Be2 = (
      Lmn_e[:, 1, 0] * dlnne_dpsi
      + (Lmn_e[:, 1, 0] + Lmn_e[:, 1, 1]) * dlnte_dpsi
      + (1 - Rpe) / Rpe * Lmn_e[:, 1, 0] * dlnni_dpsi
      + (1 - Rpe) / Rpe * (Lmn_e[:, 1, 0] + alpha * Lmn_e[:, 1, 3]) * dlnti_dpsi
      + Lmn_e[:, 1, 2] * E_parallel / geometry.B_0
  )

  # Total ion heat flux Q_i = B_i2 / T_i * (dpsi/drho) (see Angioni Sec 5)
  Bi2 = (
      alpha * Lmn_e[:, 3, 0] * dlnne_dpsi
      + alpha * (Lmn_e[:, 3, 0] + Lmn_e[:, 3, 1]) * dlnte_dpsi
      + alpha * (1 - Rpe) / Rpe * Lmn_e[:, 3, 0] * dlnni_dpsi
      + alpha * Lmn_e[:, 3, 2] * E_parallel / geometry.B_0
      + (
          Lmn_i[:, 1, 1]
          + (1 - Rpe) / Rpe * alpha**2 / core_profiles.Z_i_face * Lmn_e[:, 3, 3]
      )
      * dlnti_dpsi
  )

  # --- Step 6: Extract transport coefficients --- Angioni Section 5.

  # Q_e = - chi_e * n_e * dT/drho = B_e2 * T_e / dpsi/drho
  # Q_i = - chi_i * n_i * dT/drho = B_i2 * T_i / dpsi/drho
  chi_neo_e = -Be2 / (
      core_profiles.n_e.face_value()
      * dlnte_dpsi
      * (dpsi_drhon / geometry.rho_b) ** 2
      + constants.CONSTANTS.eps
  )
  chi_neo_i = -Bi2 / (
      core_profiles.n_i.face_value()
      * dlnti_dpsi
      * (dpsi_drhon / geometry.rho_b) ** 2
      + constants.CONSTANTS.eps
  )

  # Decomposition of particle flux Be1 = Gamma * dpsi_drho. Page 1232+1233.

  # Diffusive part of particle flux
  # D_e * dn_e/drho  = - L00 *dlog(n_e)/dpsi / dpsi/drho
  D_neo_e = -Lmn_e[:, 0, 0] / (
      core_profiles.n_e.face_value() * (dpsi_drhon / geometry.rho_b) ** 2
      + constants.CONSTANTS.eps
  )

  # Convective part of particle flux, apart from the Ware Pinch term
  # V*n*dpsi/rho = (L00+L01)*dlog(Te)/dpsi + (1-Rpe)/Rpe*L00*dlog(ni)/dpsi +
  # (1-Rpe)/Rpe * (L00+alpha*L03) *dlog(Ti)/dpsi
  V_neo_e = (
      (Lmn_e[:, 0, 0] + Lmn_e[:, 0, 1]) * dlnte_dpsi
      + (1 - Rpe) / Rpe * Lmn_e[:, 0, 0] * dlnni_dpsi
      + (1 - Rpe) / Rpe * (Lmn_e[:, 0, 0] + alpha * Lmn_e[:, 0, 3]) * dlnti_dpsi
  ) / (
      dpsi_drhon / geometry.rho_b * core_profiles.n_e.face_value()
      + constants.CONSTANTS.eps
  )

  # Ware pinch term component of particle convection
  # V_ware*n*dpsi/rho = L02*<E_parallel * B>/<B^2>
  V_neo_ware_e = (
      Lmn_e[:, 0, 2]
      * E_parallel
      / (
          geometry.B_0
          * (dpsi_drhon / geometry.rho_b)
          * core_profiles.n_e.face_value()
          + constants.CONSTANTS.eps
      )
  )

  return base.NeoclassicalTransport(
      chi_neo_i=chi_neo_i,
      chi_neo_e=chi_neo_e,
      D_neo_e=D_neo_e,
      V_neo_e=V_neo_e,
      V_neo_ware_e=V_neo_ware_e,
  )


def _calculate_Kmn(
    ftrap: array_typing.FloatVectorFace,
    ftrap_d: array_typing.FloatVectorFace,
    Z_eff: array_typing.FloatVectorFace,
    B2_avg_Bm2_avg: array_typing.FloatVectorFace,
    nu_e_star: array_typing.FloatVectorFace,
    nu_i_star: array_typing.FloatVectorFace,
    alpha_I: array_typing.FloatVectorFace,
) -> tuple[array_typing.Array, array_typing.Array]:
  """Calculates the dimensionless transport matrices Kmn."""

  # F_mn matrix, Eq. (24)
  F_ftrap = _Fmn_X(ftrap, Z_eff)
  F_ftrap_d = _Fmn_X(ftrap_d, Z_eff)

  # Coefficients from Appendix B.
  a_coeffs, b_coeffs, c_coeffs, d_coeffs, a2, b2, c2, d2 = _coeffs_appendix_B(
      Z_eff
  )

  # Effective collisionalities and geometrical factors
  # Eq. (29)
  nu_e_star_eff = nu_e_star / (1.0 + 7.0 * ftrap**2)
  nu_e_star_eff_sqrt = jnp.sqrt(nu_e_star_eff)
  nu_i_star_eff = nu_i_star / (1.0 + 7.0 * ftrap**2)
  # Eq. (30g)
  FPS = 1.0 - 1.0 / B2_avg_Bm2_avg
  FPS4 = B2_avg_Bm2_avg - 1.0

  # Banana-regime coefficients
  # Eq. 23
  K11e_0 = -0.5 * F_ftrap_d[:, 0, 0]
  K12e_0 = 0.75 * F_ftrap_d[:, 0, 1]
  K22e_0 = -(13.0 / 8.0 + 1 / jnp.sqrt(2.0) / Z_eff) * F_ftrap_d[:, 1, 1]
  K14e_0 = -0.5 * F_ftrap[:, 0, 0]
  K24e_0 = 0.75 * F_ftrap[:, 0, 1]

  # Eq. 30c
  H11_0 = K11e_0
  H12_0 = K12e_0 + 2.5 * K11e_0
  H22_0 = K22e_0 + 5.0 * K12e_0 - 6.25 * K11e_0

  # Eq. 30f
  H41_0 = K14e_0
  H42_0 = K24e_0 + 2.5 * K14e_0

  # Collisionality-dependent Hmn (Eqs. 30b, 30e)
  temp1 = nu_e_star_eff * ftrap_d**3 * (1.0 + ftrap_d**6)
  H11 = (
      H11_0
      / (
          1.0
          + a_coeffs[:, 0, 0] * nu_e_star_eff_sqrt
          + b_coeffs[:, 0, 0] * nu_e_star_eff
      )
      - d_coeffs[:, 0, 0] * temp1 / (1.0 + c_coeffs[:, 0, 0] * temp1) * FPS
  )
  H12 = (
      H12_0
      / (
          1.0
          + a_coeffs[:, 0, 1] * nu_e_star_eff_sqrt
          + b_coeffs[:, 0, 1] * nu_e_star_eff
      )
      - d_coeffs[:, 0, 1] * temp1 / (1.0 + c_coeffs[:, 0, 1] * temp1) * FPS
  )
  H22 = (
      H22_0
      / (
          1.0
          + a_coeffs[:, 1, 1] * nu_e_star_eff_sqrt
          + b_coeffs[:, 1, 1] * nu_e_star_eff
      )
      - d_coeffs[:, 1, 1] * temp1 / (1.0 + c_coeffs[:, 1, 1] * temp1) * FPS
  )

  temp2 = 1.0 / (1.0 + nu_e_star_eff**2 * ftrap**12)
  temp4 = nu_e_star_eff * ftrap**3 * (1.0 + 0.8 * ftrap**3)
  H41 = (
      H41_0
      / (
          1.0
          + a_coeffs[:, 0, 0] * nu_e_star_eff_sqrt
          + b_coeffs[:, 0, 0] * nu_e_star_eff
      )
      - d_coeffs[:, 0, 0] * temp4 / (1.0 + c_coeffs[:, 0, 0] * temp4) * FPS4
  ) * temp2
  H42 = (
      H42_0
      / (
          1.0
          + a_coeffs[:, 0, 1] * nu_e_star_eff_sqrt
          + b_coeffs[:, 0, 1] * nu_e_star_eff
      )
      - d_coeffs[:, 0, 1] * temp4 / (1.0 + c_coeffs[:, 0, 1] * temp4) * FPS4
  ) * temp2

  # Electron Kmn matrix (Eq. 30a, 30d)
  Kmn_e = jnp.zeros((ftrap.shape[0], 4, 4))
  Kmn_e = Kmn_e.at[:, 0, 0].set(H11)
  Kmn_e = Kmn_e.at[:, 0, 1].set(H12 - 2.5 * H11)
  Kmn_e = Kmn_e.at[:, 1, 0].set(Kmn_e[:, 0, 1])
  Kmn_e = Kmn_e.at[:, 1, 1].set(H22 - 5.0 * H12 + 6.25 * H11)
  Kmn_e = Kmn_e.at[:, 0, 3].set(H41)
  Kmn_e = Kmn_e.at[:, 3, 0].set(Kmn_e[:, 0, 3])
  Kmn_e = Kmn_e.at[:, 1, 3].set(H42 - 2.5 * H41)
  Kmn_e = Kmn_e.at[:, 3, 1].set(Kmn_e[:, 1, 3])

  # Supplement K matrix with "bootstrap terms" needed for Ware pinch from the
  # Sauter model (PoP 1999)
  Kmn_e = Kmn_e.at[:, 0, 2].set(
      -formulas.calculate_L31(ftrap, nu_e_star, Z_eff)
  )
  Kmn_e = Kmn_e.at[:, 2, 0].set(Kmn_e[:, 0, 2])
  Kmn_e = Kmn_e.at[:, 1, 2].set(
      -formulas.calculate_L32(ftrap, nu_e_star, Z_eff)
  )
  Kmn_e = Kmn_e.at[:, 2, 1].set(Kmn_e[:, 1, 2])

  # Ion Kmn matrix
  # alpha coefficient, Eq. (25)
  alpha = (
      -(0.62 + 1.5 * alpha_I)
      / (0.53 + alpha_I)
      * ((1.0 - ftrap) / (1.0 - 0.22 * ftrap - 0.19 * ftrap**2))
  )

  # Eq. 24d
  F22_i_ftrapd = (1.0 - 0.55) * (
      1.0 + 1.54 * alpha_I
  ) * ftrap_d + ftrap_d**2 * (0.75 + ftrap_d * (-0.7 + 0.5 * ftrap_d)) * (
      1.0 + 2.92 * alpha_I
  )

  # K11, for completeness. Bottom of page 1230.
  K11_i = 1.0 / (0.11 + 1.7 * ftrap - 1.25 * ftrap**2 + 0.44 * ftrap**3) - 1.0

  # K22_i, Eq. 30h
  mu_i_star_eff = nu_i_star_eff * (1.0 + 1.54 * alpha_I)
  Hp = 1.0 + 1.33 * alpha_I * (1.0 + 0.60 * alpha_I) / (1.0 + 1.79 * alpha_I)
  temp4_i = mu_i_star_eff * ftrap_d**3 * (1.0 + ftrap_d**6)
  K22_i = (
      -F22_i_ftrapd / (1 + a2 * jnp.sqrt(mu_i_star_eff) + b2 * mu_i_star_eff)
      - d2 * temp4_i / (1.0 + c2 * temp4_i) * Hp * FPS
  )

  Kmn_i = jnp.zeros((ftrap.shape[0], 2, 2))
  Kmn_i = Kmn_i.at[:, 0, 0].set(K11_i)
  Kmn_i = Kmn_i.at[:, 1, 1].set(K22_i)
  Kmn_i = Kmn_i.at[:, 0, 1].set(-alpha)  # Eq. 25
  Kmn_i = Kmn_i.at[:, 1, 0].set(alpha)

  return Kmn_e, Kmn_i


def _Fmn_X(
    X: array_typing.FloatVectorFace, Z_eff: array_typing.FloatVectorFace
) -> array_typing.Array:
  """Calculates the F_mn matrix from Eq. (24) of Angioni & Sauter 2000."""
  F11 = X + X * (0.9 + X * (-1.9 + X * (1.6 - 0.6 * X))) / (Z_eff + 0.5)
  F12 = X + X * (0.6 + X * (-0.95 + X * (0.3 + 0.05 * X))) / (Z_eff + 0.5)
  F22 = X + X * (-0.11 + X * (0.08 + 0.03 * X)) / (Z_eff + 0.5)
  # Transpose such that the leading dimension is the same as the input arrays.
  return jnp.array([[F11, F12], [F12, F22]]).transpose(2, 0, 1)


def _coeffs_appendix_B(
    Z_eff: array_typing.Array,
) -> tuple[
    array_typing.Array,
    array_typing.Array,
    array_typing.Array,
    array_typing.Array,
    float,
    float,
    float,
    float,
]:
  """Calculates coefficients from Appendix B of Angioni & Sauter 2000."""
  a11 = (1.0 + 3.0 * Z_eff) / (0.77 + 1.22 * Z_eff)
  a12 = (0.72 + 0.42 * Z_eff) / (1.0 + 0.5 * Z_eff)
  a22 = 0.46 * jnp.ones_like(a11)
  # Transpose such that the leading dimension is the same as the input arrays.
  a_coeffs = jnp.array([[a11, a12], [a12, a22]]).transpose(2, 0, 1)

  b11 = (1.0 + 1.1 * Z_eff) / (1.37 * Z_eff)
  b12 = (1.0 + Z_eff) / (2.99 * Z_eff)
  b22 = Z_eff / (-3.0 + 5.32 * Z_eff)
  b_coeffs = jnp.array([[b11, b12], [b12, b22]]).transpose(2, 0, 1)

  c11 = (0.1 + 0.34 * Z_eff) / (1.65 * Z_eff)
  c12 = (0.27 + 0.4 * Z_eff) / (1.0 + 3.0 * Z_eff)
  c22 = (0.22 + 0.55 * Z_eff) / (-1.0 + 7.0 * Z_eff)
  c_coeffs = jnp.array([[c11, c12], [c12, c22]]).transpose(2, 0, 1)

  d11 = 0.23 * Z_eff / (-1.0 + 3.85 * Z_eff)
  d12 = (0.22 + 0.38 * Z_eff) / (1.0 + 6.1 * Z_eff)
  d22 = (0.25 + 0.05 * Z_eff) / (1.0 + 0.82 * Z_eff)

  d_coeffs = jnp.array([[d11, d12], [d12, d22]]).transpose(2, 0, 1)
  a2, b2, c2, d2 = 1.03, 0.31, 0.22, 0.175
  return a_coeffs, b_coeffs, c_coeffs, d_coeffs, a2, b2, c2, d2


def _calculate_Lmn(
    Kmn_e: array_typing.Array,
    Kmn_i: array_typing.Array,
    geo: geometry_lib.Geometry,
    core_profiles: state.CoreProfiles,
    epsilon: array_typing.FloatVectorFace,
    nu_e_star: array_typing.FloatVectorFace,
    nu_i_star: array_typing.FloatVectorFace,
    B2_avg: array_typing.FloatVectorFace,
    Bm2_avg: array_typing.FloatVectorFace,
) -> tuple[array_typing.Array, array_typing.Array]:
  """Calculates the dimensional transport matrices Lmn."""
  # Normalization factors from Eqs. 16, 20, 21
  consts = constants.CONSTANTS
  thermal_velocity_e = jnp.sqrt(
      2 * core_profiles.T_e.face_value() * consts.keV2J / consts.me
  )
  collision_time_e = (core_profiles.q_face * geo.R_major) / (
      nu_e_star * epsilon**1.5 * thermal_velocity_e + consts.eps
  )
  thermal_velocity_i = jnp.sqrt(
      2
      * core_profiles.T_i.face_value()
      * consts.keV2J
      / (core_profiles.A_i * consts.mp)
  )
  collision_time_i = (core_profiles.q_face * geo.R_major) / (
      nu_i_star * epsilon**1.5 * thermal_velocity_i + consts.eps
  )

  r_larmor_e = consts.me * thermal_velocity_e / consts.qe
  r_larmor_i = (
      consts.mp
      * core_profiles.A_i
      * thermal_velocity_i
      / (consts.qe * core_profiles.Z_i_face)
  )

  dpsi_dr = core_profiles.psi.face_grad() / geo.rho_b

  Ld = (
      core_profiles.n_e.face_value()
      * r_larmor_e**2
      / collision_time_e
      * dpsi_dr**2
  )
  Ldi = (
      core_profiles.n_i.face_value()
      * r_larmor_i**2
      / collision_time_i
      * dpsi_dr**2
  )
  Lb = geo.F_face * core_profiles.n_e.face_value()
  Lbi = geo.F_face * core_profiles.n_i.face_value()

  Lsi = (
      core_profiles.n_i.face_value()
      * (consts.qe * core_profiles.Z_i_face) ** 2
      * collision_time_i
      * geo.B_0**2
      / (
          consts.mp
          * core_profiles.A_i
          * core_profiles.T_i.face_value()
          * consts.keV2J
      )
  )

  # Calculate electron matrix (Eq. 20)
  Lmn_e = jnp.zeros((nu_e_star.shape[0], 4, 4))

  Lmn_e = Lmn_e.at[:, 0, 0].set(Kmn_e[:, 0, 0] * Ld * Bm2_avg * geo.B_0**2)
  Lmn_e = Lmn_e.at[:, 0, 1].set(Kmn_e[:, 0, 1] * Ld * Bm2_avg * geo.B_0**2)
  Lmn_e = Lmn_e.at[:, 0, 2].set(Kmn_e[:, 0, 2] * Lb)
  Lmn_e = Lmn_e.at[:, 0, 3].set(Kmn_e[:, 0, 3] * Ld / B2_avg * geo.B_0**2)

  Lmn_e = Lmn_e.at[:, 1, 0].set(Lmn_e[:, 0, 1])
  Lmn_e = Lmn_e.at[:, 1, 1].set(Kmn_e[:, 1, 1] * Ld * Bm2_avg * geo.B_0**2)
  Lmn_e = Lmn_e.at[:, 1, 2].set(Kmn_e[:, 1, 2] * Lb)
  Lmn_e = Lmn_e.at[:, 1, 3].set(Kmn_e[:, 1, 3] * Ld / B2_avg * geo.B_0**2)

  Lmn_e = Lmn_e.at[:, 2, 0].set(Lmn_e[:, 0, 2])
  Lmn_e = Lmn_e.at[:, 2, 1].set(Lmn_e[:, 1, 2])
  # Lmn_e[:, 2 ,2] is not used
  Lmn_e = Lmn_e.at[:, 2, 3].set(Kmn_e[:, 2, 3] * Lb)

  Lmn_e = Lmn_e.at[:, 3, 0].set(Lmn_e[:, 0, 3])
  Lmn_e = Lmn_e.at[:, 3, 1].set(Lmn_e[:, 1, 3])
  Lmn_e = Lmn_e.at[:, 3, 2].set(Lmn_e[:, 2, 3])
  Lmn_e = Lmn_e.at[:, 3, 3].set(Kmn_e[:, 3, 3] * Ld / B2_avg * geo.B_0**2)

  # Calculate ion matrix
  Lmn_i = jnp.zeros((nu_i_star.shape[0], 2, 2))
  Lmn_i = Lmn_i.at[:, 0, 0].set(Kmn_i[:, 0, 0] * Lsi * B2_avg / geo.B_0**2)
  Lmn_i = Lmn_i.at[:, 0, 1].set(Kmn_i[:, 0, 1] * Lbi)
  Lmn_i = Lmn_i.at[:, 1, 0].set(-Lmn_i[:, 1, 0])
  Lmn_i = Lmn_i.at[:, 1, 1].set(Kmn_i[:, 1, 1] * Ldi * Bm2_avg * geo.B_0**2)

  return Lmn_e, Lmn_i

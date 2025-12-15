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

import dataclasses
from typing import Annotated, Literal

import jax
from jax import numpy as jnp
import pydantic
from torax._src import array_typing
from torax._src import constants
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical import formulas
from torax._src.neoclassical.transport import base
from torax._src.neoclassical.transport import runtime_params as transport_runtime_params
from torax._src.physics import collisions
from torax._src.torax_pydantic import torax_pydantic
from typing_extensions import override

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(transport_runtime_params.RuntimeParams):
  """RuntimeParams for the Angioni-Sauter neoclassical transport model."""

  use_shaing_ion_correction: array_typing.BoolScalar
  shaing_ion_multiplier: array_typing.FloatScalar
  shaing_blend_start: array_typing.FloatScalar
  shaing_blend_rate: array_typing.FloatScalar


class AngioniSauterModelConfig(base.NeoclassicalTransportModelConfig):
  """Pydantic model for the Angioni-Sauter neoclassical transport model."""

  model_name: Annotated[
      Literal['angioni_sauter'], torax_pydantic.JAX_STATIC
  ] = 'angioni_sauter'
  use_shaing_ion_correction: bool = False
  shaing_ion_multiplier: pydantic.NonNegativeFloat = 1.8
  shaing_blend_start: torax_pydantic.UnitInterval = 0.2
  shaing_blend_rate: pydantic.NonNegativeFloat = 5.0

  @override
  def build_model(self) -> 'AngioniSauterModel':
    return AngioniSauterModel()

  @override
  def build_runtime_params(self) -> RuntimeParams:
    base_kwargs = dataclasses.asdict(super().build_runtime_params())
    return RuntimeParams(
        use_shaing_ion_correction=self.use_shaing_ion_correction,
        shaing_ion_multiplier=self.shaing_ion_multiplier,
        shaing_blend_start=self.shaing_blend_start,
        shaing_blend_rate=self.shaing_blend_rate,
        **base_kwargs
    )


class AngioniSauterModel(base.NeoclassicalTransportModel):
  """Implements the Angioni-Sauter neoclassical transport model."""

  @override
  def _call_implementation(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> base.NeoclassicalTransport:
    """Calculates neoclassical transport coefficients.

    When use_shaing_ion_correction is enabled, chi_ion is smoothly blended
    between Shaing (near axis) and Angioni-Sauter (far from axis) models using
    an exponential transition function.

    Args:
      runtime_params: Runtime parameters.
      geometry: Geometry object.
      core_profiles: Core profiles object.

    Returns:
      Neoclassical transport coefficients.
    """
    angioni_sauter = _calculate_angioni_sauter_transport(
        runtime_params=runtime_params,
        geometry=geometry,
        core_profiles=core_profiles,
    )
    shaing = _calculate_shaing_transport(
        runtime_params=runtime_params,
        geometry=geometry,
        core_profiles=core_profiles,
    )

    # Needed for pytype.
    assert isinstance(runtime_params.neoclassical.transport, RuntimeParams)

    # Calculate sigmoid blend weight for Angioni-Sauter (alpha)
    # If correction disabled: alpha = 1 (pure Angioni-Sauter)
    # If correction enabled: alpha varies smoothly with rho_norm
    alpha = jnp.where(
        runtime_params.neoclassical.transport.use_shaing_ion_correction,
        _calculate_blend_alpha(
            rho_face_norm=geometry.rho_face_norm,
            start=runtime_params.neoclassical.transport.shaing_blend_start,
            rate=runtime_params.neoclassical.transport.shaing_blend_rate,
        ),
        1.0,  # Pure Angioni-Sauter when correction disabled
    )

    return base.NeoclassicalTransport(
        # Ion transport blend: (1-alpha)*Shaing + alpha*Angioni-Sauter
        chi_neo_i=(1.0 - alpha) * shaing.chi_neo_i
        + alpha * angioni_sauter.chi_neo_i,
        # Electron transport: pure Angioni-Sauter
        chi_neo_e=angioni_sauter.chi_neo_e,
        D_neo_e=angioni_sauter.D_neo_e,
        V_neo_e=angioni_sauter.V_neo_e,
        V_neo_ware_e=angioni_sauter.V_neo_ware_e,
    )

  def __hash__(self) -> int:
    return hash(self.__class__.__name__)

  def __eq__(self, other) -> bool:
    return isinstance(other, self.__class__)


def _calculate_angioni_sauter_transport(
    runtime_params: runtime_params_lib.RuntimeParams,
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
  B2_avg_Bm2_avg = geometry.gm5_face * geometry.gm4_face

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
      * geometry.R_major_profile_face
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
  pe = core_profiles.pressure_thermal_e.face_value()
  pi = core_profiles.pressure_thermal_i.face_value()
  Rpe = pe / (pe + pi)
  alpha = -Kmn_i[:, 0, 1]
  E_parallel = core_profiles.psidot.face_value() / (
      2 * jnp.pi * geometry.R_major_profile_face
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
  chi_neo_e_bulk = -Be2[1:] / (
      core_profiles.n_e.face_value()[1:]
      * dlnte_dpsi[1:]
      * (dpsi_drhon[1:] / geometry.rho_b) ** 2
      + constants.CONSTANTS.eps
  )
  chi_neo_e_axis = jnp.array([chi_neo_e_bulk[0]])
  chi_neo_e = jnp.concatenate([chi_neo_e_axis, chi_neo_e_bulk])

  chi_neo_i_bulk = -Bi2[1:] / (
      core_profiles.n_i.face_value()[1:]
      * dlnti_dpsi[1:]
      * (dpsi_drhon[1:] / geometry.rho_b) ** 2
      + constants.CONSTANTS.eps
  )
  chi_neo_i_axis = jnp.array([chi_neo_i_bulk[0]])
  chi_neo_i = jnp.concatenate([chi_neo_i_axis, chi_neo_i_bulk])

  # Decomposition of particle flux Be1 = Gamma * dpsi_drho. Page 1232+1233.

  # Diffusive part of particle flux
  # D_e * dn_e/drho  = - L00 *dlog(n_e)/dpsi / dpsi/drho
  D_neo_e_bulk = -Lmn_e[1:, 0, 0] / (
      core_profiles.n_e.face_value()[1:] * (dpsi_drhon[1:] / geometry.rho_b) ** 2
      + constants.CONSTANTS.eps
  )
  D_neo_e_axis = jnp.array([D_neo_e_bulk[0]])
  D_neo_e = jnp.concatenate([D_neo_e_axis, D_neo_e_bulk])

  # Convective part of particle flux, apart from the Ware Pinch term
  # V*n*dpsi/rho = (L00+L01)*dlog(Te)/dpsi + (1-Rpe)/Rpe*L00*dlog(ni)/dpsi +
  # (1-Rpe)/Rpe * (L00+alpha*L03) *dlog(Ti)/dpsi
  V_neo_e_bulk = (
      (Lmn_e[1:, 0, 0] + Lmn_e[1:, 0, 1]) * dlnte_dpsi[1:]
      + (1 - Rpe[1:]) / Rpe[1:] * Lmn_e[1:, 0, 0] * dlnni_dpsi[1:]
      + (1 - Rpe[1:]) / Rpe[1:] * (Lmn_e[1:, 0, 0] + alpha[1:] * Lmn_e[1:, 0, 3]) * dlnti_dpsi[1:]
  ) / (
      dpsi_drhon[1:] / geometry.rho_b * core_profiles.n_e.face_value()[1:]
      + constants.CONSTANTS.eps
  )
  V_neo_e_axis = jnp.array([V_neo_e_bulk[0]])
  V_neo_e = jnp.concatenate([V_neo_e_axis, V_neo_e_bulk])

  # Ware pinch term component of particle convection
  # V_ware*n*dpsi/rho = L02*<E_parallel * B>/<B^2>
  V_neo_ware_e_bulk = (
      Lmn_e[1:, 0, 2]
      * E_parallel[1:]
      / (
          geometry.B_0
          * (dpsi_drhon[1:] / geometry.rho_b)
          * core_profiles.n_e.face_value()[1:]
          + constants.CONSTANTS.eps
      )
  )
  V_neo_ware_e_axis = jnp.atleast_1d(V_neo_ware_e_bulk[0])
  V_neo_ware_e = jnp.concatenate([V_neo_ware_e_axis, V_neo_ware_e_bulk])

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
) -> tuple[array_typing.Array, array_typing.Array]:
  """Calculates the dimensional transport matrices Lmn."""
  # Normalization factors from Eqs. 16, 20, 21
  consts = constants.CONSTANTS
  thermal_velocity_e = jnp.sqrt(
      2 * core_profiles.T_e.face_value() * consts.keV_to_J / consts.m_e
  )
  collision_time_e = (core_profiles.q_face * geo.R_major_profile_face) / (
      nu_e_star * epsilon**1.5 * thermal_velocity_e + consts.eps
  )
  thermal_velocity_i = jnp.sqrt(
      2
      * core_profiles.T_i.face_value()
      * consts.keV_to_J
      / (core_profiles.A_i * consts.m_amu)
  )
  collision_time_i = (core_profiles.q_face * geo.R_major_profile_face) / (
      nu_i_star * epsilon**1.5 * thermal_velocity_i + consts.eps
  )

  r_larmor_e = consts.m_e * thermal_velocity_e / consts.q_e
  r_larmor_i = (
      consts.m_amu
      * core_profiles.A_i
      * thermal_velocity_i
      / (consts.q_e * core_profiles.Z_i_face)
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
      * (consts.q_e * core_profiles.Z_i_face) ** 2
      * collision_time_i
      * geo.B_0**2
      / (
          consts.m_amu
          * core_profiles.A_i
          * core_profiles.T_i.face_value()
          * consts.keV_to_J
      )
  )

  # Calculate electron matrix (Eq. 20)
  Lmn_e = jnp.zeros((nu_e_star.shape[0], 4, 4))

  Lmn_e = Lmn_e.at[:, 0, 0].set(Kmn_e[:, 0, 0] * Ld * geo.gm4_face * geo.B_0**2)
  Lmn_e = Lmn_e.at[:, 0, 1].set(Kmn_e[:, 0, 1] * Ld * geo.gm4_face * geo.B_0**2)
  Lmn_e = Lmn_e.at[:, 0, 2].set(Kmn_e[:, 0, 2] * Lb)
  Lmn_e = Lmn_e.at[:, 0, 3].set(Kmn_e[:, 0, 3] * Ld / geo.gm5_face * geo.B_0**2)

  Lmn_e = Lmn_e.at[:, 1, 0].set(Lmn_e[:, 0, 1])
  Lmn_e = Lmn_e.at[:, 1, 1].set(Kmn_e[:, 1, 1] * Ld * geo.gm4_face * geo.B_0**2)
  Lmn_e = Lmn_e.at[:, 1, 2].set(Kmn_e[:, 1, 2] * Lb)
  Lmn_e = Lmn_e.at[:, 1, 3].set(Kmn_e[:, 1, 3] * Ld / geo.gm5_face * geo.B_0**2)

  Lmn_e = Lmn_e.at[:, 2, 0].set(Lmn_e[:, 0, 2])
  Lmn_e = Lmn_e.at[:, 2, 1].set(Lmn_e[:, 1, 2])
  # Lmn_e[:, 2 ,2] is not used
  Lmn_e = Lmn_e.at[:, 2, 3].set(Kmn_e[:, 2, 3] * Lb)

  Lmn_e = Lmn_e.at[:, 3, 0].set(Lmn_e[:, 0, 3])
  Lmn_e = Lmn_e.at[:, 3, 1].set(Lmn_e[:, 1, 3])
  Lmn_e = Lmn_e.at[:, 3, 2].set(Lmn_e[:, 2, 3])
  Lmn_e = Lmn_e.at[:, 3, 3].set(Kmn_e[:, 3, 3] * Ld / geo.gm5_face * geo.B_0**2)

  # Calculate ion matrix
  Lmn_i = jnp.zeros((nu_i_star.shape[0], 2, 2))
  Lmn_i = Lmn_i.at[:, 0, 0].set(
      Kmn_i[:, 0, 0] * Lsi * geo.gm5_face / geo.B_0**2
  )
  Lmn_i = Lmn_i.at[:, 0, 1].set(Kmn_i[:, 0, 1] * Lbi)
  Lmn_i = Lmn_i.at[:, 1, 0].set(-Lmn_i[:, 1, 0])
  Lmn_i = Lmn_i.at[:, 1, 1].set(
      Kmn_i[:, 1, 1] * Ldi * geo.gm4_face * geo.B_0**2
  )

  return Lmn_e, Lmn_i


def _calculate_shaing_transport(
    runtime_params: runtime_params_lib.RuntimeParams,
    geometry: geometry_lib.Geometry,
    core_profiles: state.CoreProfiles,
) -> base.NeoclassicalTransport:
  """JIT-compatible implementation of the Shaing transport model.

  Currently only implements near-axis ion thermal transport. Other contributions
  are negligible.

  From K. C. Shaing, R. D. Hazeltine, M. C. Zarnstorff,
  Phys. Plasmas 4, 771-777 (1997)
  https://doi.org/10.1063/1.872171

  Args:
    runtime_params: Runtime parameters.
    geometry: Geometry object.
    core_profiles: Core profiles object.

  Returns:
    Neoclassical transport coefficients.
  """
  # Aliases for readability
  m_ion = core_profiles.A_i * constants.CONSTANTS.m_amu
  q = core_profiles.q_face
  kappa = geometry.elongation_face  # Note: denoted delta in Shaing
  F = geometry.F_face  # Note: denoted I in Shaing
  R = geometry.R_major_profile_face
  T_i_J = core_profiles.T_i.face_value() * constants.CONSTANTS.keV_to_J

  # Collisionality
  ln_Lambda_ii = collisions.calculate_log_lambda_ii(
      core_profiles.T_i.face_value(),
      core_profiles.n_i.face_value(),
      core_profiles.Z_i_face,
  )
  tau_ii = collisions.calculate_tau_ii(
      A_i=core_profiles.A_i,
      Z_i=core_profiles.Z_i_face,
      T_i=core_profiles.T_i.face_value(),
      n_i=core_profiles.n_i.face_value(),
      ln_Lambda_ii=ln_Lambda_ii,
  )
  nu_ii = 1 / tau_ii  # Ion-ion collision frequency

  # Thermal velocity
  v_t_ion = jnp.sqrt(2 * T_i_J / m_ion)

  # Larmor (gyro)frequency
  Omega_0_ion = (
      constants.CONSTANTS.q_e * core_profiles.Z_i_face * geometry.B_0 / m_ion
  )

  # Large aspect ratio approximation (Equation 3, Shaing March 1997)
  C_1 = (2 * q / (kappa * F * R)) ** (1 / 2)

  # Conversion from flux^2/s -> m^2/s
  # TODO(b/467357743): make a more informed choice for dpsi_drhon near the axis
  # (currently we simply copy the value at i=1). This is ok as chi[0] is never
  # used.
  dpsi_drhon = core_profiles.psi.face_grad()
  dpsi_drhon = dpsi_drhon.at[0].set(dpsi_drhon[1])
  conversion_factor = 1 / (dpsi_drhon / (2 * jnp.pi * geometry.rho_b)) ** 2

  # Trapped particle fraction (Equation 46, Shaing March 1997)
  f_t_ion = (F * v_t_ion * C_1**2 / Omega_0_ion) ** (1 / 3)

  # Orbit width in psi coordinates (Equation 73, Shaing March 1997)
  Delta_psi_ion = (F**2 * v_t_ion**2 * C_1 / Omega_0_ion**2) ** (2 / 3)

  # Chi i term (Equation 74, Shaing March 1997)
  # psi normalization difference accounted for in conversion_factor
  chi_i = (nu_ii * Delta_psi_ion**2 / f_t_ion) * conversion_factor

  # Needed for pytype.
  assert isinstance(runtime_params.neoclassical.transport, RuntimeParams)

  return base.NeoclassicalTransport(
      chi_neo_i=runtime_params.neoclassical.transport.shaing_ion_multiplier
      * chi_i,
      chi_neo_e=jnp.zeros_like(geometry.rho_face),
      D_neo_e=jnp.zeros_like(geometry.rho_face),
      V_neo_e=jnp.zeros_like(geometry.rho_face),
      V_neo_ware_e=jnp.zeros_like(geometry.rho_face),
  )


def _calculate_blend_alpha(
    rho_face_norm: array_typing.FloatVectorFace,
    start: array_typing.FloatScalar,
    rate: array_typing.FloatScalar,
) -> array_typing.FloatVectorFace:
  """Calculate blending weight between Angioni-Sauter and Shaing models.

  The blend is:
    result = (1-alpha)*Shaing + alpha*Angioni-Sauter
  where alpha = 1 / (1 + exp(-2*rate*(rho_face_norm - start))).

  This gives:
    - At axis (rho_face_norm = 0 << start): alpha ~ 0 (pure Shaing)
    - At start: alpha = 0.5 (equal blend)
    - Far from axis (rho_face_norm >> start): alpha ~ 1 (pure Angioni-Sauter)

  Args:
    rho_face_norm: Normalized toroidal flux coordinate (face grid)
    start: Rho norm value where blend transition is centered
    rate: Controls transition steepness (higher = sharper transition)

  Returns:
    Blend factor alpha in range [0, 1]
  """
  return 1.0 / (1.0 + jnp.exp(-2.0 * rate * (rho_face_norm - start)))

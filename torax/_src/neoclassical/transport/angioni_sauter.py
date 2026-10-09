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
from typing import Annotated, Final, Literal, override

import jax
from jax import numpy as jnp
import pydantic
from torax._src import array_typing
from torax._src import constants
from torax._src import math_utils
from torax._src import state
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical.formulas import formulas
from torax._src.neoclassical.formulas import sauter as sauter_formulas
from torax._src.neoclassical.transport import base
from torax._src.neoclassical.transport import runtime_params as transport_runtime_params
from torax._src.physics import collisions
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import transport_coeffs


# pylint: disable=invalid-name


# Denominators in neoclassical transport coefficient extractions scale with
# plasma density (n_e, n_i ~ 1e19 - 1e20 m^-3) and poloidal flux gradient
# (dpsi/dr ~ 1 Wb/m), giving typical denominator magnitudes of order
# ~1e20 in SI units. Choosing eps ~ 1e12 ensures a relative error < 1e-7 in the
# 'normal' range of values, while providing a finite numerical ceiling if a
# denominator vanishes.
_SAFE_DIVIDE_EPS: Final[float] = 1e12


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
    base_params = super().build_runtime_params()
    return RuntimeParams(
        use_shaing_ion_correction=self.use_shaing_ion_correction,
        shaing_ion_multiplier=self.shaing_ion_multiplier,
        shaing_blend_start=self.shaing_blend_start,
        shaing_blend_rate=self.shaing_blend_rate,
        **vars(base_params),
    )


class AngioniSauterModel(base.NeoclassicalTransportModel):
  """Implements the Angioni-Sauter neoclassical transport model."""

  @override
  def _call_implementation(
      self,
      runtime_params: transport_runtime_params.RuntimeParams,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
      neoclassical_intermediates: formulas.NeoclassicalIntermediates,
  ) -> transport_coeffs.NeoclassicalTransport:
    """Calculates neoclassical transport coefficients.

    When use_shaing_ion_correction is enabled, chi_ion is smoothly blended
    between Shaing (near axis) and Angioni-Sauter (far from axis) models using
    an exponential transition function.

    Args:
      runtime_params: Runtime parameters.
      geometry: Geometry object.
      core_profiles: Core profiles object.
      neoclassical_intermediates: Precomputed intermediate quantities.

    Returns:
      Neoclassical transport coefficients.
    """
    assert isinstance(runtime_params, RuntimeParams)
    angioni_sauter = _calculate_angioni_sauter_transport(
        geometry=geometry,
        core_profiles=core_profiles,
        neoclassical_intermediates=neoclassical_intermediates,
    )
    shaing = _calculate_shaing_transport(
        runtime_params=runtime_params,
        geometry=geometry,
        core_profiles=core_profiles,
        neoclassical_intermediates=neoclassical_intermediates,
    )

    # Calculate sigmoid blend weight for Angioni-Sauter (alpha)
    # If correction disabled: alpha = 1 (pure Angioni-Sauter)
    # If correction enabled: alpha varies smoothly with rho_norm
    alpha = jnp.where(
        runtime_params.use_shaing_ion_correction,
        _calculate_blend_alpha(
            rho_face_norm=geometry.rho_face_norm,
            start=runtime_params.shaing_blend_start,
            rate=runtime_params.shaing_blend_rate,
        ),
        1.0,  # Pure Angioni-Sauter when correction disabled
    )

    return transport_coeffs.NeoclassicalTransport(
        # Ion transport blend: (1-alpha)*Shaing + alpha*Angioni-Sauter
        chi_face_ion=(1.0 - alpha) * shaing.chi_face_ion
        + alpha * angioni_sauter.chi_face_ion,
        # Electron transport: pure Angioni-Sauter
        chi_face_el=angioni_sauter.chi_face_el,
        d_face_el=angioni_sauter.d_face_el,
        v_face_el=angioni_sauter.v_face_el,
        v_face_el_ware=angioni_sauter.v_face_el_ware,
    )

  def __hash__(self) -> int:
    return hash(self.__class__.__name__)

  def __eq__(self, other) -> bool:
    return isinstance(other, self.__class__)


def _calculate_angioni_sauter_transport(
    geometry: geometry_lib.Geometry,
    core_profiles: state.CoreProfiles,
    neoclassical_intermediates: formulas.NeoclassicalIntermediates,
) -> transport_coeffs.NeoclassicalTransport:
  """JIT-compatible implementation of the Angioni-Sauter transport model.

  Args:
    geometry: Geometry object.
    core_profiles: Core profiles object.
    neoclassical_intermediates: Precomputed intermediate quantities.

  Returns:
    Neoclassical transport coefficients.

  All internally assigned profiles are on the face grid. The face suffix is
  omitted for brevity. The Angioni-Sauter model uses poloidal flux psi in
  [Wb/rad], so TORAX's psi [Wb] is converted accordingly.
  """

  # --- Step 1: Calculate intermediate physics quantities ---

  # Calculate trapped fractions ft and ftd from paper Eq. (17)
  B2_avg_Bm2_avg = geometry.gm5_face * geometry.gm4_face
  ftrap = neoclassical_intermediates.f_trap

  # Equation (17)
  ftrap_d = 1.0 - (1.0 - ftrap) / B2_avg_Bm2_avg

  # Collisionalities
  nu_e_star = neoclassical_intermediates.nu_e_star
  log_lambda_ei = neoclassical_intermediates.log_lambda_ei
  log_lambda_ii = neoclassical_intermediates.log_lambda_ii

  # Equation (18c) from Sauter PoP 1999.
  # Note: Sauter Eq. (18c) is defined for a pure plasma (using Z_i^4); in
  # Angioni & Sauter (2000) Eq. (30h), nu_i_star is the pure main-ion
  # collisionality since impurity collisions enter separately via alpha_I.
  nu_i_star = formulas.calculate_nu_i_star(
      q=core_profiles.q_face,
      geo=geometry,
      n_i=core_profiles.n_i.face_value(),
      T_i=core_profiles.T_i.face_value(),
      Z_eff=core_profiles.Z_i_face,
      log_lambda_ii=log_lambda_ii,
  )

  # Impurity strength parameter alpha_I = n_I Z_I^2 / (n_i Z_i^2), defined
  # below Eq. 25. Since the bundled impurity has Z_impurity = <Z^2> / <Z>,
  # n_impurity * Z_impurity^2 = sum_k n_k Z_k^2 for impurity mixtures.
  alpha_I = (
      core_profiles.n_impurity.face_value()
      * core_profiles.Z_impurity_face**2
      / (core_profiles.n_i.face_value() * core_profiles.Z_i_face**2)
  )

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
      log_lambda_ei=log_lambda_ei,
      log_lambda_ii=log_lambda_ii,
  )

  # --- Step 4: Calculate thermodynamic forces ---
  # Convert to Angioni-Sauter psi units.
  dpsi_drhon = core_profiles.psi.face_grad() / (2 * jnp.pi)
  dlnne_dpsi = math_utils.safe_divide(
      num=core_profiles.n_e.face_grad() / core_profiles.n_e.face_value(),
      denom=dpsi_drhon,
      eps=1e-7,
  )
  dlnte_dpsi = math_utils.safe_divide(
      num=core_profiles.T_e.face_grad() / core_profiles.T_e.face_value(),
      denom=dpsi_drhon,
      eps=1e-7,
  )
  dlnni_dpsi = math_utils.safe_divide(
      num=core_profiles.n_i.face_grad() / core_profiles.n_i.face_value(),
      denom=dpsi_drhon,
      eps=1e-7,
  )
  dlnti_dpsi = math_utils.safe_divide(
      num=core_profiles.T_i.face_grad() / core_profiles.T_i.face_value(),
      denom=dpsi_drhon,
      eps=1e-7,
  )

  # --- Step 5: Calculate neoclassical fluxes ---
  pe = core_profiles.pressure_thermal_e.face_value()
  pi = core_profiles.pressure_thermal_i.face_value()
  Rpe = pe / (pe + pi)
  alpha = -Kmn_i[:, 0, 1]
  # <E_parallel * B> / <B^2> (Angioni & Sauter, 2000, Section V).
  E_parallel_B_over_B2 = (
      geometry.F_face
      * core_profiles.psidot.face_value()
      * geometry.g3_face
      / (2 * jnp.pi * geometry.gm5_face)
  )

  # Total electron heat flux Q_e = B_e2 * T_e / (dpsi/drho) (see Angioni Sec 5)
  Be2 = (
      Lmn_e[:, 1, 0] * dlnne_dpsi
      + (Lmn_e[:, 1, 0] + Lmn_e[:, 1, 1]) * dlnte_dpsi
      + (1 - Rpe) / Rpe * Lmn_e[:, 1, 0] * dlnni_dpsi
      + (1 - Rpe) / Rpe * (Lmn_e[:, 1, 0] + alpha * Lmn_e[:, 1, 3]) * dlnti_dpsi
      + Lmn_e[:, 1, 2] * E_parallel_B_over_B2
  )

  # Total ion heat flux Q_i = B_i2 * T_i / (dpsi/drho) (see Angioni Sec 5)
  # NOTE: Reproduces the published B_i2 (p. 1233), which has two minor
  # approximations when expanding Eq. (28b):
  # 1. It omits +alpha * (1 - Rpe) / Rpe * L_41^e * dln(T_i)/dpsi from the
  #    expansion of A_e1 (whose analogue is retained in B_en above).
  # 2. Eqs. (26–28) assume single-ion quasineutrality (n_e = Z_i * n_i) when
  #    eliminating A_i1, A_e4 via Eq. (2) and rewriting T_i / (Z_i * T_e) as
  #    (1 - Rpe) / Rpe, rather than keeping n_i / n_e for an impure plasma.
  # Since L_4m^e / L_22^i ~ O((m_e / m_i)**0.5), these affect chi_i by at most
  # ~3% (banana-regime deuterium), so the published form is kept as-is.
  Bi2 = (
      alpha * Lmn_e[:, 3, 0] * dlnne_dpsi
      + alpha * (Lmn_e[:, 3, 0] + Lmn_e[:, 3, 1]) * dlnte_dpsi
      + alpha * (1 - Rpe) / Rpe * Lmn_e[:, 3, 0] * dlnni_dpsi
      + alpha * Lmn_e[:, 3, 2] * E_parallel_B_over_B2
      + (
          Lmn_i[:, 1, 1]
          + (1 - Rpe) / Rpe * alpha**2 / core_profiles.Z_i_face * Lmn_e[:, 3, 3]
      )
      * dlnti_dpsi
  )

  # --- Step 6: Extract transport coefficients --- Angioni Section 5.
  # Q_e = - chi_e * <|grad rho|^2> * n_e * dT_e/drho = B_e2 * T_e / (dpsi/drho)
  # Q_i = - chi_i * <|grad rho|^2> * n_i * dT_i/drho = B_i2 * T_i / (dpsi/drho)

  # All transport quantities have constant extrapolation to the magnetic axis
  # to avoid division by near-zero and unphysical values.

  grad_rho_sq = geometry.rho_b**2 * geometry.g1_over_vpr2_face
  grad_rho = geometry.rho_b * geometry.g0_over_vpr_face

  chi_neo_e_bulk = math_utils.safe_divide(
      num=-Be2[1:],
      denom=(
          core_profiles.n_e.face_value()[1:]
          * dlnte_dpsi[1:]
          * (dpsi_drhon[1:] / geometry.rho_b) ** 2
          * grad_rho_sq[1:]
      ),
      eps=_SAFE_DIVIDE_EPS,
  )
  chi_neo_e = jnp.concatenate([chi_neo_e_bulk[0:1], chi_neo_e_bulk])

  chi_neo_i_bulk = math_utils.safe_divide(
      num=-Bi2[1:],
      denom=(
          core_profiles.n_i.face_value()[1:]
          * dlnti_dpsi[1:]
          * (dpsi_drhon[1:] / geometry.rho_b) ** 2
          * grad_rho_sq[1:]
      ),
      eps=_SAFE_DIVIDE_EPS,
  )
  chi_neo_i = jnp.concatenate([chi_neo_i_bulk[0:1], chi_neo_i_bulk])

  # Decomposition of particle flux Be1 = Gamma * dpsi_drho.
  # Reference: Angioni & Sauter (2000), Eq. (1a) and Section V (pp. 1232–1233).

  # Diffusive part of particle flux:
  # D_e * <|grad rho|^2> * dn_e/drho = - L00 * dlog(n_e)/dpsi / (dpsi/drho)
  D_neo_e_bulk = math_utils.safe_divide(
      num=-Lmn_e[1:, 0, 0],
      denom=(
          core_profiles.n_e.face_value()[1:]
          * (dpsi_drhon[1:] / geometry.rho_b) ** 2
          * grad_rho_sq[1:]
      ),
      eps=_SAFE_DIVIDE_EPS,
  )
  D_neo_e = jnp.concatenate([D_neo_e_bulk[0:1], D_neo_e_bulk])

  # Convective part of particle flux, apart from the Ware pinch term:
  # V * <|grad rho|> * n * dpsi/drho = (L00+L01)*dlog(Te)/dpsi +
  # (1-Rpe)/Rpe*L00*dlog(ni)/dpsi + (1-Rpe)/Rpe * (L00+alpha*L03) *dlog(Ti)/dpsi
  V_neo_e_bulk = math_utils.safe_divide(
      num=(
          (Lmn_e[1:, 0, 0] + Lmn_e[1:, 0, 1]) * dlnte_dpsi[1:]
          + (1 - Rpe[1:]) / Rpe[1:] * Lmn_e[1:, 0, 0] * dlnni_dpsi[1:]
          + (1 - Rpe[1:])
          / Rpe[1:]
          * (Lmn_e[1:, 0, 0] + alpha[1:] * Lmn_e[1:, 0, 3])
          * dlnti_dpsi[1:]
      ),
      denom=(
          (dpsi_drhon[1:] / geometry.rho_b)
          * grad_rho[1:]
          * core_profiles.n_e.face_value()[1:]
      ),
      eps=_SAFE_DIVIDE_EPS,
  )
  V_neo_e = jnp.concatenate([V_neo_e_bulk[0:1], V_neo_e_bulk])

  # Ware pinch term component of particle convection:
  # V_ware * <|grad rho|> * n * dpsi/drho = L02 * <E_parallel * B>/<B^2>
  V_neo_ware_e_bulk = math_utils.safe_divide(
      num=Lmn_e[1:, 0, 2] * E_parallel_B_over_B2[1:],
      denom=(
          (dpsi_drhon[1:] / geometry.rho_b)
          * grad_rho[1:]
          * core_profiles.n_e.face_value()[1:]
      ),
      eps=_SAFE_DIVIDE_EPS,
  )
  V_neo_ware_e = jnp.concatenate([V_neo_ware_e_bulk[0:1], V_neo_ware_e_bulk])

  return transport_coeffs.NeoclassicalTransport(
      chi_face_ion=chi_neo_i,
      chi_face_el=chi_neo_e,
      d_face_el=D_neo_e,
      v_face_el=V_neo_e + V_neo_ware_e,
      v_face_el_ware=V_neo_ware_e,
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

  # Eq. 30c (note: Angioni & Sauter 2000 Eq. 30c contains a sign erratum:
  # published as `- 6.25 * K11e_0`, but mathematically must be
  # `+ 6.25 * K11e_0`.
  # Proof: Eq. 30a defines K22e = H22 - 5*H12 + 6.25*H11. In the banana limit
  # (nu_e_star -> 0), Hmn -> Hmn_0. Substituting H11_0 = K11e_0 and
  # H12_0 = K12e_0 + 2.5*K11e_0 into Eq. 30a gives:
  #   K22e -> H22_0 - 5*(K12e_0 + 2.5*K11e_0) + 6.25*K11e_0
  #         = H22_0 - 5*K12e_0 - 6.25*K11e_0.
  # For K22e to reduce to K22e_0 in this limit, we must invert this as:
  #   H22_0 = K22e_0 + 5.0*K12e_0 + 6.25*K11e_0.
  # The published minus sign (also transcribed in NEOS as +3.125*F11 =
  # -6.25*K11e_0 at
  # https://gitlab.epfl.ch/spc/public/neos/-/blob/master/F90/neoothercoeffmod.f90#L354)
  # leaves an uncancelled -12.5*K11e_0, making K22e > 0 (and chi_e < 0)).
  H11_0 = K11e_0
  H12_0 = K12e_0 + 2.5 * K11e_0
  H22_0 = K22e_0 + 5.0 * K12e_0 + 6.25 * K11e_0

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
  Kmn_e = Kmn_e.at[:, 3, 3].set(H41)

  # Supplement K matrix with "bootstrap terms" needed for Ware pinch from the
  # Sauter model (PoP 1999)
  Kmn_e = Kmn_e.at[:, 0, 2].set(
      -sauter_formulas.calculate_L31(ftrap, nu_e_star, Z_eff)
  )
  Kmn_e = Kmn_e.at[:, 2, 0].set(Kmn_e[:, 0, 2])
  Kmn_e = Kmn_e.at[:, 1, 2].set(
      -sauter_formulas.calculate_L32(ftrap, nu_e_star, Z_eff)
  )
  Kmn_e = Kmn_e.at[:, 2, 1].set(Kmn_e[:, 1, 2])
  Kmn_e = Kmn_e.at[:, 2, 3].set(
      -sauter_formulas.calculate_L34(ftrap, nu_e_star, Z_eff)
  )
  Kmn_e = Kmn_e.at[:, 3, 2].set(Kmn_e[:, 2, 3])

  # Ion Kmn matrix:
  # Compute the banana-regime alpha_0 coefficient (Angioni & Sauter Eq. 25)
  # and interpolate across collisionality regimes to obtain alpha(nu_i_star)
  # following Section V and Sauter et al., Phys. Plasmas 6, 2834 (1999) Eq. 17b
  # (and erratum Phys. Plasmas 9, 5140 (2002)).
  alpha_0 = (
      -(0.62 + 1.5 * alpha_I)
      / (0.53 + alpha_I)
      * ((1.0 - ftrap) / (1.0 - 0.22 * ftrap - 0.19 * ftrap**2))
  )
  alpha = (
      (alpha_0 + 0.25 * (1.0 - ftrap**2) * jnp.sqrt(nu_i_star))
      / (1.0 + 0.5 * jnp.sqrt(nu_i_star))
      + 0.315 * nu_i_star**2 * ftrap**6
  ) / (1.0 + 0.15 * nu_i_star**2 * ftrap**6)

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
    log_lambda_ei: array_typing.FloatVectorFace,
    log_lambda_ii: array_typing.FloatVectorFace,
) -> tuple[array_typing.Array, array_typing.Array]:
  """Calculates the dimensional transport matrices Lmn."""
  # Normalization factors from Eqs. 16, 20, and 21.
  consts = constants.CONSTANTS
  m_i = core_profiles.A_i * consts.m_amu
  q_i = core_profiles.Z_i_face * consts.q_e

  collision_time_e = collisions.calculate_tau_e(
      T_e=core_profiles.T_e.face_value(),
      n_e=core_profiles.n_e.face_value(),
      Z_eff=core_profiles.Z_eff_face,
      ln_Lambda_ei=log_lambda_ei,
  )
  collision_time_i = collisions.calculate_tau_ii(
      A_i=core_profiles.A_i,
      Z_i=core_profiles.Z_i_face,
      T_i=core_profiles.T_i.face_value(),
      n_i=core_profiles.n_i.face_value(),
      ln_Lambda_ii=log_lambda_ii,
  )

  # Ld coefficients for species s are defined as
  #   Lds = (n_s rho_sp^2 / tau_s) * (dpsi/drho)^2   (Eq 16, Eq 21)
  # where rho_sp for species s is defined as
  #   rho_sp = sqrt(2 m_s T_s) / (q_s B_p0)
  # (Eq. 18, as in Hinton & Hazeltine 1976 Eq. 5.122)
  # with B_p0 = (B_0 / F) * (dpsi/drho) (defined below Eq. 18).
  # Simplifying,
  #   rho_sp^2 = (2 m_s T_s) / (q_s^2 B_p0^2)
  #            = 2 m_s T_s q_s^-2 B_0^-2 F^2 (dpsi/drho)^-2
  # so
  #   Lds = 2 n_s m_s T_s q_s^-2 B_0^-2 F^2 tau_s^-1,
  # where the (dpsi/drho)^2 terms have cancelled.
  Ld = (
      2
      * core_profiles.n_e.face_value()
      * consts.m_e
      * (core_profiles.T_e.face_value() * consts.keV_to_J)
      / consts.q_e**2
      / geo.B_0**2
      * geo.F_face**2
      / collision_time_e
  )
  Ldi = (
      2
      * core_profiles.n_i.face_value()
      * m_i
      * (core_profiles.T_i.face_value() * consts.keV_to_J)
      / q_i**2
      / geo.B_0**2
      * geo.F_face**2
      / collision_time_i
  )
  Lb = geo.F_face * core_profiles.n_e.face_value()
  Lbi = geo.F_face * core_profiles.n_i.face_value()

  Lsi = (
      core_profiles.n_i.face_value()
      * q_i**2
      * collision_time_i
      * geo.B_0**2
      / (m_i * core_profiles.T_i.face_value() * consts.keV_to_J)
  )

  # Calculate electron matrix (Eq. 20)
  Lmn_e = jnp.zeros((log_lambda_ei.shape[0], 4, 4))

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
  Lmn_i = jnp.zeros((log_lambda_ii.shape[0], 2, 2))
  Lmn_i = Lmn_i.at[:, 0, 0].set(
      Kmn_i[:, 0, 0] * Lsi * geo.gm5_face / geo.B_0**2
  )
  Lmn_i = Lmn_i.at[:, 0, 1].set(Kmn_i[:, 0, 1] * Lbi)
  Lmn_i = Lmn_i.at[:, 1, 0].set(-Lmn_i[:, 0, 1])
  Lmn_i = Lmn_i.at[:, 1, 1].set(
      Kmn_i[:, 1, 1] * Ldi * geo.gm4_face * geo.B_0**2
  )

  return Lmn_e, Lmn_i


def _calculate_shaing_transport(
    runtime_params: RuntimeParams,
    geometry: geometry_lib.Geometry,
    core_profiles: state.CoreProfiles,
    neoclassical_intermediates: formulas.NeoclassicalIntermediates,
) -> transport_coeffs.NeoclassicalTransport:
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
    neoclassical_intermediates: Precomputed intermediate quantities.

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
  ln_Lambda_ii = neoclassical_intermediates.log_lambda_ii
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
  dpsi_drhon = dpsi_drhon.at[0].set(dpsi_drhon[1])  # pyrefly: ignore[missing-attribute]
  conversion_factor = 1 / (dpsi_drhon / (2 * jnp.pi * geometry.rho_b)) ** 2

  # Trapped particle fraction (Equation 46, Shaing March 1997)
  f_t_ion = (F * v_t_ion * C_1**2 / Omega_0_ion) ** (1 / 3)

  # Orbit width in psi coordinates (Equation 73, Shaing March 1997)
  Delta_psi_ion = (F**2 * v_t_ion**2 * C_1 / Omega_0_ion**2) ** (2 / 3)

  # Chi i term (Equation 74, Shaing March 1997)
  # psi normalization difference accounted for in conversion_factor
  chi_i = (nu_ii * Delta_psi_ion**2 / f_t_ion) * conversion_factor

  return transport_coeffs.NeoclassicalTransport(
      chi_face_ion=runtime_params.shaing_ion_multiplier * chi_i,
      chi_face_el=jnp.zeros_like(geometry.rho_face),
      d_face_el=jnp.zeros_like(geometry.rho_face),
      v_face_el=jnp.zeros_like(geometry.rho_face),
      v_face_el_ware=jnp.zeros_like(geometry.rho_face),
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

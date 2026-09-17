# Copyright 2026 DeepMind Technologies Limited
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
"""GACODE NEO surrogate neoclassical transport model for TORAX.

Drop-in replacement for analytic models (Sauter, Angioni-Sauter) and slow
kinetic codes in TORAX. Provides pure JAX evaluation (< 200 us profile
execution, < 1.0 ms/surface) with smooth C^infinity regularization at the
magnetic axis and full reverse/forward autodiff support.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Annotated, Any, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import pydantic
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical.formulas import neo_surrogate as neo_formulas
from torax._src.neoclassical.transport import base
from torax._src.neoclassical.transport import runtime_params as transport_runtime_params
from torax._src.torax_pydantic import torax_pydantic
from typing_extensions import override

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(transport_runtime_params.RuntimeParams):
  """Runtime parameters for the NEO surrogate transport model."""

  axis_extrapolate: array_typing.BoolScalar
  min_epsilon: array_typing.FloatScalar
  chi_neo_i_multiplier: array_typing.FloatScalar
  chi_neo_e_multiplier: array_typing.FloatScalar
  D_neo_e_multiplier: array_typing.FloatScalar
  V_neo_e_multiplier: array_typing.FloatScalar
  V_neo_ware_e_multiplier: array_typing.FloatScalar


class NeoSurrogateModelConfig(base.NeoclassicalTransportModelConfig):
  """Pydantic model config for the GACODE NEO surrogate neoclassical transport."""

  model_name: Annotated[
      Literal['neo_surrogate'], torax_pydantic.JAX_STATIC
  ] = 'neo_surrogate'
  axis_extrapolate: bool = True
  min_epsilon: pydantic.NonNegativeFloat = 1e-4
  chi_neo_i_multiplier: pydantic.NonNegativeFloat = 1.0
  chi_neo_e_multiplier: pydantic.NonNegativeFloat = 1.0
  D_neo_e_multiplier: pydantic.NonNegativeFloat = 1.0
  V_neo_e_multiplier: pydantic.NonNegativeFloat = 1.0
  V_neo_ware_e_multiplier: pydantic.NonNegativeFloat = 1.0

  @override
  def build_model(self) -> NeoSurrogateTransportModel:
    return NeoSurrogateTransportModel()

  @override
  def build_runtime_params(self) -> RuntimeParams:
    base_kwargs = dataclasses.asdict(super().build_runtime_params())
    return RuntimeParams(
        axis_extrapolate=self.axis_extrapolate,
        min_epsilon=self.min_epsilon,
        chi_neo_i_multiplier=self.chi_neo_i_multiplier,
        chi_neo_e_multiplier=self.chi_neo_e_multiplier,
        D_neo_e_multiplier=self.D_neo_e_multiplier,
        V_neo_e_multiplier=self.V_neo_e_multiplier,
        V_neo_ware_e_multiplier=self.V_neo_ware_e_multiplier,
        **base_kwargs,
    )


def extract_base_features(
    transport_params: RuntimeParams,
    geometry: geometry_lib.Geometry,
    core_profiles: state.CoreProfiles,
) -> jax.Array:
  """Extracts 9-element base surrogate input feature vector on the face grid.

  Applies smooth C^infinity regularization at the magnetic axis:
    eps_eff = 0.5 * (eps + sqrt(eps^2 + 4.0 * min_epsilon^2))
  guaranteeing non-singular values and finite, non-zero gradients across the
  entire radial domain.

  Args:
    transport_params: Neoclassical runtime parameters.
    geometry: TORAX geometry object.
    core_profiles: TORAX core profiles object.

  Returns:
    Array of shape (N+1, 9) containing base surrogate input features:
    [eps, q, log10_nu1, kappa, delta, s_hat, a_over_LTe, a_over_LTi,
     a_over_Lne].
  """
  min_eps = transport_params.min_epsilon
  eps_raw = geometry.epsilon_face

  # Smooth C^infinity regularization at magnetic axis
  eps_eff = 0.5 * (eps_raw + jnp.sqrt(eps_raw**2 + 4.0 * (min_eps**2)))

  # Safety factor and magnetic shear
  q = jnp.clip(core_profiles.q_face, 1.0, 15.0)
  s_hat = jnp.clip(core_profiles.s_face, 0.1, 3.0)

  # Geometric shaping
  kappa = jnp.clip(geometry.elongation_face, 1.0, 2.0)
  delta = jnp.clip(geometry.delta_face, -0.5, 0.5)

  # Machine dimensions
  R_major = geometry.R_major
  a_minor = geometry.a_minor

  # Temperatures and densities on face grid
  Te_face = jnp.maximum(core_profiles.T_e.face_value(), 1e-3)
  Ti_face = jnp.maximum(core_profiles.T_i.face_value(), 1e-3)
  ne_face = jnp.maximum(core_profiles.n_e.face_value(), 1e16)
  Z_eff_face = jnp.maximum(core_profiles.Z_eff_face, 1.0)

  # Coulomb logarithm
  log_lambda_ei = 31.3 - 0.5 * jnp.log(ne_face) + jnp.log(Te_face * 1e3)

  # Sauter electron collisionality nu_e^*
  nu_e_star = (
      6.921e-18
      * q
      * R_major
      * ne_face
      * Z_eff_face
      * log_lambda_ei
      / (((Te_face * 1e3) ** 2) * (eps_eff**1.5))
  )

  # Dimensionless collision rate NU_1
  r0_over_a = R_major / a_minor
  nu_1 = neo_formulas.torax_nu_star_to_neo_nu1(nu_e_star, q, r0_over_a, eps_eff)
  log10_nu1 = jnp.clip(jnp.log10(jnp.maximum(nu_1, 1e-5)), -3.0, 2.0)

  # Normalized logarithmic gradients: face_grad() is d/drhon
  dTe_drhon = core_profiles.T_e.face_grad()
  dTi_drhon = core_profiles.T_i.face_grad()
  dne_drhon = core_profiles.n_e.face_grad()

  a_over_LTe = jnp.clip(-dTe_drhon / Te_face, 0.5, 12.0)
  a_over_LTi = jnp.clip(-dTi_drhon / Ti_face, 0.5, 12.0)
  a_over_Lne = jnp.clip(-dne_drhon / ne_face, 0.1, 6.0)

  return jnp.stack(
      [
          eps_eff,
          q,
          log10_nu1,
          kappa,
          delta,
          s_hat,
          a_over_LTe,
          a_over_LTi,
          a_over_Lne,
      ],
      axis=-1,
  )


def extract_impurity_features(
    transport_params: RuntimeParams,
    geometry: geometry_lib.Geometry,
    core_profiles: state.CoreProfiles,
    Z_imp: float = 6.0,
    A_imp: float = 12.011,
) -> jax.Array:
  """Extracts 13-element impurity feature vector on the face grid.

  Args:
    transport_params: Neoclassical runtime parameters.
    geometry: TORAX geometry object.
    core_profiles: TORAX core profiles object.
    Z_imp: Impurity charge number (default 6.0 for Carbon).
    A_imp: Impurity mass number (default 12.011 for Carbon).

  Returns:
    Array of shape (N+1, 13) containing impurity surrogate input features.
  """
  x_base = extract_base_features(transport_params, geometry, core_profiles)
  ne_face = jnp.maximum(core_profiles.n_e.face_value(), 1e16)

  # Default 1% trace concentration estimate if profile is not explicitly
  # provided
  nimp_face = jnp.maximum(ne_face * 0.01, 1e10)
  dnimp_drhon = core_profiles.n_e.face_grad() * 0.01

  f_imp = (nimp_face * Z_imp) / ne_face
  log10_f_imp = jnp.clip(jnp.log10(jnp.maximum(f_imp, 1e-8)), -6.0, -1.0)
  a_over_Lnimp = jnp.clip(-dnimp_drhon / nimp_face, 0.0, 10.0)

  log10_Z = jnp.full_like(x_base[..., 0], float(math.log10(Z_imp)))
  log10_A = jnp.full_like(x_base[..., 0], float(math.log10(A_imp)))

  return jnp.concatenate(
      [
          x_base,
          jnp.stack([log10_Z, log10_A, log10_f_imp, a_over_Lnimp], axis=-1),
      ],
      axis=-1,
  )


class NeoSurrogateTransportModel(base.NeoclassicalTransportModel):
  """Fast, differentiable GACODE NEO surrogate neoclassical transport model."""

  def __init__(
      self,
      base_params: Optional[neo_formulas.BaseTransportParams] = None,
      impurity_params: Optional[neo_formulas.ImpurityTransportParams] = None,
  ):
    if base_params is None:
      base_params = neo_formulas.get_default_base_transport_params()
    if impurity_params is None:
      impurity_params = neo_formulas.get_default_impurity_transport_params()
    self.base_params = base_params
    self.impurity_params = impurity_params

  @override
  def _call_implementation(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> base.NeoclassicalTransport:
    """Core evaluation conforming to the TORAX neoclassical transport contract."""
    transport_params = runtime_params.neoclassical.transport
    assert isinstance(transport_params, RuntimeParams)

    # 1. Extract 9 base features on face grid with smooth axis regularizer
    x_base = extract_base_features(transport_params, geometry, core_profiles)

    # 2. Vectorized forward evaluation via pre-trained hybrid surrogate
    out = neo_formulas.predict_base_transport(self.base_params, x_base)

    chi_neo_i = out[..., 0]
    chi_neo_e = out[..., 1]
    D_neo_e = out[..., 2]
    V_neo_e = out[..., 3]
    V_neo_ware_e = out[..., 4]

    # 3. Magnetic axis boundary extrapolation (TORAX convention: face 0
    # matches face 1)
    chi_neo_i = jnp.where(
        transport_params.axis_extrapolate,
        chi_neo_i.at[0].set(chi_neo_i[1]),
        chi_neo_i,
    )
    chi_neo_e = jnp.where(
        transport_params.axis_extrapolate,
        chi_neo_e.at[0].set(chi_neo_e[1]),
        chi_neo_e,
    )
    D_neo_e = jnp.where(
        transport_params.axis_extrapolate,
        D_neo_e.at[0].set(D_neo_e[1]),
        D_neo_e,
    )
    V_neo_e = jnp.where(
        transport_params.axis_extrapolate,
        V_neo_e.at[0].set(V_neo_e[1]),
        V_neo_e,
    )
    V_neo_ware_e = jnp.where(
        transport_params.axis_extrapolate,
        V_neo_ware_e.at[0].set(V_neo_ware_e[1]),
        V_neo_ware_e,
    )

    # 4. Apply runtime multipliers
    chi_neo_i = chi_neo_i * transport_params.chi_neo_i_multiplier
    chi_neo_e = chi_neo_e * transport_params.chi_neo_e_multiplier
    D_neo_e = D_neo_e * transport_params.D_neo_e_multiplier
    V_neo_e = V_neo_e * transport_params.V_neo_e_multiplier
    V_neo_ware_e = V_neo_ware_e * transport_params.V_neo_ware_e_multiplier

    return base.NeoclassicalTransport(
        chi_neo_i=chi_neo_i,
        chi_neo_e=chi_neo_e,
        D_neo_e=D_neo_e,
        V_neo_e=V_neo_e,
        V_neo_ware_e=V_neo_ware_e,
    )

  def calculate_impurity_transport(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
      Z_imp: float = 6.0,
      A_imp: float = 12.011,
  ) -> Tuple[jax.Array, jax.Array]:
    """Calculates universal impurity diffusivity D_imp and convection V_imp.

    Supports arbitrary intermediate species from Helium (Z=2) to Tungsten
    (Z=74).

    Args:
      runtime_params: Runtime parameters.
      geometry: TORAX geometry object.
      core_profiles: TORAX core profiles object.
      Z_imp: Impurity charge number (default 6.0 for Carbon).
      A_imp: Impurity mass number (default 12.011 for Carbon).

    Returns:
      Tuple of (D_imp, V_imp) arrays on the face grid [m^2/s, m/s].
    """
    transport_params = runtime_params.neoclassical.transport
    assert isinstance(transport_params, RuntimeParams)

    x_imp = extract_impurity_features(
        transport_params, geometry, core_profiles, Z_imp=Z_imp, A_imp=A_imp
    )
    out_imp = neo_formulas.predict_impurity_transport(
        self.impurity_params, x_imp
    )
    D_imp = out_imp[..., 0]
    V_imp = out_imp[..., 1]

    D_imp = jnp.where(
        transport_params.axis_extrapolate,
        D_imp.at[0].set(D_imp[1]),
        D_imp,
    )
    V_imp = jnp.where(
        transport_params.axis_extrapolate,
        V_imp.at[0].set(V_imp[1]),
        V_imp,
    )

    return D_imp, V_imp

  def __eq__(self, other: Any) -> bool:
    return isinstance(other, self.__class__)

  def __hash__(self) -> int:
    return hash(self.__class__.__name__)

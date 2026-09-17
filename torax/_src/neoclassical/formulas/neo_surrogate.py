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
"""Physics-informed JAX neoclassical transport surrogate for GACODE NEO.

Hybrid physics-informed surrogate model:
- Model 1 (Base Transport): 9 inputs -> 7 neoclassical transport quantities
  (chi_neo_i, chi_neo_e, D_neo_e, V_neo_e, V_neo_ware_e, j_bootstrap, sigma)
- Model 2 (FACIT Impurity Transport): 13 inputs -> 2 impurity transport
  quantities (D_imp, V_imp) for arbitrary species from He (Z=2) to W (Z=74).

References:
- B.C. Belli & J. Candy, PPCF 50, 095010 (2008); PPCF 54, 015015 (2012)
- O. Sauter, C. Angioni, Y.R. Lin-Liu, Phys. Plasmas 6, 2834 (1999)
- C. Angioni and O. Sauter, Phys. Plasmas 7, 1224 (2000)
- C. Angioni and P. Helander, Plasma Phys. Control. Fusion 56, 124001 (2014)
"""

from __future__ import annotations

import base64
import dataclasses
import io
import math
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from torax._src.neoclassical.formulas import neo_surrogate_weights
from torax._src.neoclassical.formulas import sauter as sauter_formulas

# pylint: disable=invalid-name

# Physical constants matching GACODE NEO and TORAX conventions
Q_E: float = 1.602176634e-19       # Elementary charge [C]
M_AMU: float = 1.6605390666e-27    # Unified atomic mass unit [kg]
M_D: float = 2.0141018 * M_AMU     # Deuterium reference mass [kg]
M_E: float = 9.1093837e-31         # Electron mass [kg]
M_E_OVER_M_D: float = M_E / M_D    # Mass ratio ~ 2.7237e-4

# Constant ln(e - 1) ensuring softplus(0 + C_SOFTPLUS_OFFSET) == 1.0
C_SOFTPLUS_OFFSET: float = float(math.log(math.e - 1.0))

DEFAULT_A_ION: float = 2.0141   # Deuterium mass number [amu]
DEFAULT_Z_ION: float = 1.0      # Deuterium charge [dimensionless]
DEFAULT_R0_OVER_A: float = 3.0  # Aspect ratio normalisation reference


def safe_sqrt(x: jax.Array) -> jax.Array:
  """Computes sqrt(x) with safe zero gradient at x <= 0."""
  safe_x = jnp.where(x <= 0.0, jnp.ones_like(x), x)
  safe_sqrt_out = jnp.sqrt(safe_x)
  return jnp.where(x <= 0.0, jnp.zeros_like(x), safe_sqrt_out)


def scaled_softplus(
    x: jax.Array,
    floor: float = 1e-6,
    scale: float = 1e-3,
) -> jax.Array:
  """Scaled softplus guaranteeing strict positivity without an artificial pedestal."""
  return floor + scale * jax.nn.softplus((x - floor) / scale)


def calculate_f_trap_miller(
    eps: jax.Array,
    delta: jax.Array,
) -> jax.Array:
  """Calculates effective trapped particle fraction in Miller geometry."""
  eps_eff = 0.67 * (1.0 - 1.4 * jnp.abs(delta) * delta) * eps
  aa = (1.0 - eps) / (1.0 + eps)
  return 1.0 - jnp.sqrt(jnp.maximum(aa, 0.0)) * (1.0 - eps_eff) / (
      1.0 + 2.0 * safe_sqrt(eps_eff)
  )


def calculate_sigma_ratio(
    f_trap: jax.Array,
    nu_e_star: jax.Array,
    Z_eff: jax.Array,
) -> jax.Array:
  """Calculates neoclassical conductivity ratio sigma_neo / sigma_Spitzer."""
  sqrt_nu = safe_sqrt(nu_e_star)
  z_pow_15 = Z_eff**1.5
  denom_33 = (
      1.0
      + (0.55 - 0.1 * f_trap) * sqrt_nu
      + 0.45 * (1.0 - f_trap) * nu_e_star / z_pow_15
  )
  f_t33 = f_trap / denom_33
  poly = 1.0 + 0.36 / Z_eff - f_t33 * (0.59 / Z_eff - (0.23 / Z_eff) * f_t33)
  return 1.0 - f_t33 * poly


def neo_nu1_to_torax_nu_star(
    nu_1: Any,
    q: Any,
    r_0_over_a: Any,
    epsilon: Any,
) -> Any:
  """Converts NEO normalized collision rate NU_1 to TORAX collisionality nu_e^*."""
  conversion_factor = math.sqrt(M_E_OVER_M_D / 2.0)
  return conversion_factor * nu_1 * q * r_0_over_a * (epsilon**(-1.5))


def torax_nu_star_to_neo_nu1(
    nu_e_star: Any,
    q: Any,
    r_0_over_a: Any,
    epsilon: Any,
) -> Any:
  """Converts TORAX electron collisionality nu_e^* to NEO input parameter NU_1."""
  conversion_factor = math.sqrt(M_E_OVER_M_D / 2.0)
  return nu_e_star / (conversion_factor * q * r_0_over_a * (epsilon**(-1.5)))


# ---------------------------------------------------------------------------
# Physics Backbones
# ---------------------------------------------------------------------------


def evaluate_physics_backbone(x: jax.Array) -> jax.Array:
  """Evaluates analytical neoclassical transport for 9D input.

  Args:
    x: 9-element array of shape (..., 9):
       [eps, q, log10_nu1, kappa, delta, s_hat, a_over_LTe, a_over_LTi,
        a_over_Lne]

  Returns:
    Array of shape (..., 7) containing:
    [chi_neo_i, chi_neo_e, d_neo_e, v_neo_e, v_neo_ware_e, j_bootstrap, sigma].
  """
  eps = x[..., 0]
  q = x[..., 1]
  log10_nu1 = x[..., 2]
  kappa = x[..., 3]
  delta = x[..., 4]
  a_over_LTe = x[..., 6]
  a_over_LTi = x[..., 7]
  a_over_Lne = x[..., 8]

  nu1 = 10.0**log10_nu1
  r0_over_a = DEFAULT_R0_OVER_A
  nu_e_star = neo_nu1_to_torax_nu_star(nu1, q, r0_over_a, eps)
  nu_i_star = nu_e_star

  f_trap = calculate_f_trap_miller(eps, delta)

  z_eff = jnp.ones_like(eps)
  l31 = sauter_formulas.calculate_L31(f_trap, nu_e_star, z_eff)
  l32 = sauter_formulas.calculate_L32(f_trap, nu_e_star, z_eff)
  l34 = sauter_formulas.calculate_L34(f_trap, nu_e_star, z_eff)
  alpha = sauter_formulas.calculate_alpha(f_trap, nu_i_star)
  sigma_ratio = calculate_sigma_ratio(f_trap, nu_e_star, z_eff)

  sqrt_nu_i = safe_sqrt(nu_i_star)
  sqrt_nu_e = safe_sqrt(nu_e_star)
  eps_15 = eps**1.5
  s_geom = 2.0 / (1.0 + kappa**2)

  # Ion heat transport matrix element K22
  k2_i = (0.66 * f_trap) / (
      1.0 + 1.03 * sqrt_nu_i + 0.31 * nu_i_star
  ) + (0.46 * eps_15 * nu_i_star) / (1.0 + 0.74 * nu_i_star)
  chi_neo_i_raw = (
      (q**2)
      * (1.0 / jnp.maximum(eps_15, 1e-4))
      * k2_i
      * 0.01
      * s_geom
  )
  chi_neo_i = scaled_softplus(chi_neo_i_raw, floor=1e-6, scale=1e-3)

  # Electron heat transport
  k2_e = (0.66 * f_trap) / (
      1.0 + 1.03 * sqrt_nu_e + 0.31 * nu_e_star
  ) + (0.46 * eps_15 * nu_e_star) / (1.0 + 0.74 * nu_e_star)
  chi_neo_e_raw = (
      (q**2)
      * (1.0 / jnp.maximum(eps_15, 1e-4))
      * k2_e
      * 0.000233
      * s_geom
  )
  chi_neo_e = scaled_softplus(chi_neo_e_raw, floor=1e-6, scale=1e-3)

  # Electron particle diffusivity
  d_coll = f_trap / (1.0 + sqrt_nu_e + 0.5 * nu_e_star) + eps_15 * nu_e_star / (
      1.0 + nu_e_star
  )
  d_neo_e_raw = 0.005 * (q**2) * d_coll * s_geom
  d_neo_e = scaled_softplus(d_neo_e_raw, floor=1e-6, scale=1e-3)

  # Convection and Ware pinch
  v_neo_e = -d_neo_e * (0.5 * a_over_LTe - 0.2 * a_over_Lne)
  v_neo_ware_e = -0.5 * l31

  # Bootstrap current
  j_bootstrap = (
      l31 * a_over_Lne
      + (l31 + l32) * a_over_LTe
      + (l31 + alpha * l34) * a_over_LTi
  )
  sigma = sigma_ratio

  return jnp.stack(
      [
          chi_neo_i,
          chi_neo_e,
          d_neo_e,
          v_neo_e,
          v_neo_ware_e,
          j_bootstrap,
          sigma,
      ],
      axis=-1,
  )


def evaluate_impurity_physics_backbone(x: jax.Array) -> jax.Array:
  """Evaluates analytical neoclassical impurity transport from 13D inputs.

  Args:
    x: 13-element array of shape (..., 13):
       [eps, q, log10_nu1, kappa, delta, s_hat, a_over_LTe, a_over_LTi,
        a_over_Lne, log10_Z, log10_A, log10_f_imp, a_over_Lnimp]

  Returns:
    Array of shape (..., 2) containing [D_imp, V_imp] in [m^2/s, m/s].
  """
  eps = x[..., 0]
  q = x[..., 1]
  log10_nu1 = x[..., 2]
  kappa = x[..., 3]
  delta = x[..., 4]
  a_over_LTi = x[..., 7]
  a_over_Lne = x[..., 8]
  log10_Z = x[..., 9]
  log10_A = x[..., 10]
  log10_f_imp = x[..., 11]
  a_over_Lnimp = x[..., 12]

  Z_imp = 10.0**log10_Z
  A_imp = 10.0**log10_A
  f_imp = 10.0**log10_f_imp

  nu1 = 10.0**log10_nu1
  nu_e_star = neo_nu1_to_torax_nu_star(nu1, q, DEFAULT_R0_OVER_A, eps)
  nu_i_star = nu_e_star
  sqrt_nu_i = safe_sqrt(nu_i_star)
  eps_15 = eps**1.5

  f_trap = calculate_f_trap_miller(eps, delta)
  s_geom = 2.0 / (1.0 + kappa**2)

  one_minus_f = jnp.maximum(1.0 - f_imp, 1e-4)
  alpha_z = (f_imp * Z_imp) / (DEFAULT_Z_ION**2 * one_minus_f)
  a_over_Lni = (a_over_Lne - f_imp * a_over_Lnimp) / one_minus_f

  sqrt_eps = safe_sqrt(eps)
  k_ch = 1.0 + 0.03 * sqrt_eps - 0.02 * eps
  k_bp = (0.66 * f_trap) / (1.0 + 1.03 * sqrt_nu_i + 0.31 * nu_i_star) * k_ch
  k_ps = (
      (0.46 * eps_15 * nu_i_star)
      / (1.0 + 0.74 * nu_i_star)
      * (1.0 + 0.33 * alpha_z / (1.0 + 0.5 * alpha_z))
  )

  d_imp_raw = (
      0.008
      * (q**2)
      * s_geom
      * (k_bp / jnp.maximum(eps_15, 1e-4) + k_ps)
      * (1.0 + 0.1 * alpha_z)
      * ((2.0 / A_imp)**0.2)
  )
  D_imp = scaled_softplus(d_imp_raw, floor=1e-6, scale=1e-3)

  C_n = -Z_imp / DEFAULT_Z_ION
  c_bp_t = 0.5 * Z_imp * f_trap / (1.0 + sqrt_nu_i)
  c_ps_t = -0.5 / (1.0 + 0.5 * alpha_z)
  C_T = (c_bp_t + c_ps_t * nu_i_star) / (1.0 + nu_i_star)

  V_imp = D_imp * (C_n * a_over_Lni + C_T * a_over_LTi)

  return jnp.stack([D_imp, V_imp], axis=-1)


# ---------------------------------------------------------------------------
# Neural Network PyTrees & Forward Inference
# ---------------------------------------------------------------------------


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DenseLayer:
  """Single dense affine transformation layer."""
  weight: jax.Array  # shape: (in_dim, out_dim)
  bias: jax.Array    # shape: (out_dim,)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BaseTransportParams:
  """Parameter tree for Model 1 Base Neoclassical Transport Surrogate."""
  layer1: DenseLayer
  layer2: DenseLayer
  layer_out: DenseLayer
  x_mean: jax.Array  # shape: (9,)
  x_std: jax.Array   # shape: (9,)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ImpurityTransportParams:
  """Parameter tree for Model 2 Universal Impurity Transport Surrogate."""
  layer1: DenseLayer
  layer2: DenseLayer
  layer_out: DenseLayer
  x_mean: jax.Array  # shape: (13,)
  x_std: jax.Array   # shape: (13,)


def evaluate_mlp_correction(
    params: BaseTransportParams,
    x: jax.Array,
    activation: Callable[[jax.Array], jax.Array] = jax.nn.silu,
) -> jax.Array:
  """Evaluates the 7 geometric residual correction terms from normalized inputs."""
  x_norm = (x - params.x_mean) / params.x_std
  h1 = activation(jnp.dot(x_norm, params.layer1.weight) + params.layer1.bias)
  h2 = activation(jnp.dot(h1, params.layer2.weight) + params.layer2.bias)
  return jnp.dot(h2, params.layer_out.weight) + params.layer_out.bias


def predict_base_transport(
    params: BaseTransportParams,
    x: jax.Array,
) -> jax.Array:
  """Vectorized forward pass for Model 1 Base Transport Surrogate.

  Args:
    params: BaseTransportParams PyTree.
    x: Input array of shape (..., 9).

  Returns:
    Array of shape (..., 7) containing the 7 transport quantities.
  """
  y_phys = evaluate_physics_backbone(x)
  mlp_corr = evaluate_mlp_correction(params, x)
  pos_multiplier = jax.nn.softplus(mlp_corr + C_SOFTPLUS_OFFSET)
  return y_phys * pos_multiplier


def evaluate_impurity_mlp_correction(
    params: ImpurityTransportParams,
    x: jax.Array,
    activation: Callable[[jax.Array], jax.Array] = jax.nn.silu,
) -> jax.Array:
  """Evaluates the 2 residual corrections for impurity particle transport."""
  x_norm = (x - params.x_mean) / params.x_std
  h1 = activation(jnp.dot(x_norm, params.layer1.weight) + params.layer1.bias)
  h2 = activation(jnp.dot(h1, params.layer2.weight) + params.layer2.bias)
  return jnp.dot(h2, params.layer_out.weight) + params.layer_out.bias


def predict_impurity_transport(
    params: ImpurityTransportParams,
    x: jax.Array,
) -> jax.Array:
  """Vectorized forward pass for Model 2 Impurity Transport Surrogate.

  Args:
    params: ImpurityTransportParams PyTree.
    x: Input array of shape (..., 13).

  Returns:
    Array of shape (..., 2) containing [D_imp, V_imp] in [m^2/s, m/s].
  """
  y_phys = evaluate_impurity_physics_backbone(x)
  mlp_corr = evaluate_impurity_mlp_correction(params, x)
  D_phys = y_phys[..., 0]
  V_phys = y_phys[..., 1]
  D_imp = scaled_softplus(
      D_phys * jax.nn.softplus(mlp_corr[..., 0] + C_SOFTPLUS_OFFSET),
      floor=1e-6,
      scale=1e-3,
  )
  V_imp = V_phys + mlp_corr[..., 1]
  return jnp.stack([D_imp, V_imp], axis=-1)


# ---------------------------------------------------------------------------
# Default Weights Loading
# ---------------------------------------------------------------------------

_DEFAULT_BASE_PARAMS: Optional[BaseTransportParams] = None
_DEFAULT_IMPURITY_PARAMS: Optional[ImpurityTransportParams] = None


def get_default_base_transport_params() -> BaseTransportParams:
  """Retrieves or lazily unpacks pre-trained weights for Base Transport."""
  global _DEFAULT_BASE_PARAMS
  if _DEFAULT_BASE_PARAMS is None:
    raw_bytes = base64.b64decode(
        neo_surrogate_weights.BASE_TRANSPORT_WEIGHTS_B64.encode('ascii')
    )
    data = np.load(io.BytesIO(raw_bytes), allow_pickle=False)
    _DEFAULT_BASE_PARAMS = BaseTransportParams(
        layer1=DenseLayer(
            weight=jnp.array(data['w1'], dtype=jnp.float32),
            bias=jnp.array(data['b1'], dtype=jnp.float32),
        ),
        layer2=DenseLayer(
            weight=jnp.array(data['w2'], dtype=jnp.float32),
            bias=jnp.array(data['b2'], dtype=jnp.float32),
        ),
        layer_out=DenseLayer(
            weight=jnp.array(data['w_out'], dtype=jnp.float32),
            bias=jnp.array(data['b_out'], dtype=jnp.float32),
        ),
        x_mean=jnp.array(data['x_mean'], dtype=jnp.float32),
        x_std=jnp.array(data['x_std'], dtype=jnp.float32),
    )
  return _DEFAULT_BASE_PARAMS


def get_default_impurity_transport_params() -> ImpurityTransportParams:
  """Retrieves or lazily unpacks pre-trained weights for Impurity Transport."""
  global _DEFAULT_IMPURITY_PARAMS
  if _DEFAULT_IMPURITY_PARAMS is None:
    raw_bytes = base64.b64decode(
        neo_surrogate_weights.IMPURITY_TRANSPORT_WEIGHTS_B64.encode('ascii')
    )
    data = np.load(io.BytesIO(raw_bytes), allow_pickle=False)
    _DEFAULT_IMPURITY_PARAMS = ImpurityTransportParams(
        layer1=DenseLayer(
            weight=jnp.array(data['w1'], dtype=jnp.float32),
            bias=jnp.array(data['b1'], dtype=jnp.float32),
        ),
        layer2=DenseLayer(
            weight=jnp.array(data['w2'], dtype=jnp.float32),
            bias=jnp.array(data['b2'], dtype=jnp.float32),
        ),
        layer_out=DenseLayer(
            weight=jnp.array(data['w_out'], dtype=jnp.float32),
            bias=jnp.array(data['b_out'], dtype=jnp.float32),
        ),
        x_mean=jnp.array(data['x_mean'], dtype=jnp.float32),
        x_std=jnp.array(data['x_std'], dtype=jnp.float32),
    )
  return _DEFAULT_IMPURITY_PARAMS

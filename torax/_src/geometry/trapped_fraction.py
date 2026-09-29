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
"""Formulas for computing trapped particle fraction from magnetic geometry."""

# pylint: disable=invalid-name

import enum
import jax.numpy as jnp
import numpy as np
import scipy.interpolate
from torax._src import array_typing
from torax._src import math_utils


@enum.unique
class TrappedFractionSource(enum.StrEnum):
  """Selects how the effective trapped particle fraction is computed.

  Not every option is supported by every geometry source; see
  `BaseGeometryConfig._supported_trapped_fraction_sources`.

  Attributes:
    SAUTER: Uses the analytic approximation from [1]. Supported by all geometry
      sources.
    FILE: Reads the value precomputed by the input equilibrium/geometry code
      directly from the geometry file.
    EXACT: Computes the full bounce-averaged integral [2] directly from the 2D
      equilibrium. Only supported for EQDSK and IMAS sources.

  [1] O. Sauter, Fusion Engineering and Design 112 (2016) 633-645, Eqs 33+34.
  [2] Y. R. Lin-Liu, R. L. Miller, Phys. Plasmas 2(5) 1666-1668 (1995), Eq 1.
  """

  SAUTER = 'SAUTER'
  FILE = 'FILE'
  EXACT = 'EXACT'


# TODO(b/545148156): Add finite-orbit-width effects.
def calculate_sauter_trapped_fraction(
    epsilon: array_typing.Array, delta: array_typing.Array
) -> array_typing.Array:
  """Analytic approximation for the effective trapped particle fraction.

  From O. Sauter, Fusion Engineering and Design 112 (2016) 633-645, Eqs 33+34.

  Args:
    epsilon: Local midplane inverse aspect ratio of each flux surface.
    delta: Average triangularity of each flux surface.

  Returns:
    The effective trapped particle fraction of each flux surface.
  """
  epsilon_effective = 0.67 * (1.0 - 1.4 * jnp.abs(delta) * delta) * epsilon
  aa = (1.0 - epsilon) / (1.0 + epsilon)
  return 1.0 - jnp.sqrt(aa) * (1.0 - epsilon_effective) / (
      # On the magnetic axis, epsilon_effective is 0, in order to avoid a NaN
      # gradient we define the gradient at zero to be zero.
      1.0
      + 2.0 * math_utils.sqrt_with_zero_gradient_at_zero(epsilon_effective)
  )


# Number of grid points for the bounce-averaged lambda integration.
_LAMBDA_GRID_RESOLUTION: int = 101


def calculate_bounce_averaged_trapped_fraction(
    B: array_typing.Array,
    dl_over_Bp: array_typing.Array,
    flux_surf_avg_B2: array_typing.Array | float | None = None,
    n_lambda: int = _LAMBDA_GRID_RESOLUTION,
) -> array_typing.Array:
  r"""Effective trapped particle fraction of one flux surface.

  Computed from the full bounce-averaged integral:

  .. math::
    f_t = 1 - \frac{3}{4} \langle B^2 \rangle
      \int_0^{1/B_{max}} \frac{\lambda \, d\lambda}{\langle \sqrt{1 -
      \lambda B} \rangle}

  where :math:`\langle . \rangle` is the flux surface average, using the same
  :math:`dl/B_p` weighting as other flux surface averages. This requires the
  full poloidal variation of :math:`|B|` on the flux surface.

  To remove the logarithmic derivative singularity of
  :math:`\langle \sqrt{1 - \lambda B} \rangle` at the trapping boundary
  :math:`\lambda = 1 / B_{max}`, the integral is evaluated using the coordinate
  transformation :math:`y = \sqrt{1 - \lambda B_{max}}`:

  .. math::
    \int_0^{1/B_{max}} \frac{\lambda \, d\lambda}{\langle \sqrt{1 -
      \lambda B} \rangle}
    = \frac{2}{B_{max}^2} \int_0^1 \frac{y (1 - y^2)}{\left\langle
      \sqrt{1 - (1 - y^2) \frac{B}{B_{max}}} \right\rangle} \, dy

  Reference:
  Y. R. Lin‐Liu, R. L. Miller; Upper and lower bounds of the effective trapped
  particle fraction in general tokamak equilibria. Phys. Plasmas 1 May 1995; 2
  (5): 1666–1668. https://doi.org/10.1063/1.871315

  Args:
    B: :math:`|B|` at samples of a poloidal contour around one flux surface
      [:math:`\mathrm{T}`].
    dl_over_Bp: Poloidal line-element weights :math:`dl / B_p` at the same
      contour samples [:math:`\mathrm{m/T}`].
    flux_surf_avg_B2: Optional flux surface average of :math:`B^2` for this flux
      surface [:math:`\mathrm{T}^2`]. If None, computed self-consistently on the
      same contour using ``dl_over_Bp`` weights to preserve the uniform-field
      identity :math:`f_t(B = \mathrm{const}) = 0`.
    n_lambda: Number of points in the transformed pitch-angle grid for the
      bounce integral.

  Returns:
    The effective trapped particle fraction of this flux surface.
  """
  B_max = jnp.max(B)
  norm = jnp.sum(dl_over_Bp)
  if flux_surf_avg_B2 is None:
    flux_surf_avg_B2 = jnp.sum(B**2 * dl_over_Bp) / norm

  y = jnp.linspace(0.0, 1.0, n_lambda)
  one_minus_y2 = 1.0 - y**2
  sqrt_term = jnp.sqrt(
      jnp.clip(
          1.0 - one_minus_y2[:, jnp.newaxis] * (B[jnp.newaxis, :] / B_max),
          0.0,
          None,
      )
  )
  h_y = jnp.sum(sqrt_term * dl_over_Bp[jnp.newaxis, :], axis=1) / norm
  integrand = jnp.where(
      h_y > 0.0,
      y * one_minus_y2 / jnp.maximum(h_y, 1e-12),
      one_minus_y2,
  )
  bounce_integral = (
      (4.0 / (3.0 * B_max**2))
      * jnp.trapezoid(integrand, y)
      / jnp.trapezoid(one_minus_y2, y)
  )
  return 1.0 - 0.75 * flux_surf_avg_B2 * bounce_integral


def extrapolate_exact_trapped_fraction_to_axis(
    trapped_fraction: np.ndarray | array_typing.Array,
    rhon: np.ndarray | array_typing.Array,
    sauter_trapped_fraction: np.ndarray | array_typing.Array | None = None,
) -> np.ndarray:
  """Smoothly regularizes exact trapped fraction by extrapolating to the axis.

  Numerical bounce-averaged integration can fail or produce unreliable
  (unphysical or NaN) values on flux surfaces very close to the magnetic axis
  where poloidal contours are poorly resolved on a discrete 2D grid.

  Physically, f_t(0) = 0 and f_t ∝ √ρ as ρ → 0. This function maps to u = √ρ,
  where f_t(u) is smooth and regular with f_t(0) = 0, and uses shape-preserving
  monotonic cubic Hermite interpolation (PCHIP) from the reliable exact flux
  surfaces down to (u=0, f_t=0). This avoids artificial discontinuities or
  gradient kinks that arise from patching a different analytic model on-axis.

  Args:
    trapped_fraction: 1D array of trapped fraction values across flux surfaces.
      May contain NaN or unphysical values (< 0 or > 1) near the axis.
    rhon: 1D array of normalized radius (0 to 1) corresponding to the flux
      surfaces.
    sauter_trapped_fraction: Optional fallback array to use only if too few
      exact values are valid to construct an interpolant.

  Returns:
    A 1D array with bad values replaced by the smooth monotonic extrapolation
    down to f_t(0) = 0.
  """
  trapped_fraction = np.asarray(trapped_fraction)
  rhon = np.asarray(rhon)
  if sauter_trapped_fraction is not None:
    sauter_trapped_fraction = np.asarray(sauter_trapped_fraction)

  exact_is_unreliable = (
      np.isnan(trapped_fraction)
      | (trapped_fraction < 0.0)
      | (trapped_fraction > 1.0)
  ).copy()
  if len(rhon) > 0 and rhon[0] == 0.0:
    exact_is_unreliable[0] = trapped_fraction[0] != 0.0

  valid_mask = ~exact_is_unreliable
  valid_indices_positive_rho = np.where(valid_mask & (rhon > 0.0))[0]

  if len(valid_indices_positive_rho) < 2:
    if sauter_trapped_fraction is not None:
      return np.where(
          exact_is_unreliable, sauter_trapped_fraction, trapped_fraction
      )
    raise ValueError(
        'Too few valid exact trapped fraction points to extrapolate to axis.'
    )

  # Construct monotonic cubic interpolant in u = sqrt(rhon).
  # Include the physical boundary condition at the magnetic axis: u=0 -> f_t=0.
  u = np.sqrt(np.maximum(rhon, 0.0))
  u_knots = np.concatenate([[0.0], u[valid_indices_positive_rho]])
  f_knots = np.concatenate(
      [[0.0], trapped_fraction[valid_indices_positive_rho]]
  )

  pchip = scipy.interpolate.PchipInterpolator(u_knots, f_knots)
  f_extrapolated = np.clip(pchip(u), 0.0, 1.0)
  result = np.where(exact_is_unreliable, f_extrapolated, trapped_fraction)
  if len(result) > 0:
    result[0] = 0.0
  return result

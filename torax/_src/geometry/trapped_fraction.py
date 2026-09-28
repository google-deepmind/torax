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

import enum
import jax.numpy as jnp
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

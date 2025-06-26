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

"""Prescribed formulas for computing source profiles."""
import jax
from jax import numpy as jnp

from torax._src import math_utils
from torax._src.geometry import geometry

# pylint: disable=invalid-name


def exponential_profile(
    geo: geometry.Geometry,
    *,
    decay_start: float,
    width: float,
    total: float,
) -> jax.Array:
  """Returns an exponential profile on the cell grid.

  The profile is parameterized by (decay_start, width, total) like so:

    profile = C * exp(-(decay_start - r) / width)

  Where C is a calculated prefactor to ensure the volume integral of the profile
  equals `total`.

  Args:
    geo: Geometry constants of torus.
    decay_start: See description above. In normalized radial coordinate.
    width: See description above. In normalized radial coordinate.
    total: See description above.

  Returns:
    Exponential profile on the cell grid.
  """
  r = geo.rho_norm
  S = jnp.exp(-(decay_start - r) / width)
  # calculate constant prefactor
  C = total / math_utils.volume_integration(S, geo)
  return C * S


def gaussian_profile(
    geo: geometry.Geometry,
    *,
    center: float,
    width: float,
    total: float,
) -> jax.Array:
  """Returns a gaussian profile on the cell grid.

  The profile is parameterized by (center, width, total) like so:

    profile = C * exp(-( (r - center)**2 / (2 * width**2) ))

  Where C is a calculated prefactor to ensure the volume integral of the profile
  equals `total`.

  Args:
    geo: Geometry constants of torus.
    center: See description above. In normalized radial coordinate.
    width: See description above. In normalized radial coordinate.
    total: See description above.

  Returns:
    Gaussian profile on the cell grid.
  """
  r = geo.rho_norm
  S = jnp.exp(-((r - center) ** 2) / (2 * width**2))
  # calculate constant prefactor
  C = total / math_utils.volume_integration(S, geo)
  return C * S

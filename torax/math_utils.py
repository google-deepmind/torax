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

"""Math operations.

Math operations that are needed for Torax, but are not specific to plasma
physics or differential equation solvers.
"""
from typing import Optional
from jax import numpy as jnp


def tridiag(
    diag: jnp.ndarray, above: jnp.ndarray, below: jnp.ndarray
) -> jnp.ndarray:
  """Builds a tridiagonal matrix.

  Args:
    diag: The main diagonal.
    above: The +1 diagonal.
    below: The -1 diagonal.

  Returns:
    The tridiagonal matrix.
  """

  return jnp.diag(diag) + jnp.diag(above, 1) + jnp.diag(below, -1)


def cumulative_trapezoid(
    x: jnp.ndarray, y: jnp.ndarray, initial: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
  """Cumulatively integrate y = f(x) using the trapezoid rule.

  Jax equivalent of scipy.integrate.cumulative_trapezoid.
  without as much support for different shapes / options as the scipy version.

  Args:
    x: 1-D array
    y: 1-D array
    initial: Optional array containing a single value. If specified, out[i] =
      trapz(y[:i +1], x[:i + 1]), with out[0] = initial. Usually initial should
      be 0 in this case. If left unspecified, the leftmost output, corresponding
      to summing no terms, is omitted.

  Returns:
    out: 1-D array of same shape, containing the cumulative integration by
      trapezoid rule.
  """

  d = jnp.diff(x)
  out = jnp.cumsum(d * (y[1:] + y[:-1])) / 2.0
  if initial is not None:
    out = jnp.concatenate((jnp.expand_dims(initial, 0), out))
  return out


def gradient(y: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
  """Make effective jnp.gradient function, 2nd order like numpy.

  Needed since jnp.gradient does not currently support nonuniform spacing.

  Args:
    y: array being differentiated with respect to x.
    x: differentiating array.

  Returns:
    result: dy/dx
  """
  assert x.shape == y.shape
  length = len(x)
  result = jnp.zeros(length)

  # 1st order derivatives at array boundaries
  result = result.at[0].set((y[1] - y[0]) / (x[1] - x[0]))
  result = result.at[-1].set((y[-1] - y[-2]) / (x[-1] - x[-2]))

  dx = x[1:] - x[0:-1]
  dx1 = dx[0:-1]
  dx2 = dx[1:]
  a = -(dx2) / (dx1 * (dx1 + dx2))
  b = (dx2 - dx1) / (dx1 * dx2)
  c = dx1 / (dx2 * (dx1 + dx2))

  # 2nd order derivatives at inner elements, allowing for nonuniform dx
  for idx in range(1, length - 1):
    result = result.at[idx].set(
        (
            a[idx - 1] * y[idx - 1]
            + b[idx - 1] * y[idx]
            + c[idx - 1] * y[idx + 1]
        )
    )
  return result

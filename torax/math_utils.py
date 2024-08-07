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
import functools
import jax
from jax import numpy as jnp


def tridiag(
    diag: jax.Array, above: jax.Array, below: jax.Array
) -> jax.Array:
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
    y: jax.Array,
    x: jax.Array | None = None,
    dx: float = 1.0,
    axis: int = -1,
    initial: float | None = None,
) -> jax.Array:
  """Cumulatively integrate y = f(x) using the trapezoid rule.

  JAX equivalent of scipy.integrate.cumulative_trapezoid.

  Args:
    y: array of data to integrate.
    x: optional array of sample points corresponding to the `y` values. If not
      provided, `x` defaults to equally spaced with spacing given by `dx`.
    dx: the spacing between sample points when `x` is None (default: 1.0).
    axis: the axis along which to integrate (default: -1)
    initial: a scalar value to prepend to the result. Either None (default) or
      0.0. If `initial=0`, the result is an array with the same shape as `y`. If
      ``initial=None``, the resulting array has one fewer elements than `y`
      along the `axis` dimension.

  Returns:
    The cumulative definite integral approximated by the trapezoidal rule.
  """

  if x is None:
    dx = jnp.asarray(dx, dtype=y.dtype)
  else:
    if x.ndim == 1:
      if y.shape[axis] != len(x):
        raise ValueError(
            f'The length of x is {len(x)}, but expected {y.shape[axis]}.'
        )
    else:
      if x.shape != y.shape:
        raise ValueError(
            'If x is not 1 dimensional, it must have the same shape as y.'
        )

    if x.ndim == 1:
      dx = jnp.diff(x)
      new_shape = [1] * y.ndim
      new_shape[axis] = len(dx)
      dx = jnp.reshape(dx, new_shape)
    else:
      dx = jnp.diff(x, axis=axis)

  y_sliced = functools.partial(jax.lax.slice_in_dim, y, axis=axis)

  out = jnp.cumsum(dx * (y_sliced(1, None) + y_sliced(0, -1)), axis=axis) / 2.0

  if initial is not None:
    if initial != 0.0:
      raise ValueError(
          '`initial` must be 0 or None. Non-zero values have been deprecated'
          ' since SciPy version 1.12.0.'
      )
    initial_array = jnp.asarray(initial, dtype=out.dtype)
    initial_shape = list(out.shape)
    initial_shape[axis] = 1
    initial_array = jnp.broadcast_to(initial_array, initial_shape)
    out = jnp.concatenate((initial_array, out), axis=axis)
  return out

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

"""Compatibility module mimicking numpy, producing jax outputs for torax.

qlknn implements neural networks using raw numpy and pandas.
Sending JAX tracers through qlknn *almost* works.
Patching qlknn to use this module in the place of numpy fixes some remaining
problems.

Most of this module is intentionally *not* type annotated, to avoiding
introducing a direct dep on pandas, and because most of the types are
effectively just Any as soon as jax is involved regardless.
"""

# Imports used within this file are private to avoid having fields that
# numpy would not have.
# All public imports are unused within this module. They are imported
# to mimic fields of numpy, to be used by qlknn.
from jax import numpy as _jnp
from jax.numpy import dot  # pylint: disable=unused-import
from jax.numpy import tanh  # pylint: disable=unused-import
import numpy as _np


def array(x):
  """Replacement for `array`.

  qlknn calls this function on the values it loads from disk. We want these
  to become persistent numpy arrays, not jax tracers.

  Args:
    x: Any data compatible with numpy

  Returns:
    x as a *numpy* array.
  """
  return _np.array(x)


def atleast_2d(x):
  """Replacement for atleast_2d.

  qlknn calls this on a pandas.Series, which is just an array with name
  annotations. Everything works OK if we discard the names.

  Args:
    x: An array.

  Returns:
    x, but with at least 2 dimensions.
  """
  if hasattr(x, "to_numpy"):
    x = x.to_numpy()
  return _jnp.atleast_2d(x)


def ascontiguousarray(x):
  """Replace this numpy function with a no-op."""
  return x

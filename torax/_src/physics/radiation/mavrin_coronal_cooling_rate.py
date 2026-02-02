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
"""Impurity cooling rate model for coronal equilibrium.

Implements the polynomial fit coefficients from A. A. Mavrin (2018):
Improved fits of coronal radiative cooling rates for high-temperature plasmas,
Radiation Effects and Defects in Solids, 173:5-6, 388-398,
DOI: 10.1080/10420150.2018.1462361
"""

from typing import Final, Mapping

import chex
import immutabledict
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from torax._src import array_typing

# pylint: disable=invalid-name

COEFFS: Final[Mapping[str, array_typing.FloatVector]] = (
    immutabledict.immutabledict({
        'He': np.array([
            [2.5020e-02],
            [-9.3730e-02],
            [1.0156e-01],
            [3.1469e-01],
            [-3.5551e01],
        ]),
        'Li': np.array([
            [3.5190e-02],
            [-1.6070e-01],
            [2.5082e-01],
            [1.9475e-01],
            [-3.5115e01],
        ]),
        'Be': np.array([
            [4.1690e-02],
            [-2.1384e-01],
            [3.8363e-01],
            [3.7270e-02],
            [-3.4765e01],
        ]),
        'C': np.array([
            [-7.2904e00, 4.4470e-02],
            [-1.6637e01, -2.9191e-01],
            [-1.2788e01, 6.8856e-01],
            [-5.0085e00, -3.6687e-01],
            [-3.4738e01, -3.4174e01],
        ]),
        'N': np.array([
            [-6.9621e00, 5.8770e-02, 2.8350e-02],
            [-1.1570e01, -1.7160e-01, -2.2790e-01],
            [-6.0605e00, 7.6272e-01, 7.0047e-01],
            [-2.3614e00, -5.9668e-01, -5.2628e-01],
            [-3.4065e01, -3.3899e01, -3.3913e01],
        ]),
        'O': np.array([
            [0.0000e00, 1.4360e-02],
            [-5.3765e00, -2.0850e-01],
            [-1.7141e01, 7.9655e-01],
            [-1.5635e01, -7.6211e-01],
            [-3.7257e01, -3.3640e01],
        ]),
        'Ne': np.array([
            [1.5648e01, 1.7244e-01, -2.6930e-02],
            [2.8939e01, -3.9544e-01, 4.3960e-02],
            [1.5230e01, 8.6842e-01, 2.9731e-01],
            [1.7309e00, -8.7750e-01, -4.5345e-01],
            [-3.3132e01, -3.3290e01, -3.3410e01],
        ]),
        'Ar': np.array([
            [1.5353e01, 4.9806e00, -8.2260e-02],
            [3.9161e01, -7.6887e00, 1.7480e-01],
            [3.0769e01, 1.5389e00, 6.1339e-01],
            [6.5221e00, 5.4490e-01, -1.6674e00],
            [-3.2155e01, -3.2530e01, -3.1853e01],
        ]),
        'Kr': np.array([
            [-1.3564e01, -5.2704e00, 4.8356e-01],
            [-4.0133e01, -2.5865e00, -2.9674e00],
            [-4.4723e01, 1.9148e00, 6.6831e00],
            [-2.1484e01, -5.0091e-01, -6.3683e00],
            [-3.4512e01, -3.1399e01, -2.9954e01],
        ]),
        'Xe': np.array([
            [2.5615e01, 1.0748e01, 1.0069e01, 1.0858e00],
            [5.9580e01, -1.1628e01, -3.6885e01, -7.5181e00],
            [4.7081e01, 1.2808e00, 4.8614e01, 1.9619e01],
            [1.4351e01, 5.9339e-01, -2.7526e01, -2.2592e01],
            [-2.9303e01, -3.1113e01, -2.5813e01, -2.2138e01],
        ]),
        'W': np.array([
            [-1.0103e-01, 5.1849e01, -3.6759e-01],
            [-1.0311e00, -6.3303e01, 2.6627e00],
            [-9.5126e-01, 2.2824e01, -6.2740e00],
            [3.8304e-01, -2.9208e00, 5.2499e00],
            [-3.0374e01, -3.0238e01, -3.2153e01],
        ]),
    })
)

# These intervals mark the boundaries in temperature used to select which
# column of coefficients to use.
# All temperatures in keV.
TEMPERATURE_INTERVALS: Final[Mapping[str, array_typing.FloatVector]] = (
    immutabledict.immutabledict({
        'C': np.array([0.5]),
        'N': np.array([0.5, 2.0]),
        'O': np.array([0.3]),
        'Ne': np.array([0.7, 5.0]),
        'Ar': np.array([0.6, 3.0]),
        'Kr': np.array([0.447, 2.364]),
        'Xe': np.array([0.5, 2.5, 10.0]),
        'W': np.array([1.5, 4.0]),
    })
)

MIN_TEMPERATURES: Final[Mapping[str, float]] = immutabledict.immutabledict({
    'He': 0.1,
    'Li': 0.1,
    'Be': 0.1,
    'C': 0.1,
    'N': 0.1,
    'O': 0.1,
    'Ne': 0.1,
    'Ar': 0.1,
    'Kr': 0.1,
    'Xe': 0.1,
    'W': 0.1,
})

MAX_TEMPERATURES: Final[Mapping[str, float]] = immutabledict.immutabledict({
    'He': 100.0,
    'Li': 100.0,
    'Be': 100.0,
    'C': 100.0,
    'N': 100.0,
    'O': 100.0,
    'Ne': 100.0,
    'Ar': 100.0,
    'Kr': 100.0,
    'Xe': 100.0,
    'W': 100.0,
})


def evaluate_polynomial_fit(
    T_e: array_typing.FloatVector,
    unused_ne_tau_fraction: array_typing.FloatScalar,
    coeffs: jt.Float[array_typing.Array, '5 _'],
) -> array_typing.FloatVector:
  """Evaluate the polynomial fit for the cooling rate."""
  # Check shape
  T_e = jnp.asarray(T_e)
  if T_e.ndim == 0:
    expected_shape = (5,)
  elif T_e.ndim == 1:
    expected_shape = (5, T_e.shape[0])
  else:
    raise ValueError(f'Unsupported T_e shape: {T_e.shape}')
  chex.assert_shape(coeffs, expected_shape)

  X = jnp.log10(T_e)
  log10_cooling_rate = jnp.polyval(coeffs, X)
  return 10**log10_cooling_rate

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
"""Impurity cooling rate model for non-coronal equilibrium.

Implements the polynomial fit coefficients from A. A. Mavrin (2017):
Radiative Cooling Rates for Low-Z Impurities in Non-coronal Equilibrium State.
J Fusion Energ (2017) 36:161-172
DOI 10.1007/s10894-017-0136-z
"""

from typing import Final, Mapping

import chex
import immutabledict
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from torax._src import array_typing

# Tables 1, 3, 5, 7, 9, 11, 13, 15
# Note: these coefficients are given for T_e in [eV]. Make sure to convert
# to eV before evaluating the polynomial, i.e. multiply by 1e3.
# Disable line-too-long check for the coefficients table for readability.
# pylint: disable=line-too-long
COEFFS: Final[Mapping[str, array_typing.FloatVector]] = (
    immutabledict.immutabledict({
        'He': np.array([
            [-3.9341e01, -2.7185e01, -3.4950e01, -3.1299e01, -3.3203e01],
            [2.2742e01, -3.4465e01, 5.5957e00, -4.4749e00, -2.3306e00],
            [-8.5940e-02, 3.2223e-01, 2.1542e00, 2.9614e-01, -5.3911e-01],
            [-2.5420e01, 5.0933e01, -7.4762e00, 1.5259e00, 7.2592e-01],
            [1.8843e00, 1.0589e-01, -3.7391e00, -6.1433e-01, 9.7550e-02],
            [-3.5681e-01, 1.1632e-01, 1.4444e-01, 3.2651e-01, 2.6917e-01],
            [-3.2771e00, -2.3641e01, 2.4534e00, -1.6652e-01, -6.6110e-02],
            [-4.9766e00, -7.4782e-01, 1.5000e00, 1.5704e-01, 8.9900e-03],
            [1.9730e-02, -7.6200e-03, 2.1307e-01, -8.0601e-04, 2.9240e-02],
            [-7.4260e-02, 2.1030e-02, 7.6590e-02, 5.0330e-02, 5.1180e-02],
        ]),
        'Li': np.array([
            [-3.5752e01, -3.1170e01, -3.6558e01, -3.0560e01, -3.0040e01, -3.4199e01],
            [-1.6780e00, -1.6918e01, 9.4272e00, -2.4680e00, -4.2963e00, -8.5686e-01],
            [9.5500e-03, 1.1481e-01, 3.5299e00, 1.7912e00, 2.7407e-01, -6.3246e-01],
            [-6.1560e00, 2.0492e01, -8.1056e00, -2.8659e-01, 1.1569e00, 2.4968e-01],
            [-1.5027e00, 2.6136e-01, -4.4113e00, -1.9929e00, -4.5453e-01, 9.9930e-02],
            [2.5568e-01, 2.4870e-01, 5.1430e-02, 2.8150e-01, 3.0616e-01, 2.5080e-01],
            [1.1009e01, -7.0035e00, 1.9427e00, 2.3898e-01, -9.1510e-02, -1.7230e-02],
            [2.1169e00, -3.3910e-01, 1.3459e00, 5.0412e-01, 9.7550e-02, 1.4410e-02],
            [-9.6420e-02, -3.5570e-02, 2.3865e-01, 5.8550e-02, 1.6540e-02, 3.7030e-02],
            [1.3460e-02, 4.1910e-02, 8.6850e-02, 6.7410e-02, 5.4690e-02, 5.5670e-02],
        ]),
        'Be': np.array([
            [-3.0242e01, -3.2152e01, -3.0169e01, -3.7201e01, -4.0868e01, -2.8539e01],
            [2.1405e01, 3.1572e00, -8.9830e00, -2.5643e00, 1.4625e01, -5.0020e00],
            [1.0117e-01, 1.4168e-01, 6.3656e-01, -4.0467e00, 3.3373e00, 3.1089e-01],
            [2.7450e01, -1.4617e01, 4.5232e00, 7.1732e00, -8.8128e00, 1.3149e00],
            [8.8367e-01, 1.4646e-01, -1.5126e00, 5.8147e00, -3.1064e00, -4.0022e-01],
            [-6.6110e-02, 1.4683e-01, 4.0756e-01, 4.0114e-01, 2.4343e-01, 3.1788e-01],
            [3.0202e01, 4.3653e00, -3.7497e-01, -2.5926e00, 1.5996e00, -1.0780e-01],
            [1.2175e00, -1.1290e00, 7.2552e-01, -2.0708e00, 6.8069e-01, 7.3280e-02],
            [-1.4883e-01, 3.4914e-01, -2.9810e-02, -1.4775e-01, 6.0120e-02, 1.7320e-02],
            [4.8900e-03, 4.1730e-02, 5.5620e-02, 2.1900e-02, 6.8350e-02, 6.1360e-02],
        ]),
        'C': np.array([
            [-3.4509e01, -4.9228e01, -1.9100e01, -6.7743e01, -2.4016e01, -2.8126e01],
            [6.7599e00, 5.3922e01, -1.5476e01, 4.1606e01, -7.3974e00, -4.1679e00],
            [-1.7140e-02, 8.4584e-01, 4.2962e00, -5.3665e00, 2.9707e00, 4.9937e-01],
            [-4.0337e00, -5.1128e01, 2.1893e00, -1.5734e01, 1.6859e00, 9.0578e-01],
            [1.5517e-01, -8.9366e-01, -6.1658e00, 6.1760e00, -2.1965e00, -5.3687e-01],
            [2.1110e-02, -2.2710e-02, 1.6098e-01, 7.8010e-01, 3.0521e-01, 2.5962e-01],
            [6.5977e-01, 1.4758e01, 1.1021e00, 1.7905e00, -1.1147e-01, -5.8310e-02],
            [-1.7392e-01, 1.6371e-01, 2.1568e00, -1.7320e00, 3.8653e-01, 1.0420e-01],
            [-2.9270e-02, 2.9362e-01, 1.1101e-01, -2.7897e-01, 3.8970e-02, 4.6610e-02],
            [1.7600e-03, 5.5880e-02, 4.2700e-02, 2.3450e-02, 7.8690e-02, 7.3950e-02],
        ]),
        'N': np.array([
            [-3.5312e01, -5.8692e01, -2.0301e01, -7.7571e01, -2.9401e01, -2.7201e01],
            [7.1926e00, 6.8148e01, -8.8594e00, 5.0488e01, -3.8191e-01, -4.4640e00],
            [7.8200e-03, 3.6209e-01, 6.0500e00, -6.5889e00, 3.5270e00, 7.6960e-01],
            [-3.5696e00, -5.4257e01, -2.7129e00, -1.8187e01, -1.0347e00, 9.2450e-01],
            [-1.2800e-02, 1.4835e-01, -7.6700e00, 6.8691e00, -2.4192e00, -6.7720e-01],
            [1.1180e-02, -1.4700e-03, 1.0705e-01, 8.3119e-01, 3.2269e-01, 2.6185e-01],
            [3.5812e-01, 1.3476e01, 1.9691e00, 2.0259e00, 2.2501e-01, -5.6280e-02],
            [-2.5100e-03, -2.9646e-01, 2.3943e00, -1.7572e00, 3.9511e-01, 1.2014e-01],
            [-2.2020e-02, 2.2706e-01, 1.4088e-01, -2.9376e-01, 2.6510e-02, 4.6870e-02],
            [-1.0000e-03, 5.4220e-02, 4.7450e-02, 1.7200e-02, 7.8930e-02, 7.9250e-02],
        ]),
        'O': np.array([
            [-3.6208e01, -2.9057e01, -2.9370e01, -4.4120e-02, -3.7073e01, -2.5037e01],
            [7.5487e00, -1.5228e01, 8.7451e00, -5.4918e01, 7.8826e00, -5.7568e00],
            [2.3340e-02, -3.1460e00, 6.3827e00, -9.5003e00, 3.7999e00, 1.2973e00],
            [-2.1983e00, 2.0826e01, -1.2357e01, 2.8883e01, -3.8006e00, 1.2040e00],
            [-1.0131e-01, 5.9427e00, -7.6451e00, 8.5536e00, -2.2619e00, -9.1955e-01],
            [8.0600e-03, 1.0610e-01, -2.2230e-02, 5.5336e-01, 5.0270e-01, 2.8988e-01],
            [-6.5108e-01, -8.0843e00, 3.4958e00, -4.8731e00, 5.2144e-01, -7.6780e-02],
            [8.4570e-02, -2.6827e00, 2.2661e00, -1.9172e00, 3.0219e-01, 1.4568e-01],
            [-2.1710e-02, 1.0350e-02, 2.5727e-01, -1.5709e-01, -6.6330e-02, 3.9250e-02],
            [-2.1200e-03, 2.6480e-02, 7.7800e-02, 1.6370e-02, 6.1140e-02, 8.3010e-02],
        ]),
        'Ne': np.array([
            [-3.8610e01, -3.6822e01, -6.6901e00, -1.1261e02, -2.6330e02, -1.1174e02],
            [1.2606e01, 4.9706e00, -2.4212e01, 8.5765e01, 2.1673e02, 6.1907e01],
            [1.7866e-01, -1.5334e00, 7.3589e00, -2.1093e00, 1.2973e00, 4.7967e00],
            [-1.0213e01, 1.1973e00, 5.7352e00, -3.0372e01, -6.7799e01, -1.6289e01],
            [-7.7051e-01, 2.7279e00, -7.4602e00, 2.2928e00, -7.3310e-01, -2.5731e00],
            [2.7510e-02, 9.0090e-02, -7.9030e-02, 7.7055e-01, 4.4883e-01, 4.2620e-01],
            [4.3390e00, -1.3992e00, -8.5020e-02, 3.5346e00, 7.0398e00, 1.4263e00],
            [6.4207e-01, -1.1084e00, 1.8679e00, -5.6062e-01, 9.3190e-02, 3.3443e-01],
            [-3.3560e-02, 1.3620e-02, 2.2507e-01, -1.8569e-01, -1.5390e-02, -9.3734e-04],
            [-1.3333e-04, 2.4300e-02, 7.1420e-02, 3.7550e-02, 7.7660e-02, 8.4220e-02],
        ]),
        'Ar': np.array([
            [-3.6586e01, -4.8732e01, -2.3157e01, -6.8134e01, 5.5851e01, -6.2758e01],
            [1.2841e01, 3.8185e01, -8.5132e00, 3.6408e01, -7.8618e01, 2.5163e01],
            [2.3080e-02, -7.0622e-01, 1.5617e00, -7.3868e00, 1.0520e01, -7.4717e-01],
            [-1.2087e01, -2.5859e01, 1.5478e00, -1.0735e01, 2.2871e01, -6.8170e00],
            [-9.8000e-03, 1.2850e00, -1.8880e00, 6.8800e00, -7.7061e00, 6.9486e-01],
            [-2.4600e-03, -6.8710e-02, 2.2830e-01, 3.1142e-01, -1.8530e-01, 4.6946e-01],
            [4.8823e00, 5.4372e00, 2.8279e-01, 8.0440e-01, -2.1616e00, 5.9969e-01],
            [-3.7470e-02, -5.2157e-01, 5.5767e-01, -1.5740e00, 1.4123e00, -1.3487e-01],
            [1.1100e-03, 1.4016e-01, -9.9600e-02, -9.9180e-02, 1.8409e-01, -8.1380e-02],
            [1.1100e-03, 1.9120e-02, -1.5280e-02, 9.4500e-03, 6.7470e-02, 2.5840e-02],
        ]),
    })
)
# pylint: enable=line-too-long

# These intervals mark the boundaries in temperature used to select which
# column of coefficients to use.
# All temperatures are in keV.
TEMPERATURE_INTERVALS: Final[Mapping[str, array_typing.FloatVector]] = (
    immutabledict.immutabledict({
        'He': np.array([3.0e-3, 10.0e-3, 30.0e-3, 100.0e-3]),
        'Li': np.array([7.0e-3, 30.0e-3, 60.0e-3, 100.0e-3, 1000.0e-3]),
        'Be': np.array([0.7e-3, 3.0e-3, 11.0e-3, 45.0e-3, 170.0e-3]),
        'C': np.array([7.0e-3, 20.0e-3, 70.0e-3, 200.0e-3, 700.0e-3]),
        'N': np.array([10.0e-3, 30.0e-3, 100.0e-3, 300.0e-3, 1000.0e-3]),
        'O': np.array([10.0e-3, 30.0e-3, 100.0e-3, 300.0e-3, 1000.0e-3]),
        'Ne': np.array([10.0e-3, 70.0e-3, 300.0e-3, 1000.0e-3, 3000.0e-3]),
        'Ar': np.array([10.0e-3, 50.0e-3, 150.0e-3, 500.0e-3, 1500.0e-3]),
    })
)

MIN_TEMPERATURES: Final[Mapping[str, float]] = immutabledict.immutabledict({
    'He': 1.0e-3,
    'Li': 1.0e-3,
    'Be': 0.2e-3,
    'C': 1.0e-3,
    'N': 1.0e-3,
    'O': 1.0e-3,
    'Ne': 1.0e-3,
    'Ar': 1.0e-3,
})

MAX_TEMPERATURES: Final[Mapping[str, float]] = immutabledict.immutabledict({
    'He': 15.0,
    'Li': 10.0,
    'Be': 10.0,
    'C': 15.0,
    'N': 15.0,
    'O': 15.0,
    'Ne': 15.0,
    'Ar': 10.0,
})

# pylint: disable=invalid-name


def evaluate_polynomial_fit(
    T_e: array_typing.FloatVector,
    ne_tau_fraction: array_typing.FloatScalar,
    coeffs: jt.Float[array_typing.Array, '10 _'],
) -> array_typing.FloatVector:
  """Evaluate the polynomial fit for the cooling rate."""
  # Check shape
  T_e = jnp.asarray(T_e)
  if T_e.ndim == 0:
    expected_shape = (10,)
  elif T_e.ndim == 1:
    expected_shape = (10, T_e.shape[0])
  else:
    raise ValueError(f'Unsupported T_e shape: {T_e.shape}')
  chex.assert_shape(coeffs, expected_shape)

  # Formulas are constructed for T_e in eV
  T_e_eV = T_e * 1e3
  X = jnp.log10(T_e_eV)
  Y = jnp.log10(ne_tau_fraction)

  # 2D polynomial from Mavrin 2017, Eq. 8
  log10_cooling_rate = (
      coeffs[0]
      + coeffs[1] * X
      + coeffs[2] * Y
      + coeffs[3] * X**2
      + coeffs[4] * X * Y
      + coeffs[5] * Y**2
      + coeffs[6] * X**3
      + coeffs[7] * X**2 * Y
      + coeffs[8] * X * Y**2
      + coeffs[9] * Y**3
  )

  return 10**log10_cooling_rate

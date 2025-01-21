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

"""Routines for calculating impurity charge states."""

from typing import Final, Mapping, Sequence
import immutabledict
from jax import numpy as jnp
import numpy as np
from torax import array_typing
from torax import constants
from torax.config import plasma_composition

# Polynomial fit coefficients from A. A. Mavrin (2018):
# Improved fits of coronal radiative cooling rates for high-temperature plasmas,
# Radiation Effects and Defects in Solids, 173:5-6, 388-398,
# DOI: 10.1080/10420150.2018.1462361
_MAVRIN_Z_COEFFS: Final[Mapping[str, array_typing.ArrayFloat]] = (
    immutabledict.immutabledict({
        'C': np.array([  # Carbon
            [5.8588e00, -1.7632e00, -7.3521e00, -1.2217e01, -7.2007e00],
            [6.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
        ]),
        'N': np.array([  # Nitrogen
            [6.9728e00, 1.5668e-01, 1.8861e00, 3.3818e00, 0.0000e00],
            [7.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
        ]),
        'O': np.array([  # Oxygen
            [4.0451e00, -2.2093e01, -3.8664e01, -1.8560e01, 0.0000e00],
            [7.9878e00, 8.0180e-02, -3.7050e-02, -4.6261e-01, -4.3092e00],
            [8.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
        ]),
        'Ne': np.array([  # Neon
            [8.9737e00, -1.3242e01, -5.3631e01, -6.4696e01, -2.5303e01],
            [9.9532e00, 2.1413e-01, -8.0723e-01, 3.6868e00, -7.0678e00],
            [1.0000e01, 0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00],
        ]),
        'Ar': np.array([  # Argon
            [1.3171e01, -2.0781e01, -4.3776e01, -1.1595e01, 6.8717e00],
            [1.5986e01, 1.1413e00, 2.5023e00, 1.8455e00, -4.8830e-02],
            [1.4948e01, 7.9986e00, -8.0048e00, 3.5667e00, -5.9213e-01],
        ]),
        'Kr': np.array([  # Krypton
            [7.7040e01, 3.0638e02, 5.6890e02, 4.6320e02, 1.3630e02],
            [2.4728e01, 1.5186e00, 1.5744e01, 6.8446e01, -1.0279e02],
            [2.5368e01, 2.3443e01, -2.5703e01, 1.3215e01, -2.4682e00],
        ]),
        'Xe': np.array([  # Xenon
            [3.0532e02, 1.3973e03, 2.5189e03, 1.9967e03, 5.8178e02],
            [3.2616e01, 1.6271e01, -4.8384e01, -2.9061e01, 8.6824e01],
            [4.8066e01, -1.7259e02, 6.6739e02, -9.0008e02, 4.0756e02],
            [-5.7527e01, 2.4056e02, -1.9931e02, 7.3261e01, -1.0019e01],
        ]),
        'W': np.array([  # Tungsten
            [2.6703e01, 1.6518e01, 2.1027e01, 3.4582e01, 1.6823e01],
            [3.6902e01, -7.9611e01, 2.5532e02, -1.0577e01, -2.5887e02],
            [6.3795e01, -1.0011e02, 1.5985e02, -8.4207e01, 1.5119e01],
        ]),
    })
)

# Temperature boundaries in keV, separating the rows for the fit coefficients.
_TEMPERATURE_INTERVALS: Final[Mapping[str, array_typing.ArrayFloat]] = (
    immutabledict.immutabledict({
        'C': np.array([0.7]),
        'N': np.array([0.7]),
        'O': np.array([0.3, 1.5]),
        'Ne': np.array([0.5, 2.0]),
        'Ar': np.array([0.6, 3.0]),
        'Kr': np.array([0.447, 4.117]),
        'Xe': np.array([0.3, 1.5, 8.0]),
        'W': np.array([1.5, 4.0]),
    })
)


# pylint: disable=invalid-name
def calculate_average_charge_state_single_species(
    Te: array_typing.ArrayFloat,
    ion_symbol: str,
) -> array_typing.ArrayFloat:
  """Calculates the average charge state of an impurity based on the Marvin 2018 polynomial fit.

  The polynomial fit range is 0.1-100 keV, which is well within the typical
  bounds of core tokamak modelling. For safety, inputs are clipped to avoid
  extrapolation outside this range.

  Args:
    Te: Electron temperature [keV].
    ion_symbol: Species to calculate average charge state for.

  Returns:
    Z: Average charge state [amu].
  """

  if ion_symbol not in constants.ION_SYMBOLS:
    raise ValueError(
        f'Invalid ion symbol: {ion_symbol}. Allowed symbols are :'
        f' {constants.ION_SYMBOLS}'
    )
  # Return the Z value for light ions that are fully ionized for T > 0.1 keV.
  if ion_symbol not in _MAVRIN_Z_COEFFS:
    return jnp.ones_like(Te) * constants.ION_PROPERTIES_DICT[ion_symbol].Z

  # Avoid extrapolating fitted polynomial out of bounds.
  Te_allowed_range = (0.1, 100.0)
  Te = jnp.clip(Te, *Te_allowed_range)

  # Gather coefficients for each temperature
  interval_indices = jnp.searchsorted(_TEMPERATURE_INTERVALS[ion_symbol], Te)
  Zavg_coeffs_in_range = jnp.take(
      _MAVRIN_Z_COEFFS[ion_symbol], interval_indices, axis=0
  ).transpose()

  def _calculate_in_range(X, coeffs):
    """Return Mavrin 2018 Zavg polynomial."""
    A0, A1, A2, A3, A4 = coeffs
    return A0 + A1 * X + A2 * X**2 + A3 * X**3 + A4 * X**4

  X = jnp.log10(Te)
  Zavg = _calculate_in_range(X, Zavg_coeffs_in_range)

  return Zavg


def get_average_charge_state(
    ion_symbols: Sequence[str],
    ion_mixture: plasma_composition.DynamicIonMixture,
    Te: array_typing.ArrayFloat,
) -> array_typing.ArrayFloat:
  """Calculates or prescribes average impurity charge state profile (JAX-compatible).

  Args:
    ion_symbols: Species to calculate average charge state for.
    ion_mixture: DynamicIonMixture object containing impurity information. The
      index of the ion_mixture.fractions array corresponds to the index of the
      ion_symbols array.
    Te: Electron temperature [keV]. Can be any sized array, e.g. on cell grid,
      face grid, or a single scalar.

  Returns:
    avg_Z: Average charge state profile [amu].
      The shape of avg_Z is the same as Te.
  """

  if ion_mixture.Z_override is not None:
    return jnp.ones_like(Te) * ion_mixture.Z_override

  avg_Z = jnp.zeros_like(Te)
  for ion_symbol, fraction in zip(ion_symbols, ion_mixture.fractions):
    avg_Z += fraction * calculate_average_charge_state_single_species(
        Te, ion_symbol
    )

  return avg_Z

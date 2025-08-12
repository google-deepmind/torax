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

import dataclasses
from typing import Final, Mapping, Sequence

import immutabledict
import jax
from jax import numpy as jnp
import numpy as np
from torax._src import array_typing
from torax._src import constants
from torax._src.config import plasma_composition

# pylint: disable=invalid-name

# Polynomial fit coefficients from A. A. Mavrin (2018):
# Improved fits of coronal radiative cooling rates for high-temperature plasmas,
# Radiation Effects and Defects in Solids, 173:5-6, 388-398,
# DOI: 10.1080/10420150.2018.1462361
_MAVRIN_Z_COEFFS: Final[Mapping[str, array_typing.ArrayFloat]] = (
    immutabledict.immutabledict({
        'C': np.array([  # Carbon
            [-7.2007e00, -1.2217e01, -7.3521e00, -1.7632e00, 5.8588e00],
            [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 6.0000e00],
        ]),
        'N': np.array([  # Nitrogen
            [0.0000e00, 3.3818e00, 1.8861e00, 1.5668e-01, 6.9728e00],
            [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 7.0000e00],
        ]),
        'O': np.array([  # Oxygen
            [0.0000e00, -1.8560e01, -3.8664e01, -2.2093e01, 4.0451e00],
            [-4.3092e00, -4.6261e-01, -3.7050e-02, 8.0180e-02, 7.9878e00],
            [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 8.0000e00],
        ]),
        'Ne': np.array([  # Neon
            [-2.5303e01, -6.4696e01, -5.3631e01, -1.3242e01, 8.9737e00],
            [-7.0678e00, 3.6868e00, -8.0723e-01, 2.1413e-01, 9.9532e00],
            [0.0000e00, 0.0000e00, 0.0000e00, 0.0000e00, 1.0000e01],
        ]),
        'Ar': np.array([  # Argon
            [6.8717e00, -1.1595e01, -4.3776e01, -2.0781e01, 1.3171e01],
            [-4.8830e-02, 1.8455e00, 2.5023e00, 1.1413e00, 1.5986e01],
            [-5.9213e-01, 3.5667e00, -8.0048e00, 7.9986e00, 1.4948e01],
        ]),
        'Kr': np.array([  # Krypton
            [1.3630e02, 4.6320e02, 5.6890e02, 3.0638e02, 7.7040e01],
            [-1.0279e02, 6.8446e01, 1.5744e01, 1.5186e00, 2.4728e01],
            [-2.4682e00, 1.3215e01, -2.5703e01, 2.3443e01, 2.5368e01],
        ]),
        'Xe': np.array([  # Xenon
            [5.8178e02, 1.9967e03, 2.5189e03, 1.3973e03, 3.0532e02],
            [8.6824e01, -2.9061e01, -4.8384e01, 1.6271e01, 3.2616e01],
            [4.0756e02, -9.0008e02, 6.6739e02, -1.7259e02, 4.8066e01],
            [-1.0019e01, 7.3261e01, -1.9931e02, 2.4056e02, -5.7527e01],
        ]),
        'W': np.array([  # Tungsten
            [1.6823e01, 3.4582e01, 2.1027e01, 1.6518e01, 2.6703e01],
            [-2.5887e02, -1.0577e01, 2.5532e02, -7.9611e01, 3.6902e01],
            [1.5119e01, -8.4207e01, 1.5985e02, -1.0011e02, 6.3795e01],
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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ChargeStateInfo:
  """Container for average charge state calculations.

  Attributes:
    Z_avg: Average charge of the mixture, weighted by ion fractions. <Z> =
      sum(fraction_i * Z_i).
    Z2_avg: Average squared charge of the mixture, weighted by ion fractions.
      <Z^2> = sum(fraction_i * Z_i^2).
    Z_per_species: Charge state for each individual ion species in the mixture.
      For impurities, this is the outcome of a temperature dependent charge
      state calculation.
    Z_mixture: Effective charge of the mixture, defined as <Z^2> / <Z>. This is
      the charge used in quasineutrality calculations when treating the mixture
      as a single effective species.
  """

  Z_avg: array_typing.ArrayFloat
  Z2_avg: array_typing.ArrayFloat
  Z_per_species: array_typing.ArrayFloat

  @property
  def Z_mixture(self) -> array_typing.ArrayFloat:
    return self.Z2_avg / self.Z_avg


# pylint: disable=invalid-name
def calculate_average_charge_state_single_species(
    T_e: array_typing.ArrayFloat,
    ion_symbol: str,
) -> array_typing.ArrayFloat:
  """Calculates the average charge state of an impurity based on the Marvin 2018 polynomial fit.

  The polynomial fit range is 0.1-100 keV, which is well within the typical
  bounds of core tokamak modelling. For safety, inputs are clipped to avoid
  extrapolation outside this range.

  Args:
    T_e: Electron temperature [keV].
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
    return jnp.ones_like(T_e) * constants.ION_PROPERTIES_DICT[ion_symbol].Z

  # Avoid extrapolating fitted polynomial out of bounds.
  T_e_allowed_range = (0.1, 100.0)
  T_e = jnp.clip(T_e, *T_e_allowed_range)

  # Gather coefficients for each temperature
  interval_indices = jnp.searchsorted(_TEMPERATURE_INTERVALS[ion_symbol], T_e)
  Zavg_coeffs_in_range = jnp.take(
      _MAVRIN_Z_COEFFS[ion_symbol], interval_indices, axis=0
  ).transpose()

  # Calculate Zavg from the polynomial fit.
  X = jnp.log10(T_e)
  Zavg = jnp.polyval(Zavg_coeffs_in_range, X)

  return Zavg


def get_average_charge_state(
    ion_symbols: Sequence[str],
    ion_mixture: plasma_composition.DynamicIonMixture,
    T_e: array_typing.ArrayFloat,
) -> ChargeStateInfo:
  """Calculates or prescribes average impurity charge state profile (JAX-compatible).

  Equations for quasineutrality and Zeff are the following:

  sum(n_i * Z_i) + sum(n_impurity * Z_impurity) = n_e
  sum(n_i/n_e * Z_i**2) + sum(n_impurity/n_e * Z_impurity**2) = Z_eff

  We now define effective main ion and impurity charge states and densities,
  and constrain that Zeff is the same when using a single effective main ion and
  impurity instead of the full set of ions:

  sum(n_i * Z_i) + sum(n_impurity * Z_impurity) =
    n_i_eff * Z_i_eff + n_impurity_eff * Z_impurity_eff

  sum(n_i * Z_i**2) + sum(n_impurity * Z_impurity**2) =
    n_i_eff * Z_i_eff**2 + n_impurity_eff * Z_impurity_eff**2

  We also are free to constrain that:
  sum(n_i * Z_i) = n_i_eff * Z_i_eff and
  sum(n_i * Z_i ** 2) = n_i_eff * Z_i_eff**2
  individually, for i in {main, impurity}.

  Taking the ratio of the two equations, we then get:
  Z_i_eff = sum(n_i * Z_i **2) / (sum(n_i * Z_i))
    = sum(fraction_i * Z_i ** 2) / sum(fraction_i * Z_i) = <Z^2> / <Z>

  Args:
    ion_symbols: Species to calculate average charge state for.
    ion_mixture: DynamicIonMixture object containing impurity information. The
      index of the ion_mixture.fractions array corresponds to the index of the
      ion_symbols array.
    T_e: Electron temperature [keV]. Can be any sized array, e.g. on cell grid,
      face grid, or a single scalar.

  Returns:
    AverageChargeState: dataclass with average charge state info.
  """

  if ion_mixture.Z_override is not None:
    override_val = jnp.ones_like(T_e) * ion_mixture.Z_override
    return ChargeStateInfo(
        Z_avg=override_val,
        Z2_avg=override_val**2,
        Z_per_species=jnp.stack([override_val for _ in ion_symbols]),
    )

  Z_per_species = jnp.stack([
      calculate_average_charge_state_single_species(T_e, ion_symbol)
      for ion_symbol in ion_symbols
  ])

  # ion_mixture.fractions has shape (n_species,).
  # Z_per_species has shape (n_species,) if T_e is a scalar, or
  # (n_species, n_grid) if T_e is an array.
  # We need to broadcast fractions for element-wise multiplication.
  # Reshape fractions to be broadcastable with Z_per_species.
  fractions_reshaped = jnp.reshape(
      ion_mixture.fractions,
      ion_mixture.fractions.shape + (1,) * (Z_per_species.ndim - 1),
  )

  Z_avg = jnp.sum(fractions_reshaped * Z_per_species, axis=0)
  Z2_avg = jnp.sum(fractions_reshaped * Z_per_species**2, axis=0)

  return ChargeStateInfo(
      Z_avg=Z_avg,
      Z2_avg=Z2_avg,
      Z_per_species=Z_per_species,
  )

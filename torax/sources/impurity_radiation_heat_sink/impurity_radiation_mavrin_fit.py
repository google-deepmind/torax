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

"""Routines for calculating impurity radiation based on a polynomial fit."""

import dataclasses
from typing import Sequence
import chex
import jax
import jax.numpy as jnp
import numpy as np
from torax import array_typing
from torax import constants
from torax import state
from torax.config import plasma_composition
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source_models as source_models_lib

MODEL_FUNCTION_NAME = 'impurity_radiation_mavrin_fit'

# Polynomial fit coefficients from A. A. Mavrin (2018):
# Improved fits of coronal radiative cooling rates for high-temperature plasmas,
# Radiation Effects and Defects in Solids, 173:5-6, 388-398,
# DOI: 10.1080/10420150.2018.1462361
_MAVRIN_L_COEFFS = {
    'He3': np.array([
        [-3.5551e01, 3.1469e-01, 1.0156e-01, -9.3730e-02, 2.5020e-02],
    ]),
    'He4': np.array([
        [-3.5551e01, 3.1469e-01, 1.0156e-01, -9.3730e-02, 2.5020e-02],
    ]),
    'Li': np.array([
        [-3.5115e01, 1.9475e-01, 2.5082e-01, -1.6070e-01, 3.5190e-02],
    ]),
    'Be': np.array([
        [-3.4765e01, 3.7270e-02, 3.8363e-01, -2.1384e-01, 4.1690e-02],
    ]),
    'C': np.array([
        [-3.4738e01, -5.0085e00, -1.2788e01, -1.6637e01, -7.2904e00],
        [-3.4174e01, -3.6687e-01, 6.8856e-01, -2.9191e-01, 4.4470e-02],
    ]),
    'N': np.array([
        [-3.4065e01, -2.3614e00, -6.0605e00, -1.1570e01, -6.9621e00],
        [-3.3899e01, -5.9668e-01, 7.6272e-01, -1.7160e-01, 5.8770e-02],
        [-3.3913e01, -5.2628e-01, 7.0047e-01, -2.2790e-01, 2.8350e-02],
    ]),
    'O': np.array([
        [-3.7257e01, -1.5635e01, -1.7141e01, -5.3765e00, 0.0000e00],
        [-3.3640e01, -7.6211e-01, 7.9655e-01, -2.0850e-01, 1.4360e-02],
    ]),
    'Ne': np.array([
        [-3.3132e01, 1.7309e00, 1.5230e01, 2.8939e01, 1.5648e01],
        [-3.3290e01, -8.7750e-01, 8.6842e-01, -3.9544e-01, 1.7244e-01],
        [-3.3410e01, -4.5345e-01, 2.9731e-01, 4.3960e-02, -2.6930e-02],
    ]),
    'Ar': np.array([
        [-3.2155e01, 6.5221e00, 3.0769e01, 3.9161e01, 1.5353e01],
        [-3.2530e01, 5.4490e-01, 1.5389e00, -7.6887e00, 4.9806e00],
        [-3.1853e01, -1.6674e00, 6.1339e-01, 1.7480e-01, -8.2260e-02],
    ]),
    'Kr': np.array([
        [-3.4512e01, -2.1484e01, -4.4723e01, -4.0133e01, -1.3564e01],
        [-3.1399e01, -5.0091e-01, 1.9148e00, -2.5865e00, -5.2704e00],
        [-2.9954e01, -6.3683e00, 6.6831e00, -2.9674e00, 4.8356e-01],
    ]),
    'Xe': np.array([
        [-2.9303e01, 1.4351e01, 4.7081e01, 5.9580e01, 2.5615e01],
        [-3.1113e01, 5.9339e-01, 1.2808e00, -1.1628e01, 1.0748e01],
        [-2.5813e01, -2.7526e01, 4.8614e01, -3.6885e01, 1.0069e01],
        [-2.2138e01, -2.2592e01, 1.9619e01, -7.5181e00, 1.0858e00],
    ]),
    'W': np.array([
        [-3.0374e01, 3.8304e-01, -9.5126e-01, -1.0311e00, -1.0103e-01],
        [-3.0238e01, -2.9208e00, 2.2824e01, -6.3303e01, 5.1849e01],
        [-3.2153e01, 5.2499e00, -6.2740e00, 2.6627e00, -3.6759e-01],
    ]),
}

# Temperature boundaries in keV, separating the rows for the fit coefficients.
_TEMPERATURE_INTERVALS = {
    'C': np.array([0.5]),
    'N': np.array([0.5, 2.0]),
    'O': np.array([0.3]),
    'Ne': np.array([0.7, 5.0]),
    'Ar': np.array([0.6, 3.0]),
    'Kr': np.array([0.447, 2.364]),
    'Xe': np.array([0.5, 2.5, 10.0]),
    'W': np.array([1.5, 4.0]),
}


# pylint: disable=invalid-name
def _calculate_impurity_radiation_single_species(
    Te: array_typing.ArrayFloat,
    ion_symbol: str,
) -> array_typing.ArrayFloat:
  """Calculates the line radiation for single impurity species.

  Polynomial fit range is 0.1-100 keV, which is well within the typical
  bounds of core tokamak modelling. For safety, inputs are clipped to avoid
  extrapolation outside this range.

  Args:
    Te: Electron temperature [keV].
    ion_symbol: Species to calculate line radiation for.

  Returns:
    LZ: Radiative cooling rate in units of Wm^3.
  """

  if ion_symbol not in constants.ION_SYMBOLS:
    raise ValueError(
        f'Invalid ion symbol: {ion_symbol}. Allowed symbols are :'
        f' {constants.ION_SYMBOLS}'
    )

  # Avoid extrapolating fitted polynomial out of bounds.
  Te_allowed_range = (0.1, 100.0)
  Te = jnp.clip(Te, *Te_allowed_range)

  # Gather coefficients for each temperature
  if ion_symbol in {'He3', 'He4', 'Be', 'Li'}:
    interval_indices = 0
  else:
    interval_indices = jnp.searchsorted(_TEMPERATURE_INTERVALS[ion_symbol], Te)

  L_coeffs_in_range = jnp.take(
      _MAVRIN_L_COEFFS[ion_symbol], interval_indices, axis=0
  ).transpose()

  def _calculate_in_range(X, coeffs):
    """Return Mavrin 2018 LZ polynomial."""
    A0, A1, A2, A3, A4 = coeffs
    log_LZ = A0 + A1 * X + A2 * X**2 + A3 * X**3 + A4 * X**4
    return 10**log_LZ

  X = jnp.log10(Te)
  return _calculate_in_range(X, L_coeffs_in_range)


def calculate_total_impurity_radiation(
    ion_symbols: Sequence[str],
    ion_mixture: plasma_composition.DynamicIonMixture,
    Te: array_typing.ArrayFloat,
) -> array_typing.ArrayFloat:
  """Calculates impurity line radiation profile (JAX-compatible).

  Args:
    ion_symbols: Ion symbols of the impurity species.
    ion_mixture: DynamicIonMixture object containing impurity information.
    Te: Electron temperature [keV]. Can be any sized array, e.g. on cell grid,
      face grid, or a single scalar.

  Returns:
    effective_LZ: Total effective radiative cooling rate in units of Wm^3,
      summed over all species in the mixture. The shape of LZ is the same as Te.
  """

  effective_LZ = jnp.zeros_like(Te)
  for ion_symbol, fraction in zip(ion_symbols, ion_mixture.fractions):
    effective_LZ += fraction * _calculate_impurity_radiation_single_species(
        Te, ion_symbol
    )
  return effective_LZ


def impurity_radiation_mavrin_fit(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    source_name: str,
    core_profiles: state.CoreProfiles,
    unused_source_models: source_models_lib.SourceModels | None = None,
) -> jax.Array:
  """Model function for impurity radiation heat sink."""
  del (geo, unused_source_models)
  effective_LZ = calculate_total_impurity_radiation(
      ion_symbols=static_runtime_params_slice.impurity_names,
      ion_mixture=dynamic_runtime_params_slice.plasma_composition.impurity,
      Te=core_profiles.temp_el.value,
  )
  dynamic_source_runtime_params = dynamic_runtime_params_slice.sources[
      source_name
  ]
  assert isinstance(dynamic_source_runtime_params, DynamicRuntimeParams)
  radiation_profile = (
      effective_LZ
      * core_profiles.ne.value
      * core_profiles.nimp.value
      * dynamic_source_runtime_params.radiation_multiplier
      * dynamic_runtime_params_slice.numerics.nref**2
  )

  return -radiation_profile


@dataclasses.dataclass(kw_only=True)
class RuntimeParams(runtime_params_lib.RuntimeParams):
  radiation_multiplier: float = 1.0
  mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED

  def make_provider(
      self,
      torax_mesh: geometry.Grid1D | None = None,
  ) -> 'RuntimeParamsProvider':
    return RuntimeParamsProvider(**self.get_provider_kwargs(torax_mesh))


@chex.dataclass
class RuntimeParamsProvider(runtime_params_lib.RuntimeParamsProvider):
  """Provides runtime parameters for a given time and geometry."""

  runtime_params_config: RuntimeParams

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> 'DynamicRuntimeParams':
    return DynamicRuntimeParams(**self.get_dynamic_params_kwargs(t))


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  radiation_multiplier: array_typing.ScalarFloat

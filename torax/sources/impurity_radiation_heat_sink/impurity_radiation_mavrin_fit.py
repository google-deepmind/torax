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

import functools
from typing import Final, Literal, Mapping, Sequence

import chex
import immutabledict
import jax.numpy as jnp
import numpy as np
from torax import array_typing
from torax import constants
from torax import jax_utils
from torax import state
from torax.config import plasma_composition
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.sources import base
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source as source_lib
from torax.sources import source_profiles
from torax.sources.impurity_radiation_heat_sink import impurity_radiation_heat_sink


MODEL_FUNCTION_NAME = 'impurity_radiation_mavrin_fit'

# Polynomial fit coefficients from A. A. Mavrin (2018):
# Improved fits of coronal radiative cooling rates for high-temperature plasmas,
# Radiation Effects and Defects in Solids, 173:5-6, 388-398,
# DOI: 10.1080/10420150.2018.1462361
_MAVRIN_L_COEFFS: Final[Mapping[str, array_typing.ArrayFloat]] = (
    immutabledict.immutabledict({
        'He3': np.array([
            [2.5020e-02, -9.3730e-02, 1.0156e-01, 3.1469e-01, -3.5551e01],
        ]),
        'He4': np.array([
            [2.5020e-02, -9.3730e-02, 1.0156e-01, 3.1469e-01, -3.5551e01],
        ]),
        'Li': np.array([
            [3.5190e-02, -1.6070e-01, 2.5082e-01, 1.9475e-01, -3.5115e01],
        ]),
        'Be': np.array([
            [4.1690e-02, -2.1384e-01, 3.8363e-01, 3.7270e-02, -3.4765e01],
        ]),
        'C': np.array([
            [-7.2904e00, -1.6637e01, -1.2788e01, -5.0085e00, -3.4738e01],
            [4.4470e-02, -2.9191e-01, 6.8856e-01, -3.6687e-01, -3.4174e01],
        ]),
        'N': np.array([
            [-6.9621e00, -1.1570e01, -6.0605e00, -2.3614e00, -3.4065e01],
            [5.8770e-02, -1.7160e-01, 7.6272e-01, -5.9668e-01, -3.3899e01],
            [2.8350e-02, -2.2790e-01, 7.0047e-01, -5.2628e-01, -3.3913e01],
        ]),
        'O': np.array([
            [0.0000e00, -5.3765e00, -1.7141e01, -1.5635e01, -3.7257e01],
            [1.4360e-02, -2.0850e-01, 7.9655e-01, -7.6211e-01, -3.3640e01],
        ]),
        'Ne': np.array([
            [1.5648e01, 2.8939e01, 1.5230e01, 1.7309e00, -3.3132e01],
            [1.7244e-01, -3.9544e-01, 8.6842e-01, -8.7750e-01, -3.3290e01],
            [-2.6930e-02, 4.3960e-02, 2.9731e-01, -4.5345e-01, -3.3410e01],
        ]),
        'Ar': np.array([
            [1.5353e01, 3.9161e01, 3.0769e01, 6.5221e00, -3.2155e01],
            [4.9806e00, -7.6887e00, 1.5389e00, 5.4490e-01, -3.2530e01],
            [-8.2260e-02, 1.7480e-01, 6.1339e-01, -1.6674e00, -3.1853e01],
        ]),
        'Kr': np.array([
            [-1.3564e01, -4.0133e01, -4.4723e01, -2.1484e01, -3.4512e01],
            [-5.2704e00, -2.5865e00, 1.9148e00, -5.0091e-01, -3.1399e01],
            [4.8356e-01, -2.9674e00, 6.6831e00, -6.3683e00, -2.9954e01],
        ]),
        'Xe': np.array([
            [2.5615e01, 5.9580e01, 4.7081e01, 1.4351e01, -2.9303e01],
            [1.0748e01, -1.1628e01, 1.2808e00, 5.9339e-01, -3.1113e01],
            [1.0069e01, -3.6885e01, 4.8614e01, -2.7526e01, -2.5813e01],
            [1.0858e00, -7.5181e00, 1.9619e01, -2.2592e01, -2.2138e01],
        ]),
        'W': np.array([
            [-1.0103e-01, -1.0311e00, -9.5126e-01, 3.8304e-01, -3.0374e01],
            [5.1849e01, -6.3303e01, 2.2824e01, -2.9208e00, -3.0238e01],
            [-3.6759e-01, 2.6627e00, -6.2740e00, 5.2499e00, -3.2153e01],
        ]),
    })
)

# Temperature boundaries in keV, separating the rows for the fit coefficients.
_TEMPERATURE_INTERVALS: Final[Mapping[str, array_typing.ArrayFloat]] = (
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
  Te = jnp.clip(Te, 0.1, 100.0)

  # Gather coefficients for each temperature
  if ion_symbol in {'He3', 'He4', 'Be', 'Li'}:
    interval_indices = 0
  else:
    interval_indices = jnp.searchsorted(_TEMPERATURE_INTERVALS[ion_symbol], Te)

  L_coeffs_in_range = jnp.take(
      _MAVRIN_L_COEFFS[ion_symbol], interval_indices, axis=0
  ).transpose()

  X = jnp.log10(Te)
  log10_LZ = jnp.polyval(L_coeffs_in_range, X)
  return 10**log10_LZ


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'ion_symbols',
    ],
)
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
    unused_geo: geometry.Geometry,
    source_name: str,
    core_profiles: state.CoreProfiles,
    unused_calculated_source_profiles: source_profiles.SourceProfiles | None,
) -> tuple[chex.Array, ...]:
  """Model function for impurity radiation heat sink."""
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

  # The impurity radiation heat sink is a negative source, so we return a
  # negative profile.
  return (-radiation_profile,)


class ImpurityRadiationHeatSinkMavrinFitConfig(base.SourceModelBase):
  """Configuration for the ImpurityRadiationHeatSink.

  Attributes:
    radiation_multiplier: Multiplier for the impurity radiation profile.
  """

  source_name: Literal['impurity_radiation_heat_sink'] = (
      'impurity_radiation_heat_sink'
  )
  model_function_name: Literal['impurity_radiation_mavrin_fit'] = (
      'impurity_radiation_mavrin_fit'
  )
  radiation_multiplier: float = 1.0
  mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED

  @property
  def model_func(self) -> source_lib.SourceProfileFunction:
    return impurity_radiation_mavrin_fit

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> 'DynamicRuntimeParams':
    return DynamicRuntimeParams(
        prescribed_values=self.prescribed_values.get_value(t),
        radiation_multiplier=self.radiation_multiplier,
    )

  def build_source(
      self,
  ) -> impurity_radiation_heat_sink.ImpurityRadiationHeatSink:
    return impurity_radiation_heat_sink.ImpurityRadiationHeatSink(
        model_func=self.model_func
    )


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  radiation_multiplier: array_typing.ScalarFloat

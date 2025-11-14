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
import functools
from typing import Annotated, Final, Literal, Mapping, Sequence
import chex
import immutabledict
import jax
import jax.numpy as jnp
import numpy as np
from torax._src import array_typing
from torax._src import constants
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.physics import charge_states
from torax._src.sources import base
from torax._src.sources import runtime_params as sources_runtime_params_lib
from torax._src.sources import source as source_lib
from torax._src.sources import source_profiles
from torax._src.sources.impurity_radiation_heat_sink import impurity_radiation_heat_sink
from torax._src.torax_pydantic import torax_pydantic

# Default value for the model function to be used for the impurity radiation
# source. This is also used as an identifier for the model function in
# the source config for Pydantic to "discriminate" against.
DEFAULT_MODEL_FUNCTION_NAME: str = 'mavrin_fit'

# Polynomial fit coefficients from A. A. Mavrin (2018):
# Improved fits of coronal radiative cooling rates for high-temperature plasmas,
# Radiation Effects and Defects in Solids, 173:5-6, 388-398,
# DOI: 10.1080/10420150.2018.1462361
_MAVRIN_L_COEFFS: Final[Mapping[str, array_typing.FloatVector]] = (
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
_TEMPERATURE_INTERVALS: Final[Mapping[str, array_typing.FloatVector]] = (
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
def calculate_impurity_radiation_single_species(
    T_e: array_typing.FloatVector,
    ion_symbol: str,
) -> array_typing.FloatVector:
  """Calculates the line radiation for single impurity species.

  Polynomial fit range is 0.1-100 keV, which is well within the typical
  bounds of core tokamak modelling. For safety, inputs are clipped to avoid
  extrapolation outside this range.

  Args:
    T_e: Electron temperature [keV].
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
  T_e = jnp.clip(T_e, 0.1, 100.0)

  # Gather coefficients for each temperature
  if ion_symbol in {'He3', 'He4', 'Be', 'Li'}:
    interval_indices = 0
  else:
    interval_indices = jnp.searchsorted(_TEMPERATURE_INTERVALS[ion_symbol], T_e)

  L_coeffs_in_range = jnp.take(
      _MAVRIN_L_COEFFS[ion_symbol], interval_indices, axis=0
  ).transpose()

  X = jnp.log10(T_e)
  log10_LZ = jnp.polyval(L_coeffs_in_range, X)
  return 10**log10_LZ


@functools.partial(
    jax.jit,
    static_argnames=[
        'ion_symbols',
    ],
)
def calculate_total_impurity_radiation(
    ion_symbols: Sequence[str],
    impurity_fractions: array_typing.FloatVector,
    T_e: array_typing.FloatVector,
) -> array_typing.FloatVector:
  """Calculates impurity line radiation profile (JAX-compatible).

  Args:
    ion_symbols: Ion symbols of the impurity species.
    impurity_fractions: Impurity fractions corresponding to the ion symbols.
      Input shape is (n_species, n_grid)
    T_e: Electron temperature [keV]. Can be any sized array, e.g. on cell grid,
      face grid, or a single scalar.

  Returns:
    effective_LZ: Total effective radiative cooling rate in units of Wm^3,
      summed over all species in the mixture. The shape of LZ is the same as
      T_e.
  """

  effective_LZ = jnp.zeros_like(T_e)
  for i, ion_symbol in enumerate(ion_symbols):
    fraction = impurity_fractions[i]
    effective_LZ += fraction * calculate_impurity_radiation_single_species(
        T_e, ion_symbol
    )
  return effective_LZ


def impurity_radiation_mavrin_fit(
    runtime_params: runtime_params_lib.RuntimeParams,
    unused_geo: geometry.Geometry,
    source_name: str,
    core_profiles: state.CoreProfiles,
    unused_calculated_source_profiles: source_profiles.SourceProfiles | None,
    unused_conductivity: conductivity_base.Conductivity | None,
) -> tuple[array_typing.FloatVectorCell, ...]:
  """Model function for impurity radiation heat sink."""

  # Reconstruct array from mapping for DynamicIonMixture and effective LZ
  # calculations.
  ion_symbols = runtime_params.plasma_composition.impurity_names
  impurity_fractions_arr = jnp.stack(
      [core_profiles.impurity_fractions[symbol] for symbol in ion_symbols]
  )
  impurity_fractions = {
      symbol: core_profiles.impurity_fractions[symbol] for symbol in ion_symbols
  }
  # Calculate the total effective cooling rate coming from all impurity species.
  effective_LZ = calculate_total_impurity_radiation(
      ion_symbols=runtime_params.plasma_composition.impurity_names,
      impurity_fractions=impurity_fractions_arr,
      T_e=core_profiles.T_e.value,
  )
  # The impurity density must be scaled to account for the true total impurity
  # density. This is because in an IonMixture, the impurity density is an
  # effective density as follows:
  # n_imp_true * sum(fraction_imp * Z_imp) = n_imp_eff * Z_imp_eff
  # where core_profiles.Z_impurity is the effective impurity charge for the
  # IonMixture and core_profiles.n_impurity is effective total impurity density.
  # However, the input fractions correspond to fractions of the true total
  # impurity density which must be scaled from the effective density as follows:
  # n_imp_true = n_imp_eff * Z_imp_eff / <Z>
  # It is important that the calculated radiation corresponds to the true total
  # impurity density, not the effective one.
  charge_state_info = charge_states.get_average_charge_state(
      T_e=core_profiles.T_e.value,
      fractions=impurity_fractions,
      Z_override=runtime_params.plasma_composition.impurity.Z_override,
  )
  Z_avg = charge_state_info.Z_avg
  impurity_density_scaling = core_profiles.Z_impurity / Z_avg

  source_params = runtime_params.sources[source_name]
  assert isinstance(source_params, RuntimeParams)
  radiation_profile = (
      effective_LZ
      * core_profiles.n_e.value
      * core_profiles.n_impurity.value
      * impurity_density_scaling
      * source_params.radiation_multiplier
  )

  # The impurity radiation heat sink is a negative source, so we return a
  # negative profile.
  return (-radiation_profile,)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(sources_runtime_params_lib.RuntimeParams):
  radiation_multiplier: array_typing.FloatScalar


class ImpurityRadiationHeatSinkMavrinFitConfig(base.SourceModelBase):
  """Configuration for the ImpurityRadiationHeatSink.

  Attributes:
    radiation_multiplier: Multiplier for the impurity radiation profile.
  """

  model_name: Annotated[Literal['mavrin_fit'], torax_pydantic.JAX_STATIC] = (
      'mavrin_fit'
  )
  radiation_multiplier: float = 1.0
  mode: Annotated[
      sources_runtime_params_lib.Mode, torax_pydantic.JAX_STATIC
  ] = sources_runtime_params_lib.Mode.MODEL_BASED

  @property
  def model_func(self) -> source_lib.SourceProfileFunction:
    return impurity_radiation_mavrin_fit

  def build_runtime_params(
      self,
      t: chex.Numeric,
  ) -> RuntimeParams:
    return RuntimeParams(
        prescribed_values=tuple(
            [v.get_value(t) for v in self.prescribed_values]
        ),
        mode=self.mode,
        is_explicit=self.is_explicit,
        radiation_multiplier=self.radiation_multiplier,
    )

  def build_source(
      self,
  ) -> impurity_radiation_heat_sink.ImpurityRadiationHeatSink:
    return impurity_radiation_heat_sink.ImpurityRadiationHeatSink(
        model_func=self.model_func
    )

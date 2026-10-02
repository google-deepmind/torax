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
from typing import Annotated, Literal, Sequence
import chex
import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.physics.radiation import radiation
from torax._src.sources import base
from torax._src.sources import runtime_params as sources_runtime_params_lib
from torax._src.sources import source as source_lib
from torax._src.sources import source_profiles
from torax._src.sources.impurity_radiation_heat_sink import impurity_radiation_heat_sink
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name

DEFAULT_MODEL_NAME: str = 'mavrin_fit'


@jax.jit(
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
    effective_LZ += fraction * radiation.calculate_cooling_rate(T_e, ion_symbol)
  return effective_LZ


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class MavrinImpurityRadiationHeatSink(
    impurity_radiation_heat_sink.ImpurityRadiationHeatSink
):
  """Impurity radiation heat sink based on Mavrin polynomial fit."""

  def _get_model_value(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      calculated_source_profiles: source_profiles.SourceProfiles | None = None,
      conductivity: conductivity_base.Conductivity | None = None,
  ) -> tuple[source_lib.SourceProfileElement, ...]:
    del geo, calculated_source_profiles, conductivity
    # Reconstruct array from mapping for DynamicIonMixture and effective LZ
    # calculations.
    ion_symbols = runtime_params.plasma_composition.impurity_names
    impurity_fractions_arr = jnp.stack(
        [core_profiles.impurity_fractions[symbol] for symbol in ion_symbols]
    )
    # Calculate the total effective cooling rate coming from all impurity
    # species.
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
    # IonMixture and core_profiles.n_impurity is effective total impurity
    # density.
    # However, the input fractions correspond to fractions of the true total
    # impurity density which must be scaled from the effective density as
    # follows:
    # n_imp_true = n_imp_eff * Z_imp_eff / <Z>
    # It is important that the calculated radiation corresponds to the true
    # total impurity density, not the effective one.
    impurity_density_scaling = core_profiles.impurity_density_scaling

    source_params = runtime_params.sources[self.SOURCE_ID]
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
  ) -> MavrinImpurityRadiationHeatSink:
    return MavrinImpurityRadiationHeatSink()

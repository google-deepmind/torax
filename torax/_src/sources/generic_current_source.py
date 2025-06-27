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

"""External current source profile."""
import dataclasses
from typing import ClassVar, Literal

import chex
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import math_utils
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.sources import base as source_base
from torax._src.sources import runtime_params as runtime_params_lib
from torax._src.sources import source
from torax._src.sources import source_profiles
from torax._src.torax_pydantic import torax_pydantic

# Default value for the model function to be used for the generic current
# source. This is also used as an identifier for the model function in
# the default source config for Pydantic to "discriminate" against.
DEFAULT_MODEL_FUNCTION_NAME: str = 'gaussian'


# pylint: disable=invalid-name
@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  """Dynamic runtime parameters for the external current source."""

  I_generic: array_typing.ScalarFloat
  fraction_of_total_current: array_typing.ScalarFloat
  gaussian_width: array_typing.ScalarFloat
  gaussian_location: array_typing.ScalarFloat
  use_absolute_current: bool


def calculate_generic_current(
    unused_static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    source_name: str,
    unused_state: state.CoreProfiles,
    unused_calculated_source_profiles: source_profiles.SourceProfiles | None,
    unused_conductivity: conductivity_base.Conductivity | None,
) -> tuple[chex.Array, ...]:
  """Calculates the external current density profiles on the cell grid."""
  dynamic_source_runtime_params = dynamic_runtime_params_slice.sources[
      source_name
  ]
  # pytype: enable=name-error
  assert isinstance(dynamic_source_runtime_params, DynamicRuntimeParams)
  I_generic = _calculate_I_generic(
      dynamic_runtime_params_slice,
      dynamic_source_runtime_params,
  )
  # form of external current on cell grid
  generic_current_form = jnp.exp(
      -((geo.rho_norm - dynamic_source_runtime_params.gaussian_location) ** 2)
      / (2 * dynamic_source_runtime_params.gaussian_width**2)
  )

  Cext = I_generic / math_utils.area_integration(generic_current_form, geo)
  generic_current_profile = Cext * generic_current_form
  return (generic_current_profile,)


def _calculate_I_generic(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    dynamic_source_runtime_params: DynamicRuntimeParams,
) -> chex.Numeric:
  """Calculates the total value of external current."""
  return jnp.where(
      dynamic_source_runtime_params.use_absolute_current,
      dynamic_source_runtime_params.I_generic,
      (
          dynamic_runtime_params_slice.profile_conditions.Ip
          * dynamic_source_runtime_params.fraction_of_total_current
      ),
  )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class GenericCurrentSource(source.Source):
  """A generic current density source profile."""

  SOURCE_NAME: ClassVar[str] = 'generic_current'
  model_func: source.SourceProfileFunction = calculate_generic_current

  @property
  def source_name(self) -> str:
    return self.SOURCE_NAME

  @property
  def affected_core_profiles(self) -> tuple[source.AffectedCoreProfile, ...]:
    return (source.AffectedCoreProfile.PSI,)


class GenericCurrentSourceConfig(source_base.SourceModelBase):
  """Configuration for the GenericCurrentSource.

  Attributes:
    I_generic: total "external" current in A. Used if
      use_absolute_current=True.
    fraction_of_total_current: total "external" current fraction. Used if
      use_absolute_current=False.
    gaussian_width: width of "external" Gaussian current profile
    gaussian_location: normalized radius of "external" Gaussian current profile
    use_absolute_current: Toggles if external current is provided absolutely or
      as a fraction of Ip.
  """

  model_name: Literal['gaussian'] = 'gaussian'
  I_generic: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      3.0e6
  )
  fraction_of_total_current: torax_pydantic.UnitIntervalTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.2)
  )
  gaussian_width: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.05)
  )
  gaussian_location: torax_pydantic.UnitIntervalTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.4)
  )
  use_absolute_current: bool = False
  mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED

  @property
  def model_func(self) -> source.SourceProfileFunction:
    return calculate_generic_current

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(
        prescribed_values=tuple(
            [v.get_value(t) for v in self.prescribed_values]
        ),
        I_generic=self.I_generic.get_value(t),
        fraction_of_total_current=self.fraction_of_total_current.get_value(t),
        gaussian_width=self.gaussian_width.get_value(t),
        gaussian_location=self.gaussian_location.get_value(t),
        use_absolute_current=self.use_absolute_current,
    )

  def build_source(self) -> GenericCurrentSource:
    return GenericCurrentSource(model_func=self.model_func)

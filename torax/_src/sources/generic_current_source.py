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
from typing import Annotated, ClassVar, Literal

import chex
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import math_utils
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.physics import psi_calculations
from torax._src.sources import base as source_base
from torax._src.sources import runtime_params as sources_runtime_params_lib
from torax._src.sources import source
from torax._src.sources import source_profiles
from torax._src.torax_pydantic import torax_pydantic

# Default value for the model function to be used for the generic current
# source. This is also used as an identifier for the model function in
# the default source config for Pydantic to "discriminate" against.
DEFAULT_MODEL_FUNCTION_NAME: str = 'gaussian'


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(sources_runtime_params_lib.RuntimeParams):
  """Runtime parameters for the external current source."""

  I_generic: array_typing.FloatScalar
  fraction_of_total_current: array_typing.FloatScalar
  gaussian_width: array_typing.FloatScalar
  gaussian_location: array_typing.FloatScalar
  use_absolute_current: bool


def calculate_generic_current(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    source_name: str,
    unused_state: state.CoreProfiles,
    unused_calculated_source_profiles: source_profiles.SourceProfiles | None,
    unused_conductivity: conductivity_base.Conductivity | None,
) -> tuple[array_typing.FloatVectorCell, ...]:
  """Calculates the external parallel current density profile on the cell grid."""
  source_params = runtime_params.sources[source_name]
  # pytype: enable=name-error
  assert isinstance(source_params, RuntimeParams)
  I_generic = _calculate_I_generic(
      runtime_params,
      source_params,
  )
  # form of external current on cell grid
  generic_current_form = jnp.exp(
      -((geo.rho_norm - source_params.gaussian_location) ** 2)
      / (2 * source_params.gaussian_width**2)
  )

  Cext = I_generic / math_utils.area_integration(generic_current_form, geo)
  j_tor = Cext * generic_current_form

  return (
      psi_calculations.j_toroidal_to_j_parallel(
          j_tor, geo, runtime_params.numerics.min_rho_norm
      ),
  )


def _calculate_I_generic(
    runtime_params: runtime_params_lib.RuntimeParams,
    source_params: RuntimeParams,
) -> chex.Numeric:
  """Calculates the total value of external current."""
  return jnp.where(
      source_params.use_absolute_current,
      source_params.I_generic,
      (
          runtime_params.profile_conditions.Ip
          * source_params.fraction_of_total_current
      ),
  )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
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
    I_generic: total "external" current in A. Used if use_absolute_current=True.
    fraction_of_total_current: total "external" current fraction. Used if
      use_absolute_current=False.
    gaussian_width: width of "external" Gaussian current profile
    gaussian_location: normalized radius of "external" Gaussian current profile
    use_absolute_current: Toggles if external current is provided absolutely or
      as a fraction of Ip.
  """

  model_name: Annotated[Literal['gaussian'], torax_pydantic.JAX_STATIC] = (
      'gaussian'
  )
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
  mode: Annotated[
      sources_runtime_params_lib.Mode, torax_pydantic.JAX_STATIC
  ] = sources_runtime_params_lib.Mode.MODEL_BASED

  @property
  def model_func(self) -> source.SourceProfileFunction:
    return calculate_generic_current

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
        I_generic=self.I_generic.get_value(t),
        fraction_of_total_current=self.fraction_of_total_current.get_value(t),
        gaussian_width=self.gaussian_width.get_value(t),
        gaussian_location=self.gaussian_location.get_value(t),
        use_absolute_current=self.use_absolute_current,
    )

  def build_source(self) -> GenericCurrentSource:
    return GenericCurrentSource(model_func=self.model_func)

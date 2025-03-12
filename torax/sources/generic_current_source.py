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

from __future__ import annotations

import dataclasses
from typing import ClassVar, Literal

import chex
from jax import numpy as jnp
from torax import array_typing
from torax import jax_utils
from torax import math_utils
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.sources import base as source_base
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source
from torax.sources import source_profiles
from torax.torax_pydantic import torax_pydantic


# pylint: disable=invalid-name
class GenericCurrentSourceConfig(source_base.SourceModelBase):
  """Configuration for the GenericCurrentSource.

  Attributes:
    Iext: total "external" current in MA. Used if use_absolute_current=True.
    fext: total "external" current fraction. Used if use_absolute_current=False.
    wext: width of "external" Gaussian current profile
    rext: normalized radius of "external" Gaussian current profile
    use_absolute_current: Toggles if external current is provided absolutely or
      as a fraction of Ip.
  """
  source_name: Literal['generic_current_source'] = 'generic_current_source'
  Iext: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(3.0)
  fext: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(0.2)
  wext: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(0.05)
  rext: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(0.4)
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
        prescribed_values=self.prescribed_values.get_value(t),
        Iext=self.Iext.get_value(t),
        fext=self.fext.get_value(t),
        wext=self.wext.get_value(t),
        rext=self.rext.get_value(t),
        use_absolute_current=self.use_absolute_current,
    )

  def build_source(self) -> GenericCurrentSource:
    return GenericCurrentSource(model_func=self.model_func)


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  """Dynamic runtime parameters for the external current source."""

  Iext: array_typing.ScalarFloat
  fext: array_typing.ScalarFloat
  wext: array_typing.ScalarFloat
  rext: array_typing.ScalarFloat
  use_absolute_current: bool

  def sanity_check(self):
    """Checks that all parameters are valid."""
    jax_utils.error_if_negative(self.wext, 'wext')

  def __post_init__(self):
    self.sanity_check()


# pytype bug: does not treat 'source_models.SourceModels' as a forward reference
# pytype: disable=name-error
def calculate_generic_current(
    unused_static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    source_name: str,
    unused_state: state.CoreProfiles,
    unused_calculated_source_profiles: source_profiles.SourceProfiles | None,
) -> tuple[chex.Array, ...]:
  """Calculates the external current density profiles on the cell grid."""
  dynamic_source_runtime_params = dynamic_runtime_params_slice.sources[
      source_name
  ]
  # pytype: enable=name-error
  assert isinstance(dynamic_source_runtime_params, DynamicRuntimeParams)
  Iext = _calculate_Iext(
      dynamic_runtime_params_slice,
      dynamic_source_runtime_params,
  )
  # form of external current on cell grid
  generic_current_form = jnp.exp(
      -((geo.rho_norm - dynamic_source_runtime_params.rext) ** 2)
      / (2 * dynamic_source_runtime_params.wext**2)
  )

  Cext = (
      Iext
      * 1e6
      / math_utils.area_integration(generic_current_form, geo)
  )

  generic_current_profile = (
      Cext * generic_current_form
  )
  return (generic_current_profile,)


def _calculate_Iext(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    dynamic_source_runtime_params: DynamicRuntimeParams,
) -> chex.Numeric:
  """Calculates the total value of external current."""
  return jnp.where(
      dynamic_source_runtime_params.use_absolute_current,
      dynamic_source_runtime_params.Iext,
      (
          dynamic_runtime_params_slice.profile_conditions.Ip_tot
          * dynamic_source_runtime_params.fext
      ),
  )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class GenericCurrentSource(source.Source):
  """A generic current density source profile."""

  SOURCE_NAME: ClassVar[str] = 'generic_current_source'
  DEFAULT_MODEL_FUNCTION_NAME: ClassVar[str] = 'calc_generic_current'
  model_func: source.SourceProfileFunction = calculate_generic_current

  @property
  def source_name(self) -> str:
    return self.SOURCE_NAME

  @property
  def affected_core_profiles(self) -> tuple[source.AffectedCoreProfile, ...]:
    return (source.AffectedCoreProfile.PSI,)

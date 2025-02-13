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
from typing import ClassVar

import chex
from jax import numpy as jnp
from torax import array_typing
from torax import interpolated_param
from torax import jax_utils
from torax import math_utils
from torax import state
from torax.config import base
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source
from torax.sources import source_profiles
# pylint: disable=invalid-name


@dataclasses.dataclass(kw_only=True)
class RuntimeParams(runtime_params_lib.RuntimeParams):
  """Runtime parameters for the external current source."""

  # total "external" current in MA. Used if use_absolute_current=True.
  Iext: runtime_params_lib.TimeInterpolatedInput = 3.0
  # total "external" current fraction. Used if use_absolute_current=False.
  fext: runtime_params_lib.TimeInterpolatedInput = 0.2
  # width of "external" Gaussian current profile
  wext: runtime_params_lib.TimeInterpolatedInput = 0.05
  # normalized radius of "external" Gaussian current profile
  rext: runtime_params_lib.TimeInterpolatedInput = 0.4

  # Toggles if external current is provided absolutely or as a fraction of Ip.
  use_absolute_current: bool = False
  mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED

  @property
  def grid_type(self) -> base.GridType:
    return base.GridType.CELL

  def make_provider(
      self,
      torax_mesh: geometry.Grid1D | None = None,
  ) -> RuntimeParamsProvider:
    return RuntimeParamsProvider(**self.get_provider_kwargs(torax_mesh))


@chex.dataclass
class RuntimeParamsProvider(runtime_params_lib.RuntimeParamsProvider):
  """Provides runtime parameters for a given time and geometry."""

  runtime_params_config: RuntimeParams
  Iext: interpolated_param.InterpolatedVarSingleAxis
  fext: interpolated_param.InterpolatedVarSingleAxis
  wext: interpolated_param.InterpolatedVarSingleAxis
  rext: interpolated_param.InterpolatedVarSingleAxis

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(**self.get_dynamic_params_kwargs(t))


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
      / math_utils.cell_integration(generic_current_form * geo.spr, geo)
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

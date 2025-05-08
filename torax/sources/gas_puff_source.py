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

"""Gas puff source for the n_e equation."""
import dataclasses
from typing import ClassVar, Literal

import chex
from torax import array_typing
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.sources import base
from torax.sources import formulas
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source
from torax.sources import source_profiles
from torax.torax_pydantic import torax_pydantic


# Default value for the model function to be used for the gas puff
# source. This is also used as an identifier for the model function in
# the default source config for Pydantic to "discriminate" against.
DEFAULT_MODEL_FUNCTION_NAME: str = 'calc_puff_source'


# pylint: disable=invalid-name
@chex.dataclass(frozen=True)
class DynamicGasPuffRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  puff_decay_length: array_typing.ScalarFloat
  S_puff_tot: array_typing.ScalarFloat


# Default formula: exponential with density_reference normalization.
def calc_puff_source(
    unused_static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    source_name: str,
    unused_state: state.CoreProfiles,
    unused_calculated_source_profiles: source_profiles.SourceProfiles | None,
) -> tuple[chex.Array, ...]:
  """Calculates external source term for n from puffs."""
  dynamic_source_runtime_params = dynamic_runtime_params_slice.sources[
      source_name
  ]
  assert isinstance(dynamic_source_runtime_params, DynamicGasPuffRuntimeParams)
  return (formulas.exponential_profile(
      decay_start=1.0,
      width=dynamic_source_runtime_params.puff_decay_length,
      total=(
          dynamic_source_runtime_params.S_puff_tot
          / dynamic_runtime_params_slice.numerics.density_reference
      ),
      geo=geo,
  ),)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class GasPuffSource(source.Source):
  """Gas puff source for the n_e equation."""

  SOURCE_NAME: ClassVar[str] = 'gas_puff'
  model_func: source.SourceProfileFunction = calc_puff_source

  @property
  def source_name(self) -> str:
    return self.SOURCE_NAME

  @property
  def affected_core_profiles(self) -> tuple[source.AffectedCoreProfile, ...]:
    return (source.AffectedCoreProfile.NE,)


class GasPuffSourceConfig(base.SourceModelBase):
  """Gas puff source for the n_e equation.

  Attributes:
    puff_decay_length: exponential decay length of gas puff ionization
      [normalized radial coord]
    S_puff_tot: total gas puff particles/s
  """
  model_name: Literal['calc_puff_source'] = 'calc_puff_source'
  puff_decay_length: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.05)
  )
  S_puff_tot: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1e22)
  )
  mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED

  @property
  def model_func(self) -> source.SourceProfileFunction:
    return calc_puff_source

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> DynamicGasPuffRuntimeParams:
    return DynamicGasPuffRuntimeParams(
        prescribed_values=tuple(
            [v.get_value(t) for v in self.prescribed_values]
        ),
        puff_decay_length=self.puff_decay_length.get_value(t),
        S_puff_tot=self.S_puff_tot.get_value(t),
    )

  def build_source(self) -> GasPuffSource:
    return GasPuffSource(model_func=self.model_func)

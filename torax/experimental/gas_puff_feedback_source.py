# Copyright 2026 DeepMind Technologies Limited
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

"""Gas puff feedback source."""

import dataclasses
import enum
from typing import Annotated, Literal
import chex
import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import math_utils
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.sources import base
from torax._src.sources import formulas
from torax._src.sources import gas_puff_source
from torax._src.sources import runtime_params as sources_runtime_params_lib
from torax._src.sources import source
from torax._src.sources import source_profiles
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name


class AverageType(enum.StrEnum):
  LINE = 'line'
  VOLUME = 'volume'


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(sources_runtime_params_lib.RuntimeParams):
  puff_decay_length: array_typing.FloatScalar
  S_total: array_typing.FloatScalar
  target_average_n_e: array_typing.FloatScalar
  average_type: AverageType = dataclasses.field(metadata={'static': True})
  feedback_gain: array_typing.FloatScalar


def calc_puff_feedback_source(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    source_name: str,
    core_profiles: state.CoreProfiles,
    unused_calculated_source_profiles: source_profiles.SourceProfiles | None,
    unused_conductivity: conductivity_base.Conductivity | None,
) -> tuple[array_typing.FloatVectorCell, ...]:
  """Calculates external source term for n from puffs with feedback."""
  source_params = runtime_params.sources[source_name]
  assert isinstance(source_params, RuntimeParams)

  if source_params.average_type == 'line':
    current_avg_n_e = math_utils.line_average(core_profiles.n_e.value, geo)
  else:
    current_avg_n_e = math_utils.volume_average(core_profiles.n_e.value, geo)

  error = source_params.target_average_n_e - current_avg_n_e

  S_total = source_params.feedback_gain * error
  S_total = jnp.clip(S_total, 0.0, jnp.inf)

  return (
      formulas.exponential_profile(
          decay_start=1.0,
          width=source_params.puff_decay_length,
          total=S_total,
          geo=geo,
      ),
  )


class GasPuffFeedbackSourceConfig(base.SourceModelBase):
  """Gas puff source with feedback for the n_e equation."""

  model_name: Annotated[Literal['feedback'], torax_pydantic.JAX_STATIC] = (
      'feedback'
  )
  puff_decay_length: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.05)
  )
  S_total: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      1e22
  )
  target_average_n_e: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.0)
  )
  average_type: Annotated[AverageType, torax_pydantic.JAX_STATIC] = (
      AverageType.LINE
  )
  feedback_gain: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.0)
  )
  mode: Annotated[
      sources_runtime_params_lib.Mode, torax_pydantic.JAX_STATIC
  ] = sources_runtime_params_lib.Mode.MODEL_BASED

  @property
  def model_func(self) -> source.SourceProfileFunction:
    return calc_puff_feedback_source

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
        puff_decay_length=self.puff_decay_length.get_value(t),
        S_total=self.S_total.get_value(t),
        target_average_n_e=self.target_average_n_e.get_value(t),
        average_type=self.average_type,
        feedback_gain=self.feedback_gain.get_value(t),
    )

  def build_source(self) -> gas_puff_source.GasPuffSource:
    return gas_puff_source.GasPuffSource(model_func=self.model_func)

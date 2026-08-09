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

"""Gas puff feedback source for the electron density (n_e) equation.

This module implements an experimental particle source for electron density that
combines a feedforward gas puff term with a proportional feedback control loop.
The feedback loop adjusts the total particle fueling rate to track a prescribed
line-averaged or volume-averaged electron density target.

To use this source, you must register it with TORAX:
```python
import torax
from torax.experimental import gas_puff_feedback_source
from torax._src.sources import register_model

register_model.register_source_model_config(
    gas_puff_feedback_source.GasPuffFeedbackSourceConfig, 'gas_puff'
)
```
"""

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
  S_feedforward: array_typing.FloatScalar
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

  match source_params.average_type:
    case AverageType.LINE:
      current_avg_n_e = math_utils.line_average(core_profiles.n_e.value, geo)
    case AverageType.VOLUME:
      current_avg_n_e = math_utils.volume_average(core_profiles.n_e.value, geo)
    case _ as unknown:
      raise ValueError(f'Unknown average type: {unknown}')

  error = source_params.target_average_n_e - current_avg_n_e

  S_feedback = source_params.feedback_gain * error
  S_total = source_params.S_feedforward + S_feedback
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
  """Configuration for the gas puff electron density source with feedback control.

  The total particle injection rate is computed using a proportional feedback
  control loop added to a baseline feedforward rate:

    S_total = max(0, S_feedforward + feedback_gain * (target_avg_n_e -
    current_avg_n_e))

  where `current_avg_n_e` is the instantaneous line- or volume-averaged electron
  density. To prevent unphysical particle removal during over-fueling
  transients, the total source is clipped to be non-negative (>= 0).

  The resulting total particle flux [particles/s] is deposited radially using an
  exponential profile decaying inwards from the outer edge boundary (r/a = 1.0)
  with a characteristic width set by `puff_decay_length`.

  Attributes:
    puff_decay_length: Exponential decay length of gas puff ionization from the
      edge boundary inwards [normalized radial coordinate r/a].
    S_feedforward: Baseline feedforward particle injection rate [particles/s].
    average_type: Method for spatial electron density averaging ('line' or
      'volume') used to calculate the feedback error.
    feedback_gain: Proportional feedback gain [m^3 / s] mapping density error to
      an incremental particle fueling rate.
    mode: Defines whether source values are computed from a model, prescribed,
      or loaded from a file (defaults to MODEL_BASED).
    target_average_n_e: Desired target average electron density [m^-3] for
      feedback regulation.
  """

  model_name: Annotated[Literal['feedback'], torax_pydantic.JAX_STATIC] = (
      'feedback'
  )
  puff_decay_length: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.05)
  )
  S_feedforward: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1e22)
  )
  average_type: Annotated[AverageType, torax_pydantic.JAX_STATIC] = (
      AverageType.LINE
  )
  feedback_gain: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  mode: Annotated[
      sources_runtime_params_lib.Mode, torax_pydantic.JAX_STATIC
  ] = sources_runtime_params_lib.Mode.MODEL_BASED

  target_average_n_e: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.77e20)
  )

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
        S_feedforward=self.S_feedforward.get_value(t),
        target_average_n_e=self.target_average_n_e.get_value(t),
        average_type=self.average_type,
        feedback_gain=self.feedback_gain.get_value(t),
    )

  def build_source(self) -> gas_puff_source.GasPuffSource:
    return gas_puff_source.GasPuffSource(model_func=self.model_func)

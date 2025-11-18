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

"""Base pydantic config and model for sawtooth trigger."""

import abc
import dataclasses

import chex
from torax._src import array_typing
from torax._src import state
from torax._src import static_dataclass
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.mhd.sawtooth import runtime_params as sawtooth_runtime_params
from torax._src.torax_pydantic import torax_pydantic


@dataclasses.dataclass(frozen=True, eq=False)
class TriggerModel(static_dataclass.StaticDataclass, abc.ABC):
  """Abstract base class for sawtooth trigger models."""

  @abc.abstractmethod
  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> tuple[array_typing.BoolScalar, array_typing.FloatScalar]:
    """Indicates if a crash is triggered and the radius of the q=1 surface."""


class TriggerConfig(torax_pydantic.BaseModelFrozen):
  """Base config for all trigger models.

  Attributes:
    minimum_radius: The minimum radius of the q=1 surface for triggering.
  """

  minimum_radius: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.05)
  )

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> sawtooth_runtime_params.TriggerRuntimeParams:
    return sawtooth_runtime_params.TriggerRuntimeParams(
        minimum_radius=self.minimum_radius.get_value(t),
    )

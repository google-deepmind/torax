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


"""Class for impurity radiation heat sinks.

Model functions are in separate files.
"""

import dataclasses
from typing import ClassVar

from torax.sources import source as source_lib
from torax.sources import source_models as source_models_lib
from torax.sources.impurity_radiation_heat_sink import impurity_radiation_mavrin_fit


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ImpurityRadiationHeatSink(source_lib.Source):
  """Impurity radiation heat sink for electron heat equation."""

  SOURCE_NAME = "impurity_radiation_heat_sink"
  DEFAULT_MODEL_FUNCTION_NAME: ClassVar[str] = (
      impurity_radiation_mavrin_fit.MODEL_FUNCTION_NAME
  )
  model_func: source_lib.SourceProfileFunction
  source_models: source_models_lib.SourceModels | None = None

  @property
  def source_name(self) -> str:
    return self.SOURCE_NAME

  @property
  def affected_core_profiles(
      self,
  ) -> tuple[source_lib.AffectedCoreProfile, ...]:
    return (source_lib.AffectedCoreProfile.TEMP_EL,)

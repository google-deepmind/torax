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

"""Base class for conductivity models."""
import abc

import chex
from torax._src import state
from torax._src.geometry import geometry as geometry_lib
from torax._src.torax_pydantic import torax_pydantic


@chex.dataclass(kw_only=True, frozen=True)
class Conductivity:
  """Values returned by a conductivity model."""

  sigma: chex.Array
  sigma_face: chex.Array


class ConductivityModel(abc.ABC):
  """Base class for conductivity models."""

  @abc.abstractmethod
  def calculate_conductivity(
      self,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> Conductivity:
    """Calculates conductivity."""


class ConductivityModelConfig(torax_pydantic.BaseModelFrozen, abc.ABC):
  """Base class for conductivity model configs."""

  @abc.abstractmethod
  def build_model(self) -> ConductivityModel:
    """Builds conductivity model."""

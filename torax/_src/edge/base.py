# Copyright 2025 DeepMind Technologies Limited
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

"""Base classes for edge models."""

import abc
import dataclasses
import chex
import jax
from torax._src import state
from torax._src import static_dataclass
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.torax_pydantic import torax_pydantic


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class EdgeModelOutputsBase:
  """Base class for outputs from an edge model.

  Attributes:
    q_parallel: Parallel heat flux upstream [W/m^2].
    heat_flux_perp_to_target: Heat flux perpendicular to the target [W/m^2].
    separatrix_electron_temp: Electron temperature at the separatrix [keV].
    target_electron_temp: Electron temperature at sheath entrance [eV].
    neutral_pressure_in_divertor: Neutral pressure in the divertor [Pa].
  """
  q_parallel: jax.Array
  heat_flux_perp_to_target: jax.Array
  separatrix_electron_temp: jax.Array
  target_electron_temp: jax.Array
  neutral_pressure_in_divertor: jax.Array


@dataclasses.dataclass(frozen=True, eq=False)
class EdgeModelBase(static_dataclass.StaticDataclass, abc.ABC):
  """Abstract base class for edge models."""

  @abc.abstractmethod
  def __call__(
      self,
      runtime_params: runtime_params_slice.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> EdgeModelOutputsBase:
    """Evaluates the edge model at the given time."""
  pass


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams:
  """Base for edge model runtime parameters."""

  pass


class EdgeModelConfigBase(torax_pydantic.BaseModelFrozen, abc.ABC):
  """Base pydantic configuration for all edge models."""

  @abc.abstractmethod
  def build_runtime_params(self, t: chex.Numeric) -> RuntimeParams:
    """Builds the runtime parameters for the edge model at time t."""
    pass

  @abc.abstractmethod
  def build_edge_model(self) -> EdgeModelBase:
    """Builds an edge model from the config."""
    pass

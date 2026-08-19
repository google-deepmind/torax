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
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.edge import runtime_params as edge_runtime_params
from torax._src.geometry import geometry
from torax._src.output_tools import output_grid_context
from torax._src.output_tools import output_keys
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.torax_pydantic import torax_pydantic
import xarray as xr

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class EdgeModelOutputs:
  """Base class for outputs from an edge model.

  Attributes:
    q_parallel: Parallel heat flux upstream [W/m^2].
    q_perpendicular_target: Heat flux perpendicular to the target [W/m^2].
    T_e_separatrix: Electron temperature at the separatrix [keV].
    T_e_target: Electron temperature at sheath entrance [eV].
    pressure_neutral_divertor: Neutral pressure in the divertor [Pa].
  """

  q_parallel: jax.Array
  q_perpendicular_target: jax.Array
  T_e_separatrix: jax.Array
  T_e_target: jax.Array
  pressure_neutral_divertor: jax.Array

  def to_output_dict(
      self, context: output_grid_context.OutputGridContext
  ) -> dict[str, output_grid_context.OutputVar]:
    """Returns a dictionary of standard edge output variable tuples."""
    outputs = {
        output_keys.Q_PARALLEL: context.pack(
            output_keys.Q_PARALLEL, self.q_parallel
        ),
        output_keys.Q_PERPENDICULAR_TARGET: context.pack(
            output_keys.Q_PERPENDICULAR_TARGET, self.q_perpendicular_target
        ),
        output_keys.T_E_SEPARATRIX: context.pack(
            output_keys.T_E_SEPARATRIX, self.T_e_separatrix
        ),
        output_keys.T_E_TARGET: context.pack(
            output_keys.T_E_TARGET, self.T_e_target
        ),
        output_keys.PRESSURE_NEUTRAL_DIVERTOR: context.pack(
            output_keys.PRESSURE_NEUTRAL_DIVERTOR,
            self.pressure_neutral_divertor,
        ),
    }
    return {k: v for k, v in outputs.items() if v is not None}

  def to_xr_datatree(
      self, context: output_grid_context.OutputGridContext
  ) -> xr.DataTree:
    """Builds an xr.DataTree of the edge model outputs."""
    return xr.DataTree(
        dataset=context.build_dataset(
            self.to_output_dict(context),
            coords={output_keys.TIME: context.times},
        )
    )


@dataclasses.dataclass(frozen=True, eq=False)
class EdgeModel(static_dataclass.StaticDataclass, abc.ABC):
  """Abstract base class for edge models."""

  @abc.abstractmethod
  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      core_sources: source_profiles_lib.SourceProfiles,
      previous_edge_outputs: EdgeModelOutputs | None = None,
  ) -> EdgeModelOutputs:
    """Evaluates the edge model at the given time."""


class EdgeModelConfig(torax_pydantic.BaseModelFrozen, abc.ABC):
  """Base pydantic configuration for all edge models."""

  @abc.abstractmethod
  def build_runtime_params(
      self, t: chex.Numeric
  ) -> edge_runtime_params.RuntimeParams:
    """Builds the runtime parameters for the edge model at time t."""

  @abc.abstractmethod
  def build_edge_model(self) -> EdgeModel:
    """Builds an edge model from the config."""

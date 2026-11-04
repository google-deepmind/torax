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

"""Base class for pedestal structure models."""

import abc
import dataclasses
from torax._src import static_dataclass
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import core_profiles as core_profiles_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_state as pedestal_state_lib
from torax._src.pedestal_model.structure import pedestal_structure
import xarray as xr


@dataclasses.dataclass(frozen=True, eq=False)
class PedestalStructureModel(static_dataclass.StaticDataclass, abc.ABC):
  """Calculates temperature, density, and location of the pedestal."""

  def initial_state(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: core_profiles_lib.CoreProfiles,
  ) -> pedestal_state_lib.StructureState:
    """Returns the initial StructureState for this model.

    Args:
      runtime_params: Runtime parameters for the simulation.
      geo: The magnetic geometry of the tokamak.
      core_profiles: Initial core plasma profiles.

    Returns:
      Initial `StructureState` for the pedestal structure model.
    """
    del runtime_params, geo, core_profiles
    return pedestal_state_lib.StructureState()

  def initial_state_from_datatree(
      self,
      datatree: xr.DataTree | None,
      runtime_params: runtime_params_lib.RuntimeParams,
  ) -> pedestal_state_lib.StructureState:
    """Reconstructs StructureState from a DataTree.

    Args:
      datatree: DataTree containing structure model state, or None.
      runtime_params: Runtime parameters for the simulation.

    Returns:
      Reconstructed StructureState.
    """
    del datatree, runtime_params
    return pedestal_state_lib.StructureState()

  @abc.abstractmethod
  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: core_profiles_lib.CoreProfiles,
      structure_state: pedestal_state_lib.StructureState,
  ) -> tuple[
      pedestal_structure.PedestalStructure,
      pedestal_state_lib.StructureState,
  ]:
    """Calculates the pedestal structure properties and updated state.

    Args:
      runtime_params: Runtime parameters for the simulation.
      geo: The magnetic geometry of the tokamak.
      core_profiles: Core plasma profiles.
      structure_state: Current state of the pedestal structure model.

    Returns:
      Tuple of (`PedestalStructure`, updated `StructureState`).
    """

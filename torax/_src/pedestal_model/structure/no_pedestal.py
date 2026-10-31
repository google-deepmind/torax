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
"""A pedestal model for when there is no pedestal."""

import dataclasses
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import profiles
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_state as pedestal_state_lib
from torax._src.pedestal_model.structure import base as structure_base
from torax._src.pedestal_model.structure import pedestal_structure


@dataclasses.dataclass(frozen=True, eq=False)
class NoPedestal(structure_base.PedestalStructureModel):
  """A pedestal structure model for when there is no pedestal.

  This is a placeholder pedestal structure model that is used when there is no
  pedestal. It returns infinite pedestal location and zero temperature and
  density.
  """

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: profiles.CoreProfiles,
      structure_state: pedestal_state_lib.StructureState,
  ) -> tuple[
      pedestal_structure.PedestalStructure,
      pedestal_state_lib.StructureState,
  ]:
    del runtime_params, geo, core_profiles
    return (
        pedestal_structure.PedestalStructure.no_pedestal(),
        structure_state,
    )

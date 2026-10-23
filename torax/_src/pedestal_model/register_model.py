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
"""Register a pedestal model with TORAX."""
from typing import Annotated, get_args
from torax._src.pedestal_model import config
from torax._src.torax_pydantic import model_config


def register_pedestal_structure_model(
    pydantic_model_class: type[config.StructureConfigBase],
):
  """Registers a pedestal structure model with TORAX."""
  inner_union, discriminator = get_args(config.StructureConfig)
  new_union = inner_union | pydantic_model_class
  setattr(
      config,
      'StructureConfig',
      Annotated[new_union, discriminator],
  )
  setattr(
      config.PedestalConfig.model_fields['structure_model'],
      'annotation',
      new_union,
  )
  config.PedestalConfig.model_rebuild(force=True)
  model_config.ToraxConfig.model_rebuild(force=True)

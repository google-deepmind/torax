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
"""Register edge models with TORAX."""

from collections.abc import Mapping
from typing import Annotated, get_args
from torax._src.edge import base
from torax._src.edge import pydantic_model
from torax._src.torax_pydantic import model_config


def register_edge_model(
    pydantic_model_class: type[base.EdgeModelConfig],
) -> None:
  """Registers an edge model with TORAX.

  The registered model can be used both as a top-level edge model and as a
  sub-model inside `CombinedEdgeConfig.sub_models`.

  Args:
    pydantic_model_class: The Pydantic model configuration class to register.
      Must inherit from `base.EdgeModelConfig` and declare a unique `model_name`
      discriminator.
  """
  edge_models, discriminator = get_args(pydantic_model.EdgeConfig)
  new_edge_union = edge_models | pydantic_model_class
  new_edge_config = Annotated[new_edge_union, discriminator]
  setattr(
      pydantic_model,
      'EdgeConfig',
      new_edge_config,
  )
  setattr(
      model_config.ToraxConfig.model_fields['edge'],
      'annotation',
      new_edge_config | None,
  )

  sub_models, sub_discriminator = get_args(pydantic_model.SubModelConfig)
  new_sub_union = sub_models | pydantic_model_class
  new_sub_config = Annotated[new_sub_union, sub_discriminator]
  setattr(
      pydantic_model,
      'SubModelConfig',
      new_sub_config,
  )
  pydantic_model.CombinedEdgeConfig.model_fields['sub_models'].annotation = (
      Mapping[str, new_sub_config]
  )
  pydantic_model.CombinedEdgeConfig.model_rebuild(force=True)
  model_config.ToraxConfig.model_rebuild(force=True)

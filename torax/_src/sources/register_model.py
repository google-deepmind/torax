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
"""Utilities for registering new pydantic configs."""

from torax._src.sources import base
from torax._src.sources import pydantic_config as sources_pydantic_config
from torax._src.torax_pydantic import model_config


def register_source_config(
    source_config_class: type[base.SourceConfigBase],
):
  """Update Pydantic schema to include a source config.

  See torax.torax_pydantic.tests.register_config_test.py for an example of how
  to use this function and expected behavior. Note that calling this function
  will trigger construction of the source object, which could be slow. If this
  causes issues, please raise this in a bug to the TORAX team.

  Args:
    source_config_class: The new source config to register. This
      should be a subclass of SourceConfigBase that implements the interface and
      has a unique `model_name`.
  """
  source = source_config_class().build_source()
  source_id = source.SOURCE_ID
  if source_id == 'ei_exchange':
    raise ValueError(
        'Cannot register a new source model config for the ei_exchange source.'
    )
  if source_id not in sources_pydantic_config.Sources.model_fields:
    raise ValueError(f'The source name {source_id} is not supported.')

  # Update the Sources pydantic model to be aware of the new config.
  field_info = sources_pydantic_config.Sources.model_fields[source_id]
  assert field_info.annotation is not None
  field_info.annotation |= source_config_class
  sources_pydantic_config.Sources.model_rebuild(force=True)
  model_config.ToraxConfig.model_rebuild(force=True)

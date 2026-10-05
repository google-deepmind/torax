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


def _validate_source_model_config(
    source_model_config_class: type[base.SourceConfigBase],
    source_name: str,
):
  """Validates that the source model config is valid."""
  if source_name in ('ei_exchange', 'j_bootstrap'):
    raise ValueError(
        'Cannot register a new source model config for the ei_exchange or'
        ' j_bootstrap sources.'
    )

  source_model_config = source_model_config_class()
  if not hasattr(source_model_config, 'model_name'):
    raise ValueError(
        'The source model config must have a model_name attribute.'
    )
  model_name: str = source_model_config.model_name

  existing_classes = (
      sources_pydantic_config.Sources.get_registered_config_classes(
          source_name
      )
  )
  if not existing_classes:
    raise ValueError(f'The source name {source_name} is not supported.')

  existing_model_names = {
      cls.model_fields['model_name'].default for cls in existing_classes
  }
  if model_name in existing_model_names:
    raise ValueError(
        f'The model name {model_name} is already registered for'
        f' the source {source_name}.'
    )


def register_source_model_config(
    source_model_config_class: type[base.SourceConfigBase],
    source_name: str | None = None,
):
  """Update Pydantic schema to include a source model config.

  See torax.torax_pydantic.tests.register_config_test.py for an example of how
  to use this function and expected behavior.

  Args:
    source_model_config_class: The new source model config to register. This
      should be a subclass of SourceConfigBase that implements the interface and
      has a unique `model_name`.
    source_name: The name of the source to register the model config against.
      If None, inferred from
      `source_model_config_class().build_source().SOURCE_ID`. For the two
      "special" sources ("ei_exchange" and "j_bootstrap") registering a new
      implementation is not supported.
  """
  if source_name is None:
    source_name = source_model_config_class().build_source().SOURCE_ID
  _validate_source_model_config(source_model_config_class, source_name)
  # Update the Sources pydantic model to be aware of the new config.
  field_info = sources_pydantic_config.Sources.model_fields[source_name]
  assert field_info.annotation is not None
  field_info.annotation |= source_model_config_class
  # Rebuild the pydantic schema for both the Sources and ToraxConfig models so
  # that uses of either will have access to the new config.
  sources_pydantic_config.Sources.model_rebuild(force=True)
  model_config.ToraxConfig.model_rebuild(force=True)

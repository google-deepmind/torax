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
from torax.sources import base
from torax.sources import pydantic_model as sources_pydantic_model
from torax.torax_pydantic import model_config


def register_source_model_config(
    source_model_config: type[base.SourceModelBase],
    source_name: str,
):
  """Update Pydantic schema to include a source model config.

  See torax.torax_pydantic.tests.register_config_test.py for an example of how
  to use this function and expected behavior.

  Args:
    source_model_config: The new source model config to register. This should
      be a subclass of SourceModelBase that implements the interface.
    source_name: The name of the source to register the model config against.
      This should be one of the fields in the Sources pydantic model. For the
      two "special" sources ("qei" and "j_bootstrap") registering a new
      implementation is not supported.
  """
  if source_name in ('qei', 'j_bootstrap'):
    raise ValueError(
        'Cannot register a new source model config for the qei or j_bootstrap'
        ' sources.'
    )
  # Update the Sources pydantic model to be aware of the new config.
  sources_pydantic_model.Sources.model_fields[
      f'{source_name}'
  ].annotation |= source_model_config
  # Rebuild the pydantic schema for both the Sources and ToraxConfig models so
  # that uses of either will have access to the new config.
  sources_pydantic_model.Sources.model_rebuild(force=True)
  model_config.ToraxConfig.model_rebuild(force=True)

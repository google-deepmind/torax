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
"""Register a transport model with TORAX."""
from typing import get_args

from torax._src.torax_pydantic import model_config
from torax._src.transport_model import pydantic_model
from torax._src.transport_model import pydantic_model_base


def register_transport_model(
    pydantic_model_class: type[pydantic_model_base.ComponentTransportBase],
):
  """Registers a transport model with TORAX.

  This function adds the transport model to the config model such that it can
  be configured via pydantic. The pydantic model class should inherit from
  ComponentTransportBase and should have a distinct model_name. It should also
  define a build_transport_model method which returns a ComponentTransportModel.

  It can then be used in the `core_transport_models` or
  `pedestal_transport_models` mapping of the transport configuration.

  Args:
    pydantic_model_class: The pydantic model class to register.
  """
  combined_model = model_config.ToraxConfig.model_fields['transport'].annotation
  assert combined_model is pydantic_model.TransportModel

  # The annotation for core_transport_models and pedestal_transport_models is
  # dict[str, UnionType]. We need to extract the value type and extend it.
  for field_name in ('core_transport_models', 'pedestal_transport_models'):
    field_annotation = combined_model.model_fields[field_name].annotation
    dict_args = get_args(field_annotation)
    assert len(dict_args) == 2
    value_type = dict_args[1]
    combined_model.model_fields[field_name].annotation = dict[
        str, value_type | pydantic_model_class
    ]
  combined_model.model_rebuild(force=True)

  model_config.ToraxConfig.model_rebuild(force=True)

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

from typing import Sequence, Union, get_args

from torax._src.torax_pydantic import model_config
from torax._src.transport_model import pydantic_model, pydantic_model_base


def register_transport_model(
    pydantic_model_class: type[pydantic_model_base.TransportBase],
):
    """Registers a transport model with TORAX.

    This functions adds the transport model to the config model such that it can
    be configured via pydantic. The pydantic model class should inherit from
    TransportBase and should have a distinct model_name. It should also define a
    build_transport_model method which returns a TransportModel.

    It can then be used either directly in the transport config or as a transport
    model in a combined transport model.

    Args:
      pydantic_model_class: The pydantic model class to register.
    """
    combined_model, *submodels = get_args(
        model_config.ToraxConfig.model_fields["transport"].annotation
    )
    assert combined_model is pydantic_model.CombinedTransportModel
    assert isinstance(submodels, list)

    # Check if already registered to avoid duplicate registration.
    # We check by model_name since the same class may be re-imported with a
    # different identity when loaded via importlib.
    new_model_name = pydantic_model_class.model_fields["model_name"].default
    for existing_type in submodels:
        if hasattr(existing_type, "model_fields"):
            existing_model_name = existing_type.model_fields.get("model_name", {})
            if hasattr(existing_model_name, "default"):
                if existing_model_name.default == new_model_name:
                    return

    combined_model_types = get_args(
        combined_model.model_fields["transport_models"].annotation
    )
    assert isinstance(combined_model_types, tuple)
    assert len(combined_model_types) == 1
    combined_model.model_fields["transport_models"].annotation = Sequence[
        combined_model_types[0] | pydantic_model_class
    ]
    combined_model.model_rebuild(force=True)

    type_tuple = (combined_model, *submodels, pydantic_model_class)
    model_config.ToraxConfig.model_fields["transport"].annotation = Union[*type_tuple]
    model_config.ToraxConfig.model_rebuild(force=True)

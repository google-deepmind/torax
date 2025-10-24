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
"""Transport model API for TORAX.

This module contains the transport model config and implementation API needed
for interacting with the transport model or implementing a custom transport
model.
"""

# pylint: disable=g-importing-member
from torax._src.transport_model.pydantic_model_base import TransportBase
from torax._src.transport_model.register_model import register_transport_model
from torax._src.transport_model.runtime_params import RuntimeParams
from torax._src.transport_model.transport_model import TransportModel
from torax._src.transport_model.transport_model import TurbulentTransport

__all__ = [
    'RuntimeParams',
    'TransportBase',  # pydantic config
    'TransportModel',  # model interface
    'TurbulentTransport',
    'register_transport_model',
]

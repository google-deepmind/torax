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
"""Pedestal model API for TORAX.

This module contains the pedestal model config and implementation API needed
for interacting with the pedestal model or implementing a custom pedestal
model.
"""

# pylint: disable=g-importing-member
from torax._src.pedestal_model.pedestal_model import PedestalModel
from torax._src.pedestal_model.pedestal_model import PedestalModelOutput
from torax._src.pedestal_model.pydantic_model import BasePedestal
from torax._src.pedestal_model.register_model import register_pedestal_model
from torax._src.pedestal_model.runtime_params import RuntimeParams

__all__ = [
    'BasePedestal',  # pydantic config base class
    'PedestalModel',  # model interface
    'PedestalModelOutput',  # model output
    'RuntimeParams',  # runtime parameters
    'register_pedestal_model',  # registration function
]

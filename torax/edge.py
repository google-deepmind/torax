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
"""Edge model API for TORAX.

This module contains the edge model config and implementation API needed
for interacting with the edge model or implementing a custom edge model.
"""

# pylint: disable=g-importing-member
from torax._src.edge.base import EdgeModel
from torax._src.edge.base import EdgeModelConfig
from torax._src.edge.base import EdgeModelOutputs
from torax._src.edge.register_model import register_edge_model
from torax._src.edge.runtime_params import RuntimeParams

__all__ = [
    'EdgeModel',
    'EdgeModelConfig',
    'EdgeModelOutputs',
    'RuntimeParams',
    'register_edge_model',
]

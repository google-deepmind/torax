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

"""Backwards compatibility shim for combined transport model.

This module re-exports symbols from `torax._src.transport_model.transport_model`
for backwards compatibility.

Use `torax._src.transport_model.transport_model` directly.
"""

from torax._src.transport_model import transport_model

# TODO(b/426132633): Remove backwards compatibility alias.
CombinedTransportModel = transport_model.CombinedTransportModel
MIN_SMOOTHING_WIDTH = transport_model.MIN_SMOOTHING_WIDTH
TransportModel = transport_model.TransportModel

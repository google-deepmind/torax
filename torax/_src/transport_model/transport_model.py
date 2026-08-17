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

"""Backwards compatibility shim.

This module re-exports all symbols from component.py under their original names
to maintain backwards compatibility during the migration.

Use `torax._src.transport_model.component` directly for the new API.
"""

from torax._src.transport_model import component

CHANNEL_CONFIG_STRUCT = component.CHANNEL_CONFIG_STRUCT
ComponentTransportModel = component.ComponentTransportModel
TransportModel = component.ComponentTransportModel
compute_core_domain_mask = component.compute_core_domain_mask
TurbulentTransport = component.TurbulentTransport

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
"""Sources API for TORAX.

This module contains the sources config and implementation API needed
for interacting with the sources or implementing a custom sources.
"""

# pylint: disable=g-importing-member
from torax._src.sources.base import SourceModelBase
from torax._src.sources.register_model import register_source_model_config
from torax._src.sources.runtime_params import Mode
from torax._src.sources.runtime_params import RuntimeParams
from torax._src.sources.source import AffectedCoreProfile
from torax._src.sources.source import Source
from torax._src.sources.source import SourceProfileFunction
from torax._src.sources.source_profiles import SourceProfiles

__all__ = [
    'AffectedCoreProfile',
    'Mode',
    'RuntimeParams',
    'Source',
    'SourceModelBase',
    'SourceProfileFunction',
    'SourceProfiles',
    'register_source_model_config',
]

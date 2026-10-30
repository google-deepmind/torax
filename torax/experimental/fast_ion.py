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
"""Experimental fast-ion physics functionality for TORAX."""

# pylint: disable=g-importing-member
from torax._src.physics.collisions import mikkelsen_fast_ion_heating_fraction
from torax._src.physics.fast_ion import FastIon

__all__ = [
    'FastIon',
    'mikkelsen_fast_ion_heating_fraction',
]

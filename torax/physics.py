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
"""Physics utilities and data structures for TORAX."""

# pylint: disable=g-importing-member
from torax._src.physics.collisions import fast_ion_fractional_heating_formula
from torax._src.physics.fast_ion import FAST_ION_SPECIES
from torax._src.physics.fast_ion import FastIon
from torax._src.physics.fast_ion_utils import bimaxwellian_split

__all__ = [
    'FAST_ION_SPECIES',
    'FastIon',
    'bimaxwellian_split',
    'fast_ion_fractional_heating_formula',
]

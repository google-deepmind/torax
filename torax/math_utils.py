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
"""Math and geometry integration utilities for TORAX."""

# pylint: disable=g-importing-member
from torax._src.math_utils import area_integration
from torax._src.math_utils import cell_integration
from torax._src.math_utils import cell_to_face
from torax._src.math_utils import cumulative_area_integration
from torax._src.math_utils import cumulative_cell_integration
from torax._src.math_utils import cumulative_volume_integration
from torax._src.math_utils import IntegralPreservationQuantity
from torax._src.math_utils import line_average
from torax._src.math_utils import volume_average
from torax._src.math_utils import volume_integration

__all__ = [
    'IntegralPreservationQuantity',
    'area_integration',
    'cell_integration',
    'cell_to_face',
    'cumulative_area_integration',
    'cumulative_cell_integration',
    'cumulative_volume_integration',
    'line_average',
    'volume_average',
    'volume_integration',
]

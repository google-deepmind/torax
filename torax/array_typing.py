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
"""Common array types for using jaxtyping in TORAX."""

# pylint: disable=g-importing-member
from torax._src.array_typing import Array
from torax._src.array_typing import BoolScalar
from torax._src.array_typing import BoolVector
from torax._src.array_typing import BoolVectorCell
from torax._src.array_typing import BoolVectorFace
from torax._src.array_typing import FloatMatrixCell
from torax._src.array_typing import FloatScalar
from torax._src.array_typing import FloatVector
from torax._src.array_typing import FloatVectorCell
from torax._src.array_typing import FloatVectorCellPlusBoundaries
from torax._src.array_typing import FloatVectorFace
from torax._src.array_typing import IntScalar
from torax._src.array_typing import IntVector
from torax._src.array_typing import jaxtyped

__all__ = [
    'Array',
    'BoolScalar',
    'BoolVector',
    'BoolVectorCell',
    'BoolVectorFace',
    'FloatMatrixCell',
    'FloatScalar',
    'FloatVector',
    'FloatVectorCell',
    'FloatVectorCellPlusBoundaries',
    'FloatVectorFace',
    'IntScalar',
    'IntVector',
    'jaxtyped',
]

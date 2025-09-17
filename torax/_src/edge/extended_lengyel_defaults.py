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

"""Default input parameters for the extended Lengyel model."""

from torax._src import array_typing
from typing_extensions import Final

DIVERTOR_BROADENING_FACTOR: Final[array_typing.FloatScalar] = 3.0
RATIO_UPSTREAM_TO_AVG_BPOL: Final[array_typing.FloatScalar] = 4.0 / 3.0
NE_TAU: Final[array_typing.FloatScalar] = 0.5e17
SHEATH_HEAT_TRANSMISSION_FACTOR: Final[array_typing.FloatScalar] = 8.0
FRACTION_OF_PSOL_TO_DIVERTOR: Final[array_typing.FloatScalar] = 2.0 / 3.0
SOL_CONDUCTION_FRACTION: Final[array_typing.FloatScalar] = 1.0
RATIO_MOLECULAR_TO_ION_MASS: Final[array_typing.FloatScalar] = 2.0
WALL_TEMPERATURE: Final[array_typing.FloatScalar] = 300.0
SEPARATRIX_MACH_NUMBER: Final[array_typing.FloatScalar] = 0.0
SEPARATRIX_RATIO_ION_TO_ELECTRON_TEMP: Final[array_typing.FloatScalar] = 1.0
SEPARATRIX_RATIO_ELECTRON_TO_ION_DENSITY: Final[array_typing.FloatScalar] = 1.0
TARGET_RATIO_ION_TO_ELECTRON_TEMP: Final[array_typing.FloatScalar] = 1.0
TARGET_RATIO_ELECTRON_TO_ION_DENSITY: Final[array_typing.FloatScalar] = 1.0
TARGET_MACH_NUMBER: Final[array_typing.FloatScalar] = 1.0
TOROIDAL_FLUX_EXPANSION: Final[array_typing.FloatScalar] = 1.0
INNER_LOOP_ITERATIONS: Final[int] = 5
OUTER_LOOP_ITERATIONS: Final[int] = 5

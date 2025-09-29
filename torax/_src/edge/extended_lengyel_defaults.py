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

import dataclasses
from typing import Final, Mapping
import immutabledict

KAPPA_E_0: Final[float] = (
    2390.0  # Spitzer parallel conductivity prefactor [W/ m (m * ev**3.5)]
)
DIVERTOR_BROADENING_FACTOR: Final[float] = 3.0
TARGET_ANGLE_OF_INCIDENCE: Final[float] = 3.0
RATIO_UPSTREAM_TO_AVG_BPOL: Final[float] = 4.0 / 3.0
NE_TAU: Final[float] = 0.5e17
SHEATH_HEAT_TRANSMISSION_FACTOR: Final[float] = 8.0
FRACTION_OF_PSOL_TO_DIVERTOR: Final[float] = 2.0 / 3.0
SOL_CONDUCTION_FRACTION: Final[float] = 1.0
RATIO_MOLECULAR_TO_ION_MASS: Final[float] = 2.0
WALL_TEMPERATURE: Final[float] = 300.0
SEPARATRIX_MACH_NUMBER: Final[float] = 0.0
SEPARATRIX_RATIO_ION_TO_ELECTRON_TEMP: Final[float] = 1.0
SEPARATRIX_RATIO_ELECTRON_TO_ION_DENSITY: Final[float] = 1.0
TARGET_RATIO_ION_TO_ELECTRON_TEMP: Final[float] = 1.0
TARGET_RATIO_ELECTRON_TO_ION_DENSITY: Final[float] = 1.0
TARGET_MACH_NUMBER: Final[float] = 1.0
TOROIDAL_FLUX_EXPANSION: Final[float] = 1.0
ITERATIONS: Final[int] = 25


@dataclasses.dataclass(frozen=True)
class _FitParams:
  """Parameters for the temperature fit function."""

  amplitude: float
  width: float
  shape: float


# See Table 1 in T. Body et al 2025 Nucl. Fusion 65 086002.
# Exact values taken from
# https://github.com/cfs-energy/extended-lengyel/blob/5b56194/extended_lengyel/curve_fit.yml
TEMPERATURE_FIT_PARAMS: Final[Mapping[str, _FitParams]] = (
    immutabledict.immutabledict({
        'momentum_loss': _FitParams(
            amplitude=0.8858679172531956,
            width=3.8263045353064467,
            shape=0.8282347762381935,
        ),
        'density_ratio': _FitParams(
            amplitude=0.5587910467003282,
            width=2.020427078509838,
            shape=0.9600157520406738,
        ),
        'power_loss': _FitParams(
            amplitude=0.8532115334413933,
            width=5.195481324376164,
            shape=0.9642427916765323,
        ),
    })
)

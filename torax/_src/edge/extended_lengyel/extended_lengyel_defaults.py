# Copyright 2025 DeepMind Technologies Limited
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
ANGLE_OF_INCIDENCE_TARGET: Final[float] = 3.0
RATIO_BPOL_OMP_TO_BPOL_AVG: Final[float] = 4.0 / 3.0
# TODO(b/434175938): (v2) Rename to n_e_tau for consistency.
NE_TAU: Final[float] = 0.5e17
SHEATH_HEAT_TRANSMISSION_FACTOR: Final[float] = 8.0
FRACTION_OF_PSOL_TO_DIVERTOR: Final[float] = 2.0 / 3.0
SOL_CONDUCTION_FRACTION: Final[float] = 1.0
RATIO_MOLECULAR_TO_ION_MASS: Final[float] = 2.0
T_WALL: Final[float] = 300.0
MACH_SEPARATRIX: Final[float] = 0.0
T_I_T_E_RATIO_SEPARATRIX: Final[float] = 1.0
N_E_N_I_RATIO_SEPARATRIX: Final[float] = 1.0
T_I_T_E_RATIO_TARGET: Final[float] = 1.0
N_E_N_I_RATIO_TARGET: Final[float] = 1.0
MACH_TARGET: Final[float] = 1.0
TOROIDAL_FLUX_EXPANSION: Final[float] = 1.0
FIXED_POINT_ITERATIONS: Final[int] = 25
NEWTON_RAPHSON_ITERATIONS: Final[int] = 50
NEWTON_RAPHSON_TOL: Final[float] = 1e-5
HYBRID_FIXED_POINT_ITERATIONS: Final[int] = 5
NEWTON_RAPHSON_TAU_MIN: Final[float] = 1e-4

# Multistart Solver
# Number of guesses to use for the multistart solver in forward mode.
MULTISTART_NUM_GUESSES: Final[int] = 10
# Range for the grid of T_e_target guesses [eV]
MULTISTART_T_E_TARGET_MIN: Final[float] = 1.0
MULTISTART_T_E_TARGET_MAX: Final[float] = 500.0
# Values for alpha_t grid
MULTISTART_ALPHA_T_VALUES: Final[tuple[float, float]] = (0.1, 0.6)
# Tolerances for T_e_target difference for establishing unique roots.
MULTISTART_ROOT_ATOL: Final[float] = 0.01  # eV
MULTISTART_ROOT_RTOL: Final[float] = 0.01


# Physics defaults for initialization. Found by experimentation to be
# robust for a variety of scenarios.
DEFAULT_ALPHA_T_INIT: Final[float] = 0.1
DEFAULT_KAPPA_E_INIT: Final[float] = 1800.0
DEFAULT_C_Z_PREFACTOR_INIT: Final[float] = 1e-4
DEFAULT_T_E_SEPARATRIX_INIT: Final[float] = 200.0  # eV
DEFAULT_T_E_TARGET_INIT_FORWARD: Final[float] = 100.0  # eV


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

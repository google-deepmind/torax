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

"""Library functionality for TORAX."""

# pylint: disable=g-importing-member

import os

import jax
from torax import config
from torax import fvm
from torax import math_utils
from torax import physics
from torax.config import recursive_replace
from torax.constants import CONSTANTS
from torax.geometry import build_chease_geometry
from torax.geometry import build_circular_geometry
from torax.geometry import Geometry
from torax.geometry import Grid1D
from torax.opt import interp
from torax.physics import internal_boundary
from torax.sim import build_sim_from_config
from torax.sim import run_simulation
from torax.sim import Sim
from torax.state import AuxOutput
from torax.state import State
from torax.stepper.stepper import Stepper
from torax.time_step_calculator.chi_time_step_calculator import ChiTimeStepCalculator
from torax.time_step_calculator.fixed_time_step_calculator import FixedTimeStepCalculator
from torax.time_step_calculator.time_step_calculator import TimeStepCalculator
# Unsure why but `from torax.config import Config` doesn't work in some
# circumstances.
Config = config.Config


# pylint: enable=g-importing-member

# TORAX version. Follows semantic versioning: https://semver.org/
__version__ = '0.1.0'

# Default TORAX JAX precision is f64
precision = os.getenv('JAX_PRECISION', 'f64')
assert precision == 'f64' or precision == 'f32', (
    'Unknown JAX precision environment variable: %s' % precision
)
if precision == 'f64':
  jax.config.update('jax_enable_x64', True)

CellVariable = fvm.CellVariable

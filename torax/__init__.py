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
from torax import fvm
from torax import math_utils
from torax import physics
from torax.config import runtime_params as general_runtime_params
from torax.config.config_args import recursive_replace
from torax.constants import CONSTANTS
from torax.geometry import build_circular_geometry
from torax.geometry import build_standard_geometry
from torax.geometry import CircularAnalyticalGeometry
from torax.geometry import Geometry
from torax.geometry import Grid1D
from torax.geometry import StandardGeometry
from torax.geometry import StandardGeometryIntermediates
from torax.geometry_provider import ConstantGeometryProvider
from torax.geometry_provider import GeometryProvider
from torax.physics import internal_boundary
from torax.sim import build_sim_object
from torax.sim import get_consistent_dynamic_runtime_params_slice_and_geometry
from torax.sim import run_simulation
from torax.sim import Sim
from torax.sources.source_profiles import SourceProfiles
from torax.state import CoreProfiles
from torax.state import CoreTransport
from torax.stepper.stepper import Stepper
from torax.time_step_calculator.chi_time_step_calculator import ChiTimeStepCalculator
from torax.time_step_calculator.fixed_time_step_calculator import FixedTimeStepCalculator
from torax.time_step_calculator.time_step_calculator import TimeStepCalculator
# Unsure why but `from torax.config import GeneralRuntimeParams` doesn't work in
# some circumstances.
GeneralRuntimeParams = general_runtime_params.GeneralRuntimeParams


# pylint: enable=g-importing-member

# Default TORAX JAX precision is f64
precision = os.getenv('JAX_PRECISION', 'f64')
assert precision == 'f64' or precision == 'f32', (
    'Unknown JAX precision environment variable: %s' % precision
)
if precision == 'f64':
  jax.config.update('jax_enable_x64', True)

CellVariable = fvm.cell_variable.CellVariable

# Throughout TORAX, we maintain the following canonical argument order for
# common argument names passed to many functions. This is a stylistic
# convention that helps to remember the order of arguments for a function.
# For each individual function only a subset of these are
# passed, but the order should be maintained.
CANONICAL_ORDER = [
    't',
    'dt',
    'source_type',
    'static_runtime_params_slice',
    'dynamic_runtime_params_slice',
    'dynamic_runtime_params_slice_t',
    'dynamic_runtime_params_slice_t_plus_dt',
    'dynamic_runtime_params_slice_provider',
    'unused_config',
    'dynamic_source_runtime_params',
    'geo',
    'geo_t',
    'geo_t_plus_dt',
    'geometry_provider',
    'x_old',
    'state',
    'unused_state',
    'core_profiles',
    'core_profiles_t',
    'core_profiles_t_plus_dt',
    'temp_ion',
    'temp_el',
    'ne',
    'ni',
    'psi',
    'transport_model',
    'source_profiles',
    'source_profile',
    'explicit_source_profiles',
    'source_models',
    'time_step_calculator',
    'coeffs_callback',
    'evolving_names',
    'spectator',
    'explicit',
    'maxiter',
    'tol',
    'delta_reduction_factor',
    'file_restart',
]

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
import os

import jax

# pylint: disable=g-importing-member

from torax.config.config_loader import import_module
from torax.interpolated_param import InterpolatedVarSingleAxis
from torax.interpolated_param import InterpolatedVarTimeRho
from torax.interpolated_param import InterpolationMode
from torax.orchestration.run_simulation import run_simulation
from torax.output import ToraxSimOutputs
from torax.state import SimError
from torax.torax_pydantic.model_config import ToraxConfig

# pylint: enable=g-importing-member


__all__ = [
    'import_module',
    'InterpolatedVarSingleAxis',
    'InterpolatedVarTimeRho',
    'InterpolationMode',
    'run_simulation',
    'SimError',
    'ToraxConfig',
    'ToraxSimOutputs',
]


def set_jax_precision():
  # Default TORAX JAX precision is f64
  precision = os.getenv('JAX_PRECISION', 'f64')
  assert precision == 'f64' or precision == 'f32', (
      'Unknown JAX precision environment variable: %s' % precision
  )
  if precision == 'f64':
    jax.config.update('jax_enable_x64', True)


set_jax_precision()

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
    'static_source_runtime_params',
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
    'source_name',
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
    'model_func',
    'source_models',
    'pedestal_model',
    'time_step_calculator',
    'coeffs_callback',
    'evolving_names',
    'step_fn',
    'spectator',
    'explicit',
    'maxiter',
    'tol',
    'delta_reduction_factor',
    'file_restart',
    'ds',
]

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

"""Adaptive source for internal boundary conditions."""

import jax.numpy as jnp
from torax._src import array_typing
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.internal_boundary_conditions import internal_boundary_conditions as internal_boundary_conditions_lib


def apply_adaptive_source(
    *,
    source_T_i: array_typing.FloatVectorCell,
    source_T_e: array_typing.FloatVectorCell,
    source_n_e: array_typing.FloatVectorCell,
    source_mat_ii: array_typing.FloatVectorCell,
    source_mat_ee: array_typing.FloatVectorCell,
    source_mat_nn: array_typing.FloatVectorCell,
    runtime_params: runtime_params_lib.RuntimeParams,
    internal_boundary_conditions: internal_boundary_conditions_lib.InternalBoundaryConditions,
) -> tuple[
    array_typing.FloatVectorCell,
    array_typing.FloatVectorCell,
    array_typing.FloatVectorCell,
    array_typing.FloatVectorCell,
    array_typing.FloatVectorCell,
    array_typing.FloatVectorCell,
]:
  """Applies an adaptive source to the source profiles to set internal boundary conditions."""

  # Ion temperature
  source_T_i += (
      runtime_params.numerics.adaptive_T_source_prefactor
      * internal_boundary_conditions.T_i
  )
  source_mat_ii -= jnp.where(
      internal_boundary_conditions.T_i != 0.0,
      runtime_params.numerics.adaptive_T_source_prefactor,
      0.0,
  )

  # Electron temperature
  source_T_e += (
      runtime_params.numerics.adaptive_T_source_prefactor
      * internal_boundary_conditions.T_e
  )
  source_mat_ee -= jnp.where(
      internal_boundary_conditions.T_e != 0.0,
      runtime_params.numerics.adaptive_T_source_prefactor,
      0.0,
  )

  # Density
  source_n_e += (
      runtime_params.numerics.adaptive_n_source_prefactor
      * internal_boundary_conditions.n_e
  )
  source_mat_nn -= jnp.where(
      internal_boundary_conditions.n_e != 0.0,
      runtime_params.numerics.adaptive_n_source_prefactor,
      0.0,
  )

  return (
      source_T_i,
      source_T_e,
      source_n_e,
      source_mat_ii,
      source_mat_ee,
      source_mat_nn,
  )

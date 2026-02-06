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

"""Pedestal model that sets the pedestal by adding a large source/sink term."""

import jax.numpy as jnp
from torax._src import array_typing
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib


def apply_adaptive_source(
    *,
    source_T_i: array_typing.FloatVectorCell,
    source_T_e: array_typing.FloatVectorCell,
    source_n_e: array_typing.FloatVectorCell,
    source_mat_ii: array_typing.FloatVectorCell,
    source_mat_ee: array_typing.FloatVectorCell,
    source_mat_nn: array_typing.FloatVectorCell,
    runtime_params: runtime_params_lib.RuntimeParams,
    pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
    geo: geometry.Geometry,
) -> tuple[
    array_typing.FloatVectorCell,
    array_typing.FloatVectorCell,
    array_typing.FloatVectorCell,
    array_typing.FloatVectorCell,
    array_typing.FloatVectorCell,
    array_typing.FloatVectorCell,
]:
  """Applies the adaptive source to the source profiles."""
  pedestal_mask = (
      jnp.zeros_like(geo.rho, dtype=bool)
      .at[pedestal_model_output.rho_norm_ped_top_idx]
      .set(True)
  )

  # Ion temperature
  source_T_i += (
      pedestal_mask
      * runtime_params.numerics.adaptive_T_source_prefactor
      * pedestal_model_output.T_i_ped
  )
  source_mat_ii -= (
      pedestal_mask * runtime_params.numerics.adaptive_T_source_prefactor
  )

  # Electron temperature
  source_T_e += (
      pedestal_mask
      * runtime_params.numerics.adaptive_T_source_prefactor
      * pedestal_model_output.T_e_ped
  )
  source_mat_ee -= (
      pedestal_mask * runtime_params.numerics.adaptive_T_source_prefactor
  )

  # Density
  source_n_e += (
      pedestal_mask
      * runtime_params.numerics.adaptive_n_source_prefactor
      * pedestal_model_output.n_e_ped
  )
  source_mat_nn -= (
      pedestal_mask * runtime_params.numerics.adaptive_n_source_prefactor
  )

  return (
      source_T_i,
      source_T_e,
      source_n_e,
      source_mat_ii,
      source_mat_ee,
      source_mat_nn,
  )

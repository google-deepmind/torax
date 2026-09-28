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
"""Methods for updating runtime parameters based on edge model outputs."""

import dataclasses

import jax
from torax._src import math_utils
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles.plasma_composition import electron_density_ratios
from torax._src.edge import base as edge_base
from torax._src.edge.extended_lengyel import extended_lengyel_model

# pylint: disable=invalid-name


def update_runtime_params(
    runtime_params: runtime_params_lib.RuntimeParams,
    edge_outputs: edge_base.EdgeModelOutputs | None,
) -> runtime_params_lib.RuntimeParams:
  """Updates runtime parameters based on edge model outputs.

  This function takes the outputs from the edge model and updates the
  runtime parameters. This allows the edge model to dynamically control boundary
  conditions (like temperatures at the LCFS) and impurity concentrations.

  Args:
    runtime_params: The current runtime parameters.
    edge_outputs: The outputs from the edge model execution, or None if no edge
      model is active, or if it's the first step of the simulation.

  Returns:
    Updated runtime parameters.
  """
  # If there is no edge model, there is nothing to update.
  if edge_outputs is None:
    return runtime_params

  assert isinstance(runtime_params.edge, extended_lengyel_model.RuntimeParams)

  # Conditionally update temperatures based on the update_temperatures flag.
  runtime_params = jax.lax.cond(
      runtime_params.edge.update_temperatures,  # pyrefly: ignore[missing-attribute]
      lambda runtime_params: _update_temperatures(runtime_params, edge_outputs),
      lambda runtime_params: runtime_params,
      runtime_params,
  )

  # Conditionally update impurities based on the update_impurities flag.
  runtime_params = jax.lax.cond(
      runtime_params.edge.update_impurities,
      lambda runtime_params: _update_impurities(runtime_params, edge_outputs),
      lambda runtime_params: runtime_params,
      runtime_params,
  )

  return runtime_params


def _update_temperatures(
    runtime_params: runtime_params_lib.RuntimeParams,
    edge_outputs: edge_base.EdgeModelOutputs,
) -> runtime_params_lib.RuntimeParams:
  """Updates temperature boundary conditions based on edge model outputs."""
  return dataclasses.replace(
      runtime_params,
      profile_conditions=dataclasses.replace(
          runtime_params.profile_conditions,
          T_e_right_bc=edge_outputs.T_e_right_bc,
          T_i_right_bc=edge_outputs.T_i_right_bc,
      ),
  )


def _update_impurities(
    runtime_params: runtime_params_lib.RuntimeParams,
    edge_outputs: edge_base.EdgeModelOutputs,
) -> runtime_params_lib.RuntimeParams:
  """Updates impurity concentrations based on edge model outputs.

  Iterates over all core impurity species. For each species present in
  `edge_outputs.impurity_right_bc`, computes a uniform scaling factor so that
  the profile value at the right boundary (LCFS) matches the edge-determined
  boundary value, while preserving the user-defined profile shape.

  Important: This requires the user's input profile to have a non-zero value
  at the LCFS. If the profile is zero at the LCFS, the scaling factor is
  effectively infinite but clipped by safe_divide, and the rescaled profile
  remains near zero regardless of the edge model output. A pydantic validator
  in ToraxConfig catches this at config time.

  Args:
    runtime_params: The current runtime parameters.
    edge_outputs: The outputs from the edge model.

  Returns:
    Updated runtime parameters with rescaled core impurity profiles.
  """
  impurity_params = runtime_params.plasma_composition.impurity

  if not isinstance(impurity_params, electron_density_ratios.RuntimeParams):
    raise NotImplementedError(
        'Impurity updates from the edge model are only supported for the'
        ' `n_e_ratios` impurity mode.'
    )

  new_n_e_ratios, new_n_e_ratios_face = {}, {}

  for species, n_e_ratio in impurity_params.n_e_ratios.items():
    if species in edge_outputs.impurity_right_bc:
      target_val_at_edge = edge_outputs.impurity_right_bc[species]
      current_val_at_edge = impurity_params.n_e_ratios_face[species][-1]
      scaling_factor = math_utils.safe_divide(
          num=target_val_at_edge,
          denom=current_val_at_edge,
          eps=1e-7,
      )
    else:
      scaling_factor = 1.0

    new_n_e_ratios[species] = n_e_ratio * scaling_factor
    new_n_e_ratios_face[species] = (
        impurity_params.n_e_ratios_face[species] * scaling_factor
    )

  return dataclasses.replace(
      runtime_params,
      plasma_composition=dataclasses.replace(
          runtime_params.plasma_composition,
          impurity=dataclasses.replace(
              impurity_params,
              n_e_ratios=new_n_e_ratios,
              n_e_ratios_face=new_n_e_ratios_face,
          ),
      ),
  )

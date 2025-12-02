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

import chex
import jax
from torax._src import array_typing
from torax._src import constants
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles.plasma_composition import electron_density_ratios
from torax._src.edge import base as edge_base
from torax._src.edge import extended_lengyel_model
from torax._src.edge import extended_lengyel_standalone

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

  if (
      runtime_params.edge.use_enrichment_model
      and runtime_params.edge.impurity_sot
      == extended_lengyel_model.FixedImpuritySourceOfTruth.CORE
  ):
    # If the enrichment model is used and the core is the source of truth for
    # fixed impurities, then we need to update the enrichment factors in the
    # runtime_params for use in the edge model, consistent with last
    # edge model outputs.
    runtime_params = _update_enrichment_factor(runtime_params, edge_outputs)

  # Conditionally update temperatures based on the update_temperatures flag.
  runtime_params = jax.lax.cond(
      runtime_params.edge.update_temperatures,
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


def _update_enrichment_factor(
    runtime_params: runtime_params_lib.RuntimeParams,
    edge_outputs: edge_base.EdgeModelOutputs,
) -> runtime_params_lib.RuntimeParams:
  """Updates enrichment factors based on edge model outputs."""
  if not isinstance(runtime_params.edge, extended_lengyel_model.RuntimeParams):
    raise ValueError(
        'Enrichment factor updates from the edge model are only supported for'
        ' the extended Lengyel model.'
    )
  if not isinstance(
      edge_outputs, extended_lengyel_standalone.ExtendedLengyelOutputs
  ):
    raise ValueError(
        'Enrichment factor updates from the edge model are only supported for'
        ' the extended Lengyel model.'
    )
  enrichment_factor = edge_outputs.calculated_enrichment
  return dataclasses.replace(
      runtime_params,
      edge=dataclasses.replace(
          runtime_params.edge, enrichment_factor=enrichment_factor
      ),
  )


def _update_temperatures(
    runtime_params: runtime_params_lib.RuntimeParams,
    edge_outputs: edge_base.EdgeModelOutputs,
) -> runtime_params_lib.RuntimeParams:
  """Updates temperature boundary conditions based on edge model outputs."""
  assert isinstance(runtime_params.edge, extended_lengyel_model.RuntimeParams)
  T_e_bc = edge_outputs.separatrix_electron_temp
  T_i_bc = T_e_bc * runtime_params.edge.T_i_T_e_ratio_target
  return dataclasses.replace(
      runtime_params,
      profile_conditions=dataclasses.replace(
          runtime_params.profile_conditions,
          T_e_right_bc=T_e_bc,
          T_i_right_bc=T_i_bc,
      ),
  )


def _calculate_impurity_scaling_factor(
    conc: array_typing.FloatScalar,
    edge_outputs: edge_base.EdgeModelOutputs,
    species: str,
    runtime_params: runtime_params_lib.RuntimeParams,
    impurity_params: electron_density_ratios.RuntimeParams,
) -> chex.Numeric:
  """Calculates the scaling factor for impurity profiles."""
  assert isinstance(runtime_params.edge, extended_lengyel_model.RuntimeParams)
  if runtime_params.edge.use_enrichment_model:
    if not isinstance(
        edge_outputs, extended_lengyel_standalone.ExtendedLengyelOutputs
    ):
      raise ValueError(
          'Impurity updates from the edge model are only supported for the'
          ' extended Lengyel model.'
      )
    enrichment = edge_outputs.calculated_enrichment[species]
  else:
    enrichment = runtime_params.edge.enrichment_factor[species]
  # Enrichment factor is ratio of divertor to upstream concentration.
  # c_core = c_edge / enrichment.
  # Concentration at the LCFS reduced by enrichment factor.
  conc_lcfs = conc / enrichment
  # Calculate scaling from the current value of the profile at the lcfs.
  # This scales the whole profile shape to match the edge value.
  current_val_at_edge = impurity_params.n_e_ratios_face[species][-1]
  return conc_lcfs / (current_val_at_edge + constants.CONSTANTS.eps)


def _update_impurities(
    runtime_params: runtime_params_lib.RuntimeParams,
    edge_outputs: edge_base.EdgeModelOutputs,
) -> runtime_params_lib.RuntimeParams:
  """Updates impurity concentrations based on edge model outputs."""
  if not isinstance(runtime_params.edge, extended_lengyel_model.RuntimeParams):
    raise ValueError(
        'Impurity updates from the edge model are only supported for the'
        ' extended Lengyel model.'
    )
  impurity_params = runtime_params.plasma_composition.impurity

  if not isinstance(impurity_params, electron_density_ratios.RuntimeParams):
    raise NotImplementedError(
        'Impurity updates from the edge model are only supported for the'
        ' `n_e_ratios` impurity mode.'
    )

  new_n_e_ratios, new_n_e_ratios_face = {}, {}

  for species, n_e_ratio in impurity_params.n_e_ratios.items():
    # Case 1: Seeded impurity (Inverse Mode).
    if species in edge_outputs.seed_impurity_concentrations:
      conc = edge_outputs.seed_impurity_concentrations[species]
      scaling_factor = _calculate_impurity_scaling_factor(
          conc, edge_outputs, species, runtime_params, impurity_params
      )

    # Case 2: Fixed impurity with EDGE source of truth.
    elif (
        species in runtime_params.edge.fixed_impurity_concentrations
        and runtime_params.edge.impurity_sot
        == extended_lengyel_model.FixedImpuritySourceOfTruth.EDGE
    ):
      conc = runtime_params.edge.fixed_impurity_concentrations[species]
      scaling_factor = _calculate_impurity_scaling_factor(
          conc, edge_outputs, species, runtime_params, impurity_params
      )
    # Case 3: This species is not updated from the edge model. Leave untouched.
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
              runtime_params.plasma_composition.impurity,
              n_e_ratios=new_n_e_ratios,
              n_e_ratios_face=new_n_e_ratios_face,
          ),
      ),
  )

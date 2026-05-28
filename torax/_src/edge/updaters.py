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
from torax._src import math_utils
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
  T_e_bc = edge_outputs.T_e_separatrix
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
  """Calculates the scaling factor to rescale a core impurity profile.

  The extended Lengyel edge model determines impurity concentrations at the
  divertor. This function computes a uniform scaling factor that, when applied
  to the user-provided core impurity profile, adjusts it so that the profile
  value at the LCFS matches the edge-determined concentration.

  The scaling is computed as:
    conc_lcfs = edge_concentration / enrichment_factor
    scaling_factor = conc_lcfs / current_profile_value_at_lcfs

  Where:
    - `edge_concentration` is the divertor concentration from the edge model.
    - `enrichment_factor` is the ratio of divertor to upstream concentration
      (either calculated by the enrichment model or user-specified).
    - `current_profile_value_at_lcfs` is the user-provided profile's value
      at rho_norm=1.

  The entire core profile is then uniformly scaled by this factor, preserving
  the user-defined profile shape while matching the edge boundary value.

  Important: This requires the user's input profile to have a non-zero value
  at the LCFS. If the profile is zero at the LCFS, the scaling factor is
  effectively infinite but clipped by safe_divide, and the rescaled profile
  remains near zero regardless of the edge model output, silently breaking
  the edge-core coupling. A pydantic validator in ToraxConfig catches this
  at config time.

  Args:
    conc: The target edge concentration for this impurity species.
    edge_outputs: The outputs from the edge model.
    species: The symbol of the impurity species (e.g., 'Ne').
    runtime_params: The current runtime parameters.
    impurity_params: The current core impurity runtime parameters.

  Returns:
    A scaling factor (numeric) to apply to the core impurity profile.
  """
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
  return math_utils.safe_divide(
      num=conc_lcfs, denom=current_val_at_edge, eps=1e-7
  )


def _update_impurities(
    runtime_params: runtime_params_lib.RuntimeParams,
    edge_outputs: edge_base.EdgeModelOutputs,
) -> runtime_params_lib.RuntimeParams:
  """Updates impurity concentrations based on edge model outputs.

  Iterates over all impurity species and applies one of three cases:

  Case 1 - Seeded impurities (inverse mode): The edge model has computed
    a divertor concentration for this species via the inverse solver
    (stored in `edge_outputs.seed_impurity_concentrations`). The core
    profile is rescaled to match the edge-determined LCFS concentration.

  Case 2 - Fixed impurities with EDGE source of truth: The concentration
    is specified in `runtime_params.edge.fixed_impurity_concentrations`
    and the edge config is set as the authoritative source. The core
    profile is rescaled to match this fixed concentration at the LCFS
    (adjusted by the enrichment factor).

  Case 3 - No edge update: The species is either a fixed impurity with
    CORE as the source of truth, or not referenced by the edge model at
    all. The profile is left unchanged (scaling_factor=1.0).

  In cases 1 and 2, the rescaling preserves the user-defined profile shape
  while adjusting the magnitude to match the edge boundary condition.
  See `_calculate_impurity_scaling_factor` for details on the scaling.

  Args:
    runtime_params: The current runtime parameters.
    edge_outputs: The outputs from the edge model.

  Returns:
    Updated runtime parameters with rescaled core impurity profiles.
  """
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

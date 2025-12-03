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

"""Pydantic configs for all edge models, currently only extended_lengyel."""

import logging
from typing import Annotated, Any, Literal, Mapping
import chex
import pydantic
from torax._src.edge import base
from torax._src.edge import extended_lengyel_defaults
from torax._src.edge import extended_lengyel_enums
from torax._src.edge import extended_lengyel_formulas
from torax._src.edge import extended_lengyel_model
from torax._src.torax_pydantic import torax_pydantic
import typing_extensions

# pylint: disable=invalid-name


class ExtendedLengyelConfig(base.EdgeModelConfig):
  """Configuration for the extended Lengyel edge model."""

  model_name: Annotated[
      Literal['extended_lengyel'], torax_pydantic.JAX_STATIC
  ] = 'extended_lengyel'
  # --- Control parameters ---
  computation_mode: Annotated[
      extended_lengyel_enums.ComputationMode, torax_pydantic.JAX_STATIC
  ] = extended_lengyel_enums.ComputationMode.FORWARD
  solver_mode: Annotated[
      extended_lengyel_enums.SolverMode, torax_pydantic.JAX_STATIC
  ] = extended_lengyel_enums.SolverMode.HYBRID
  impurity_sot: Annotated[
      extended_lengyel_model.FixedImpuritySourceOfTruth,
      torax_pydantic.JAX_STATIC,
  ] = extended_lengyel_model.FixedImpuritySourceOfTruth.CORE
  # Flags allowing user to test simulation sensitivity to boundary condition
  # updates, while still providing edge model outputs even if not used.
  update_temperatures: bool = True
  update_impurities: bool = True
  fixed_point_iterations: pydantic.PositiveInt | None = None
  newton_raphson_iterations: pydantic.PositiveInt = (
      extended_lengyel_defaults.NEWTON_RAPHSON_ITERATIONS
  )
  newton_raphson_tol: pydantic.PositiveFloat = (
      extended_lengyel_defaults.NEWTON_RAPHSON_TOL
  )

  # Optional boolean to specify if the geometry is diverted.
  # Required for non-FBT geometries. Not allowed for FBT geometries.
  diverted: torax_pydantic.TimeVaryingScalar | None = None

  # --- Physical parameters ---
  ne_tau: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(extended_lengyel_defaults.NE_TAU)
  )
  divertor_broadening_factor: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(
          extended_lengyel_defaults.DIVERTOR_BROADENING_FACTOR
      )
  )
  sheath_heat_transmission_factor: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(
          extended_lengyel_defaults.SHEATH_HEAT_TRANSMISSION_FACTOR
      )
  )
  fraction_of_P_SOL_to_divertor: (
      torax_pydantic.UnitIntervalTimeVaryingScalar
  ) = torax_pydantic.ValidatedDefault(
      extended_lengyel_defaults.FRACTION_OF_PSOL_TO_DIVERTOR
  )
  SOL_conduction_fraction: torax_pydantic.UnitIntervalTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(
          extended_lengyel_defaults.SOL_CONDUCTION_FRACTION
      )
  )
  ratio_of_molecular_to_ion_mass: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(
          extended_lengyel_defaults.RATIO_MOLECULAR_TO_ION_MASS
      )
  )
  T_wall: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(extended_lengyel_defaults.T_WALL)
  )
  mach_separatrix: torax_pydantic.NonNegativeTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(extended_lengyel_defaults.MACH_SEPARATRIX)
  )
  T_i_T_e_ratio_separatrix: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(
          extended_lengyel_defaults.T_I_T_E_RATIO_SEPARATRIX
      )
  )
  n_e_n_i_ratio_separatrix: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(
          extended_lengyel_defaults.N_E_N_I_RATIO_SEPARATRIX
      )
  )
  T_i_T_e_ratio_target: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(
          extended_lengyel_defaults.T_I_T_E_RATIO_TARGET
      )
  )
  n_e_n_i_ratio_target: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(
          extended_lengyel_defaults.N_E_N_I_RATIO_TARGET
      )
  )
  mach_target: torax_pydantic.NonNegativeTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(extended_lengyel_defaults.MACH_TARGET)
  )

  # Optional inputs that may not be available for all geometry types, but if so
  # will be overridden by TORAX state.
  connection_length_target: torax_pydantic.PositiveTimeVaryingScalar | None = (
      None
  )
  connection_length_divertor: (
      torax_pydantic.PositiveTimeVaryingScalar | None
  ) = None
  angle_of_incidence_target: torax_pydantic.PositiveTimeVaryingScalar | None = (
      None
  )
  toroidal_flux_expansion: torax_pydantic.PositiveTimeVaryingScalar | None = (
      None
  )
  ratio_bpol_omp_to_bpol_avg: (
      torax_pydantic.PositiveTimeVaryingScalar | None
  ) = None

  # Optional input for inverse mode
  T_e_target: torax_pydantic.PositiveTimeVaryingScalar | None = None

  # --- Impurity Configuration ---
  # Will be validated for consistency with plasma_composition impurity symbols.
  seed_impurity_weights: (
      Mapping[str, torax_pydantic.PositiveTimeVaryingScalar] | None
  ) = None
  fixed_impurity_concentrations: Mapping[
      str, torax_pydantic.NonNegativeTimeVaryingScalar
  ] = {}
  # Enrichment is the ratio between divertor impurity concentration and the
  # upstream impurity concentration. It is species-dependent.
  # If `use_enrichment_model` is True, then it does not need to be provided
  # and its value will be calculated from the enrichment model.
  enrichment_factor: (
      Mapping[str, torax_pydantic.PositiveTimeVaryingScalar] | None
  ) = None

  use_enrichment_model: Annotated[bool, torax_pydantic.JAX_STATIC] = True
  enrichment_model_multiplier: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )

  @pydantic.model_validator(mode='before')
  @classmethod
  def _set_default_fixed_point_iterations(cls, data: Any) -> Any:
    if isinstance(data, dict):
      if 'fixed_point_iterations' not in data:
        # Get solver_mode, defaulting to the class default if not in data
        solver_mode = data.get(
            'solver_mode', cls.model_fields['solver_mode'].default
        )
        if solver_mode == extended_lengyel_enums.SolverMode.HYBRID:
          data['fixed_point_iterations'] = (
              extended_lengyel_defaults.HYBRID_FIXED_POINT_ITERATIONS
          )
        else:  # FIXED_POINT or NEWTON_RAPHSON
          data['fixed_point_iterations'] = (
              extended_lengyel_defaults.FIXED_POINT_ITERATIONS
          )
    return data

  @pydantic.model_validator(mode='after')
  def _log_warning_for_unused_enrichment_factor(
      self,
  ) -> typing_extensions.Self:
    """Logs a warning if enrichment_factor is provided when use_enrichment_model is True."""
    if self.use_enrichment_model and self.enrichment_factor is not None:
      logging.warning(
          'enrichment_factor is provided but use_enrichment_model is True. '
          'The provided enrichment_factor will be ignored and values will be '
          'calculated from the enrichment model.'
      )
    return self

  @pydantic.model_validator(mode='after')
  def _validate_enrichment_factor_keys(
      self,
  ) -> typing_extensions.Self:
    """Validates that enrichment_factor keys are the same as impurity keys."""

    if self.use_enrichment_model:
      # No need to validate if enrichment model is used.
      return self

    if self.enrichment_factor is None:
      raise ValueError(
          'enrichment_factor must be provided when use_enrichment_model is'
          ' False.'
      )

    if self.seed_impurity_weights is None:
      impurity_keys = set(self.fixed_impurity_concentrations.keys())
    else:
      impurity_keys = set(self.seed_impurity_weights.keys()) | set(
          self.fixed_impurity_concentrations.keys()
      )

    enrichment_keys = set(self.enrichment_factor.keys())

    if enrichment_keys != impurity_keys:
      missing_keys = impurity_keys - enrichment_keys
      extra_keys = enrichment_keys - impurity_keys

      error_messages = []
      if missing_keys:
        error_messages.append(
            'enrichment_factor is missing keys present in impurity fields:'
            f' {sorted(list(missing_keys))}'
        )
      if extra_keys:
        error_messages.append(
            'enrichment_factor has extra keys not present in impurity fields:'
            f' {sorted(list(extra_keys))}'
        )

      raise ValueError('. '.join(error_messages))
    return self

  @pydantic.model_validator(mode='after')
  def _validate_computation_mode_inputs(
      self,
  ) -> typing_extensions.Self:
    """Validates inputs based on the specified computation mode."""
    if self.computation_mode == extended_lengyel_enums.ComputationMode.FORWARD:
      if self.T_e_target is not None:
        raise ValueError(
            'T_e_target must not be provided for forward computation mode.'
        )
      if (
          self.seed_impurity_weights is not None
          and self.seed_impurity_weights.keys()
      ):
        raise ValueError(
            'seed_impurity_weights must not be provided for forward'
            ' computation mode.'
        )
    elif (
        self.computation_mode == extended_lengyel_enums.ComputationMode.INVERSE
    ):
      if self.T_e_target is None:
        raise ValueError(
            'T_e_target must be provided for inverse computation mode.'
        )
      if not self.seed_impurity_weights:
        raise ValueError(
            'seed_impurity_weights must be provided for inverse computation'
            ' mode.'
        )
    return self

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> extended_lengyel_model.RuntimeParams:
    def _get_optional_value(
        param: torax_pydantic.TimeVaryingScalar | None,
        t: chex.Numeric,
    ):
      """Sets optional values to None if not provided."""
      return param.get_value(t) if param is not None else None

    if self.seed_impurity_weights is None:
      seed_impurity_weights = None
    else:
      seed_impurity_weights = {
          k: v.get_value(t) for k, v in self.seed_impurity_weights.items()
      }
    fixed_impurity_concentrations = {
        k: v.get_value(t) for k, v in self.fixed_impurity_concentrations.items()
    }

    enrichment_model_multiplier = self.enrichment_model_multiplier.get_value(t)

    if self.use_enrichment_model:
      # * if True, the user does not need to provide enrichment_factor in config
      # * However, a `runtime_params.edge.enrichment_factor`` still needs to be
      #   present if `PlasmaComposition.impurity_source_of_truth == CORE`.
      #   It will be used to set the edge fixed impurity concentrations.
      # * Therefore we populate runtime_params.edge.enrichment_factor with
      #   calculated enrichment factors for this case.
      # * For the first timestep, when an EdgeOutputs is not available, we make
      #   an assumption p0=1.0 [Pa] for the divertor neutral pressure.
      enrichment_factor = {}
      if self.seed_impurity_weights is None:
        all_impurities = set(self.fixed_impurity_concentrations.keys())
      else:
        all_impurities = set(self.seed_impurity_weights.keys()) | set(
            self.fixed_impurity_concentrations.keys()
        )
      for species in all_impurities:
        enrichment_factor[species] = (
            extended_lengyel_formulas.calc_enrichment_kallenbach(
                1.0, species, enrichment_model_multiplier
            )
        )
    else:
      if self.enrichment_factor is None:
        raise ValueError(
            'edge model: enrichment_factor must be provided when'
            ' use_enrichment_model is False.'
        )
      enrichment_factor = {
          k: v.get_value(t) for k, v in self.enrichment_factor.items()
      }

    return extended_lengyel_model.RuntimeParams(
        computation_mode=self.computation_mode,
        solver_mode=self.solver_mode,
        impurity_sot=self.impurity_sot,
        diverted=_get_optional_value(self.diverted, t),
        update_temperatures=self.update_temperatures,
        update_impurities=self.update_impurities,
        fixed_point_iterations=self.fixed_point_iterations,
        newton_raphson_iterations=self.newton_raphson_iterations,
        newton_raphson_tol=self.newton_raphson_tol,
        ne_tau=self.ne_tau.get_value(t),
        divertor_broadening_factor=self.divertor_broadening_factor.get_value(t),
        sheath_heat_transmission_factor=self.sheath_heat_transmission_factor.get_value(
            t
        ),
        fraction_of_P_SOL_to_divertor=self.fraction_of_P_SOL_to_divertor.get_value(
            t
        ),
        SOL_conduction_fraction=self.SOL_conduction_fraction.get_value(t),
        ratio_of_molecular_to_ion_mass=self.ratio_of_molecular_to_ion_mass.get_value(
            t
        ),
        T_wall=self.T_wall.get_value(t),
        mach_separatrix=self.mach_separatrix.get_value(t),
        T_i_T_e_ratio_separatrix=self.T_i_T_e_ratio_separatrix.get_value(t),
        n_e_n_i_ratio_separatrix=self.n_e_n_i_ratio_separatrix.get_value(t),
        T_i_T_e_ratio_target=self.T_i_T_e_ratio_target.get_value(t),
        n_e_n_i_ratio_target=self.n_e_n_i_ratio_target.get_value(t),
        mach_target=self.mach_target.get_value(t),
        connection_length_target=_get_optional_value(
            self.connection_length_target, t
        ),
        connection_length_divertor=_get_optional_value(
            self.connection_length_divertor, t
        ),
        angle_of_incidence_target=_get_optional_value(
            self.angle_of_incidence_target, t
        ),
        toroidal_flux_expansion=_get_optional_value(
            self.toroidal_flux_expansion, t
        ),
        ratio_bpol_omp_to_bpol_avg=_get_optional_value(
            self.ratio_bpol_omp_to_bpol_avg, t
        ),
        seed_impurity_weights=seed_impurity_weights,
        fixed_impurity_concentrations=fixed_impurity_concentrations,
        enrichment_factor=enrichment_factor,
        T_e_target=_get_optional_value(self.T_e_target, t),
        use_enrichment_model=self.use_enrichment_model,
        enrichment_model_multiplier=enrichment_model_multiplier,
    )

  def build_edge_model(self) -> extended_lengyel_model.ExtendedLengyelModel:
    return extended_lengyel_model.ExtendedLengyelModel()


EdgeConfig = Annotated[
    ExtendedLengyelConfig,
    pydantic.Field(discriminator='model_name'),
]

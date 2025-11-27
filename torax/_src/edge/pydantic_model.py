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

from typing import Annotated, Any, Literal, Mapping
import chex
import pydantic
from torax._src.edge import base
from torax._src.edge import extended_lengyel_defaults
from torax._src.edge import extended_lengyel_enums
from torax._src.edge import extended_lengyel_model
from torax._src.torax_pydantic import torax_pydantic
import typing_extensions

# pylint: disable=invalid-name


# TODO(b/446608829) - consider splitting into separate configs for inverse
# and forward modes inheriting from a base extended_lengyel config.
# TODO(b/446608829) - decide on final version of namings, possibly more
# consistent with the rest of TORAX
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
  fixed_step_iterations: pydantic.PositiveInt | None = None
  newton_raphson_iterations: pydantic.PositiveInt = (
      extended_lengyel_defaults.NEWTON_RAPHSON_ITERATIONS
  )
  newton_raphson_tol: pydantic.PositiveFloat = (
      extended_lengyel_defaults.NEWTON_RAPHSON_TOL
  )
  # --- Physical parameters ---
  ne_tau: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(extended_lengyel_defaults.NE_TAU)
  )
  divertor_broadening_factor: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(
          extended_lengyel_defaults.DIVERTOR_BROADENING_FACTOR
      )
  )
  ratio_bpol_omp_to_bpol_avg: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(
          extended_lengyel_defaults.RATIO_BPOL_OMP_TO_BPOL_AVG
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
  wall_temperature: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(
          extended_lengyel_defaults.WALL_TEMPERATURE
      )
  )
  separatrix_mach_number: torax_pydantic.NonNegativeTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(
          extended_lengyel_defaults.SEPARATRIX_MACH_NUMBER
      )
  )
  separatrix_ratio_of_ion_to_electron_temp: (
      torax_pydantic.PositiveTimeVaryingScalar
  ) = torax_pydantic.ValidatedDefault(
      extended_lengyel_defaults.SEPARATRIX_RATIO_ION_TO_ELECTRON_TEMP
  )
  separatrix_ratio_of_electron_to_ion_density: (
      torax_pydantic.PositiveTimeVaryingScalar
  ) = torax_pydantic.ValidatedDefault(
      extended_lengyel_defaults.SEPARATRIX_RATIO_ELECTRON_TO_ION_DENSITY
  )
  target_ratio_of_ion_to_electron_temp: (
      torax_pydantic.PositiveTimeVaryingScalar
  ) = torax_pydantic.ValidatedDefault(
      extended_lengyel_defaults.TARGET_RATIO_ION_TO_ELECTRON_TEMP
  )
  target_ratio_of_electron_to_ion_density: (
      torax_pydantic.PositiveTimeVaryingScalar
  ) = torax_pydantic.ValidatedDefault(
      extended_lengyel_defaults.TARGET_RATIO_ELECTRON_TO_ION_DENSITY
  )
  target_mach_number: torax_pydantic.NonNegativeTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(
          extended_lengyel_defaults.TARGET_MACH_NUMBER
      )
  )

  # Optional inputs that may not be available for all geometry types, but if so
  # will be overridden by TORAX state.
  parallel_connection_length: (
      torax_pydantic.PositiveTimeVaryingScalar | None
  ) = None
  divertor_parallel_length: torax_pydantic.PositiveTimeVaryingScalar | None = (
      None
  )
  target_angle_of_incidence: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(
          extended_lengyel_defaults.TARGET_ANGLE_OF_INCIDENCE
      )
  )
  toroidal_flux_expansion: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(
          extended_lengyel_defaults.TOROIDAL_FLUX_EXPANSION
      )
  )

  # Optional input for inverse mode
  target_electron_temp: torax_pydantic.PositiveTimeVaryingScalar | None = None

  # --- Impurity Configuration ---
  # Will be validated for consistency with plasma_composition impurity symbols.
  # TODO(b/446608829) - add validation on the ToraxConfig level following full
  # integration.
  seed_impurity_weights: (
      Mapping[str, torax_pydantic.PositiveTimeVaryingScalar] | None
  ) = None
  fixed_impurity_concentrations: Mapping[
      str, torax_pydantic.NonNegativeTimeVaryingScalar
  ] = {}
  # Enrichment is the ratio between divertor impurity concentration and the
  # upstream impurity concentration. It is used to set boundary condition
  # impurity density for the core, in inverse mode. It is species-dependent.
  # Later we will implement a semi-empirical model for enrichment. For now, we
  # will use a user-provided fixed enrichment value for each species.
  # TODO(b/446608829) - add validation that impurity symbol strings are
  # consistent.
  # TODO(b/446608829) - add option for model-based enrichment based on
  # Kallenbach 2024.
  enrichment_factor: Mapping[str, torax_pydantic.PositiveTimeVaryingScalar] = {}

  @pydantic.model_validator(mode='before')
  @classmethod
  def _set_default_fixed_step_iterations(cls, data: Any) -> Any:
    if isinstance(data, dict):
      if 'fixed_step_iterations' not in data:
        # Get solver_mode, defaulting to the class default if not in data
        solver_mode = data.get(
            'solver_mode', cls.model_fields['solver_mode'].default
        )
        if solver_mode == extended_lengyel_enums.SolverMode.HYBRID:
          data['fixed_step_iterations'] = (
              extended_lengyel_defaults.HYBRID_FIXED_STEP_ITERATIONS
          )
        else:  # FIXED_STEP or NEWTON_RAPHSON
          data['fixed_step_iterations'] = (
              extended_lengyel_defaults.FIXED_STEP_ITERATIONS
          )
    return data

  @pydantic.model_validator(mode='after')
  def _validate_enrichment_factor_keys(
      self,
  ) -> typing_extensions.Self:
    """Validates that enrichment_factor keys are the same as impurity keys."""

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
      if self.target_electron_temp is not None:
        raise ValueError(
            'target_electron_temp must not be provided for forward'
            ' computation mode.'
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
      if self.target_electron_temp is None:
        raise ValueError(
            'target_electron_temp must be provided for inverse computation'
            ' mode.'
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

    return extended_lengyel_model.RuntimeParams(
        computation_mode=self.computation_mode,
        solver_mode=self.solver_mode,
        impurity_sot=self.impurity_sot,
        update_temperatures=self.update_temperatures,
        update_impurities=self.update_impurities,
        fixed_step_iterations=self.fixed_step_iterations,
        newton_raphson_iterations=self.newton_raphson_iterations,
        newton_raphson_tol=self.newton_raphson_tol,
        ne_tau=self.ne_tau.get_value(t),
        divertor_broadening_factor=self.divertor_broadening_factor.get_value(t),
        ratio_bpol_omp_to_bpol_avg=self.ratio_bpol_omp_to_bpol_avg.get_value(t),
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
        wall_temperature=self.wall_temperature.get_value(t),
        separatrix_mach_number=self.separatrix_mach_number.get_value(t),
        separatrix_ratio_of_ion_to_electron_temp=self.separatrix_ratio_of_ion_to_electron_temp.get_value(
            t
        ),
        separatrix_ratio_of_electron_to_ion_density=self.separatrix_ratio_of_electron_to_ion_density.get_value(
            t
        ),
        target_ratio_of_ion_to_electron_temp=self.target_ratio_of_ion_to_electron_temp.get_value(
            t
        ),
        target_ratio_of_electron_to_ion_density=self.target_ratio_of_electron_to_ion_density.get_value(
            t
        ),
        target_mach_number=self.target_mach_number.get_value(t),
        parallel_connection_length=_get_optional_value(
            self.parallel_connection_length, t
        ),
        divertor_parallel_length=_get_optional_value(
            self.divertor_parallel_length, t
        ),
        target_angle_of_incidence=self.target_angle_of_incidence.get_value(t),
        toroidal_flux_expansion=self.toroidal_flux_expansion.get_value(t),
        seed_impurity_weights=seed_impurity_weights,
        fixed_impurity_concentrations=fixed_impurity_concentrations,
        enrichment_factor={
            k: v.get_value(t) for k, v in self.enrichment_factor.items()
        },
        target_electron_temp=_get_optional_value(self.target_electron_temp, t),
    )

  def build_edge_model(self) -> extended_lengyel_model.ExtendedLengyelModel:
    return extended_lengyel_model.ExtendedLengyelModel()


EdgeConfig = Annotated[
    ExtendedLengyelConfig,
    pydantic.Field(discriminator='model_name'),
]

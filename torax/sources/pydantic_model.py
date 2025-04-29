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

"""Pydantic config for source models."""
import copy
from typing import Any

import pydantic
from torax.sources import base
from torax.sources import bootstrap_current_source as bootstrap_current_source_lib
from torax.sources import bremsstrahlung_heat_sink as bremsstrahlung_heat_sink_lib
from torax.sources import cyclotron_radiation_heat_sink as cyclotron_radiation_heat_sink_lib
from torax.sources import electron_cyclotron_source as electron_cyclotron_source_lib
from torax.sources import fusion_heat_source as fusion_heat_source_lib
from torax.sources import gas_puff_source as gas_puff_source_lib
from torax.sources import generic_current_source as generic_current_source_lib
from torax.sources import generic_ion_el_heat_source as generic_ion_el_heat_source_lib
from torax.sources import generic_particle_source as generic_particle_source_lib
from torax.sources import ion_cyclotron_source as ion_cyclotron_source_lib
from torax.sources import ohmic_heat_source as ohmic_heat_source_lib
from torax.sources import pellet_source as pellet_source_lib
from torax.sources import qei_source as qei_source_lib
from torax.sources import runtime_params
from torax.sources.impurity_radiation_heat_sink import impurity_radiation_constant_fraction
from torax.sources.impurity_radiation_heat_sink import impurity_radiation_mavrin_fit
from torax.torax_pydantic import torax_pydantic
from typing_extensions import Self


class Sources(torax_pydantic.BaseModelFrozen):
  """Config for source models.

  The `from_dict` method of constructing this class supports the config
  described in: https://torax.readthedocs.io/en/latest/configuration.html
  """
  j_bootstrap: bootstrap_current_source_lib.BootstrapCurrentSourceConfig = (
      torax_pydantic.ValidatedDefault({'mode': 'ZERO'})
  )
  ei_exchange: qei_source_lib.QeiSourceConfig = torax_pydantic.ValidatedDefault(
      {'mode': 'ZERO'}
  )
  # keep-sorted start
  bremsstrahlung: (
      bremsstrahlung_heat_sink_lib.BremsstrahlungHeatSinkConfig | None
  ) = pydantic.Field(
      discriminator='model_function_name',
      default=None,
  )
  cyclotron_radiation: (
      cyclotron_radiation_heat_sink_lib.CyclotronRadiationHeatSinkConfig | None
  ) = pydantic.Field(
      discriminator='model_function_name',
      default=None,
  )
  ecrh: (
      electron_cyclotron_source_lib.ElectronCyclotronSourceConfig | None
  ) = pydantic.Field(
      discriminator='model_function_name',
      default=None,
  )
  fusion: fusion_heat_source_lib.FusionHeatSourceConfig | None = (
      pydantic.Field(
          discriminator='model_function_name',
          default=None,
      )
  )
  gas_puff: gas_puff_source_lib.GasPuffSourceConfig | None = (
      pydantic.Field(
          discriminator='model_function_name',
          default=None,
      )
  )
  generic_current: (
      generic_current_source_lib.GenericCurrentSourceConfig
  ) = torax_pydantic.ValidatedDefault({'mode': 'ZERO'})
  generic_heat: (
      generic_ion_el_heat_source_lib.GenericIonElHeatSourceConfig | None
  ) = pydantic.Field(
      discriminator='model_function_name',
      default=None,
  )
  generic_particle: (
      generic_particle_source_lib.GenericParticleSourceConfig | None
  ) = pydantic.Field(
      discriminator='model_function_name',
      default=None,
  )
  icrh: (
      ion_cyclotron_source_lib.IonCyclotronSourceConfig | None
  ) = pydantic.Field(
      discriminator='model_function_name',
      default=None,
  )
  impurity_radiation: (
      impurity_radiation_mavrin_fit.ImpurityRadiationHeatSinkMavrinFitConfig
      | impurity_radiation_constant_fraction.ImpurityRadiationHeatSinkConstantFractionConfig
      | None
  ) = pydantic.Field(
      discriminator='model_function_name',
      default=None,
  )
  ohmic: ohmic_heat_source_lib.OhmicHeatSourceConfig | None = (
      pydantic.Field(
          discriminator='model_function_name',
          default=None,
      )
  )
  pellet: pellet_source_lib.PelletSourceConfig | None = pydantic.Field(
      discriminator='model_function_name',
      default=None,
  )
  # keep-sorted end

  @pydantic.model_validator(mode='before')
  @classmethod
  def _set_default_model_functions(cls, x: dict[str, Any]) -> dict[str, Any]:
    constructor_data = copy.deepcopy(x)
    for k, v in x.items():
      # If this an already validated model, skip it.
      if isinstance(v, base.SourceModelBase) or v is None:
        continue
      match k:
        case 'bremsstrahlung':
          if 'model_function_name' not in v:
            constructor_data[k][
                'model_function_name'
            ] = bremsstrahlung_heat_sink_lib.DEFAULT_MODEL_FUNCTION_NAME
        case 'cyclotron_radiation':
          if 'model_function_name' not in v:
            constructor_data[k][
                'model_function_name'
            ] = cyclotron_radiation_heat_sink_lib.DEFAULT_MODEL_FUNCTION_NAME
        case 'ecrh':
          if 'model_function_name' not in v:
            constructor_data[k][
                'model_function_name'
            ] = electron_cyclotron_source_lib.DEFAULT_MODEL_FUNCTION_NAME
        case 'gas_puff':
          if 'model_function_name' not in v:
            constructor_data[k][
                'model_function_name'
            ] = gas_puff_source_lib.DEFAULT_MODEL_FUNCTION_NAME
        case 'generic_particle':
          if 'model_function_name' not in v:
            constructor_data[k][
                'model_function_name'
            ] = generic_particle_source_lib.DEFAULT_MODEL_FUNCTION_NAME
        case 'pellet':
          if 'model_function_name' not in v:
            constructor_data[k][
                'model_function_name'
            ] = pellet_source_lib.DEFAULT_MODEL_FUNCTION_NAME
        case 'fusion':
          if 'model_function_name' not in v:
            constructor_data[k][
                'model_function_name'
            ] = fusion_heat_source_lib.DEFAULT_MODEL_FUNCTION_NAME
        case 'generic_heat':
          if 'model_function_name' not in v:
            constructor_data[k][
                'model_function_name'
            ] = generic_ion_el_heat_source_lib.DEFAULT_MODEL_FUNCTION_NAME
        case 'impurity_radiation':
          if 'model_function_name' not in v:
            constructor_data[k][
                'model_function_name'
            ] = impurity_radiation_mavrin_fit.DEFAULT_MODEL_FUNCTION_NAME
        case 'icrh':
          if 'model_function_name' not in v:
            constructor_data[k][
                'model_function_name'
            ] = ion_cyclotron_source_lib.DEFAULT_MODEL_FUNCTION_NAME
        case 'ohmic':
          if 'model_function_name' not in v:
            constructor_data[k][
                'model_function_name'
            ] = ohmic_heat_source_lib.DEFAULT_MODEL_FUNCTION_NAME
    return constructor_data

  @pydantic.model_validator(mode='after')
  def validate_radiation_models(self) -> Self:
    """Validate that bremsstrahlung and Mavrin models are not both active at the same time.

    This prevents double counting radiation losses.

    Returns:
      Self for method chaining.

    Raises:
      ValueError: If both bremsstrahlung and Mavrin models are active.
    """
    # Check if both sources are defined
    if isinstance(
        self.bremsstrahlung,
        bremsstrahlung_heat_sink_lib.BremsstrahlungHeatSinkConfig,
    ) and isinstance(
        self.impurity_radiation,
        impurity_radiation_mavrin_fit.ImpurityRadiationHeatSinkMavrinFitConfig,
    ):

      bremsstrahlung_active = (
          self.bremsstrahlung.mode != runtime_params.Mode.ZERO
      )

      impurity_active = (
          self.impurity_radiation.mode != runtime_params.Mode.ZERO
      )

      # Only raise error if both are active (not in ZERO mode)
      if bremsstrahlung_active and impurity_active:
        raise ValueError("""
            Both bremsstrahlung and impurity_radiation
            with the Mavrin model should not be active at the same time to avoid
            double-counting Bremstrahlung losses. Please either set one of them
            to Mode.ZERO or remove one of them (most likely Bremstrahlung).
            """)

    return self

  @property
  def source_model_config(self) -> dict[str, base.SourceModelBase]:
    return {
        k: v
        for k, v in self.__dict__.items()
        if isinstance(v, base.SourceModelBase)
    }

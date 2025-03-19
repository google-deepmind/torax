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

from collections.abc import Mapping
import copy
from typing import Any, Union

import pydantic
from torax.sources import bootstrap_current_source
from torax.sources import bremsstrahlung_heat_sink
from torax.sources import cyclotron_radiation_heat_sink
from torax.sources import electron_cyclotron_source
from torax.sources import fusion_heat_source
from torax.sources import gas_puff_source
from torax.sources import generic_current_source
from torax.sources import generic_ion_el_heat_source
from torax.sources import generic_particle_source
from torax.sources import ion_cyclotron_source
from torax.sources import ohmic_heat_source
from torax.sources import pellet_source
from torax.sources import qei_source
from torax.sources import runtime_params
from torax.sources.impurity_radiation_heat_sink import impurity_radiation_constant_fraction
from torax.sources.impurity_radiation_heat_sink import impurity_radiation_mavrin_fit
from torax.torax_pydantic import torax_pydantic
from typing_extensions import Annotated


def get_impurity_heat_sink_discriminator_value(model: dict[str, Any]) -> str:
  """Returns the discriminator value for a given model."""
  # Default to impurity_radiation_mavrin_fit if no model_func is specified.
  return model.get('model_function_name', 'impurity_radiation_mavrin_fit')


ImpurityRadiationHeatSinkConfig = Annotated[
    Union[
        Annotated[
            impurity_radiation_mavrin_fit.ImpurityRadiationHeatSinkMavrinFitConfig,
            pydantic.Tag('impurity_radiation_mavrin_fit'),
        ],
        Annotated[
            impurity_radiation_constant_fraction.ImpurityRadiationHeatSinkConstantFractionConfig,
            pydantic.Tag('radially_constant_fraction_of_Pin'),
        ],
    ],
    pydantic.Field(
        discriminator=pydantic.Discriminator(
            get_impurity_heat_sink_discriminator_value
        )
    ),
]

SourceModelConfig = Union[
    bootstrap_current_source.BootstrapCurrentSourceConfig,
    bremsstrahlung_heat_sink.BremsstrahlungHeatSinkConfig,
    cyclotron_radiation_heat_sink.CyclotronRadiationHeatSinkConfig,
    electron_cyclotron_source.ElectronCyclotronSourceConfig,
    gas_puff_source.GasPuffSourceConfig,
    generic_particle_source.GenericParticleSourceConfig,
    pellet_source.PelletSourceConfig,
    fusion_heat_source.FusionHeatSourceConfig,
    generic_current_source.GenericCurrentSourceConfig,
    generic_ion_el_heat_source.GenericIonElHeatSourceConfig,
    ImpurityRadiationHeatSinkConfig,
    ion_cyclotron_source.IonCyclotronSourceConfig,
    ohmic_heat_source.OhmicHeatSourceConfig,
    qei_source.QeiSourceConfig,
]


class Sources(torax_pydantic.BaseModelFrozen):
  """Config for source models.

  The `from_dict` method of constructing this class supports the config
  described in: https://torax.readthedocs.io/en/latest/configuration.html
  """
  source_model_config: Mapping[str, SourceModelConfig] = pydantic.Field(
      discriminator='source_name'
  )

  @pydantic.model_validator(mode='before')
  @classmethod
  def _conform_data(cls, data: dict[str, Any]) -> dict[str, Any]:
    bootstrap_current_source_name = (
        bootstrap_current_source.BootstrapCurrentSource.SOURCE_NAME
    )
    generic_current_source_name = (
        generic_current_source.GenericCurrentSource.SOURCE_NAME
    )
    qei_source_name = qei_source.QeiSource.SOURCE_NAME
    # If we are running with the standard class constructor (for example after
    # serialisation) then we can skip the validation and return the data.
    if 'source_model_config' in data:
      keys_to_check = {
          bootstrap_current_source_name,
          generic_current_source_name,
          qei_source_name,
      }
      if not keys_to_check.issubset(data['source_model_config'].keys()):
        raise ValueError(
            'The following default source keys are not in the input dict:'
            f' {keys_to_check - set(data["source_model_config"].keys())}'
        )
      return data

    constructor_data = copy.deepcopy(data)
    bootstrap_found = generic_current_found = qei_found = False
    for key in data.keys():
      constructor_data[key].update({'source_name': key})
      if key == bootstrap_current_source_name:
        bootstrap_found = True
      elif key == qei_source_name:
        qei_found = True
      elif key == generic_current_source_name:
        generic_current_found = True

    if not bootstrap_found:
      bootstrap_name = (
          bootstrap_current_source.BootstrapCurrentSource.SOURCE_NAME
      )
      constructor_data[bootstrap_name] = {
          'mode': runtime_params.Mode.ZERO,
          'source_name': bootstrap_name,
      }
    if not generic_current_found:
      generic_current_name = (
          generic_current_source.GenericCurrentSource.SOURCE_NAME
      )
      constructor_data[generic_current_name] = {
          'mode': runtime_params.Mode.ZERO,
          'source_name': generic_current_name,
      }
    if not qei_found:
      constructor_data[qei_source.QeiSource.SOURCE_NAME] = {
          'mode': runtime_params.Mode.ZERO,
          'source_name': qei_source.QeiSource.SOURCE_NAME,
      }

    return {'source_model_config': constructor_data}
    
  @pydantic.model_validator(mode='after')
  def validate_radiation_models(self) -> 'Sources':
    """Validates that bremsstrahlung and Mavrin impurity radiation are not both active.
    
    Both models account for bremsstrahlung radiation, so having both active would
    result in double-counting radiation losses. This validator ensures that users
    don't accidentally configure a simulation with both radiation loss models active
    simultaneously.
    
    The Mavrin model for impurity radiation already includes bremsstrahlung radiation,
    so having a separate bremsstrahlung model active would result in this physical
    process being counted twice in the simulation.
    
    Returns:
      The validated Sources object.
      
    Raises:
      ValueError: If both models are active simultaneously.
    """
    bremsstrahlung_name = 'bremsstrahlung_heat_sink'
    impurity_radiation_name = 'impurity_radiation_heat_sink'
    
    # Extract configs from source_model_config
    bremsstrahlung_config = None
    impurity_config = None
    
    for source_name, source_config in self.source_model_config.items():
      if source_name == bremsstrahlung_name:
        bremsstrahlung_config = source_config
      elif source_name == impurity_radiation_name:
        impurity_config = source_config
    
    if bremsstrahlung_config and impurity_config:
      # Check if both sources are active
      bremsstrahlung_active = (
          getattr(bremsstrahlung_config, 'mode', None) != runtime_params.Mode.ZERO
      )
      
      # Check if impurity radiation is active and using the Mavrin model
      impurity_active = (
          getattr(impurity_config, 'mode', None) != runtime_params.Mode.ZERO
      )
      using_mavrin = (
          getattr(impurity_config, 'model_function_name', 'impurity_radiation_mavrin_fit') 
          == 'impurity_radiation_mavrin_fit'
      )
      
      if bremsstrahlung_active and impurity_active and using_mavrin:
        raise ValueError(
            f"Both {bremsstrahlung_name} and {impurity_radiation_name} with Mavrin model "
            f"should not be active at the same time to avoid double-counting radiation losses. "
            f"Please set one of them to Mode.ZERO or remove one of these sources from the config entirely."
        )
    
    return self

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
from typing import Any, Union

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
from torax.sources.impurity_radiation_heat_sink import impurity_radiation_constant_fraction
from torax.sources.impurity_radiation_heat_sink import impurity_radiation_mavrin_fit
from torax.torax_pydantic import torax_pydantic
from typing_extensions import Annotated
from torax.sources import runtime_params


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


class Sources(torax_pydantic.BaseModelFrozen):
  """Config for source models.

  The `from_dict` method of constructing this class supports the config
  described in: https://torax.readthedocs.io/en/latest/configuration.html
  """

  j_bootstrap: bootstrap_current_source_lib.BootstrapCurrentSourceConfig = (
      torax_pydantic.ValidatedDefault({'mode': 'ZERO'})
  )
  generic_current_source: (
      generic_current_source_lib.GenericCurrentSourceConfig
  ) = torax_pydantic.ValidatedDefault({'mode': 'ZERO'})
  qei_source: qei_source_lib.QeiSourceConfig = torax_pydantic.ValidatedDefault(
      {'mode': 'ZERO'}
  )
  bremsstrahlung_heat_sink: (
      bremsstrahlung_heat_sink_lib.BremsstrahlungHeatSinkConfig | None
  ) = pydantic.Field(default=None)
  cyclotron_radiation_heat_sink: (
      cyclotron_radiation_heat_sink_lib.CyclotronRadiationHeatSinkConfig | None
  ) = pydantic.Field(default=None)
  electron_cyclotron_source: (
      electron_cyclotron_source_lib.ElectronCyclotronSourceConfig | None
  ) = pydantic.Field(default=None)
  gas_puff_source: gas_puff_source_lib.GasPuffSourceConfig | None = (
      pydantic.Field(default=None)
  )
  generic_particle_source: (
      generic_particle_source_lib.GenericParticleSourceConfig | None
  ) = pydantic.Field(default=None)
  pellet_source: pellet_source_lib.PelletSourceConfig | None = pydantic.Field(
      default=None
  )
  fusion_heat_source: fusion_heat_source_lib.FusionHeatSourceConfig | None = (
      pydantic.Field(default=None)
  )
  generic_ion_el_heat_source: (
      generic_ion_el_heat_source_lib.GenericIonElHeatSourceConfig | None
  ) = pydantic.Field(default=None)
  impurity_radiation_heat_sink: ImpurityRadiationHeatSinkConfig | None = (
      pydantic.Field(default=None)
  )
  ion_cyclotron_source: (
      ion_cyclotron_source_lib.IonCyclotronSourceConfig | None
  ) = pydantic.Field(default=None)
  ohmic_heat_source: ohmic_heat_source_lib.OhmicHeatSourceConfig | None = (
      pydantic.Field(default=None)
  )

  @pydantic.model_validator(mode='after')
  def validate_radiation_models(self) -> 'Sources':
    """Validate that bremsstrahlung and Mavrin models are not both active at the same time.
    
    This prevents double counting radiation losses.
    
    Returns:
      Self for method chaining.
      
    Raises:
      ValueError: If both bremsstrahlung and Mavrin models are active.
    """
    # Check if both sources are defined
    if (self.bremsstrahlung_heat_sink is not None and 
        self.impurity_radiation_heat_sink is not None):
        
        # Get mode from bremsstrahlung config
        bremsstrahlung_active = (hasattr(self.bremsstrahlung_heat_sink, 'mode') and 
                                self.bremsstrahlung_heat_sink.mode != runtime_params.Mode.ZERO)
        
        # Check if impurity radiation is active and using the Mavrin model
        impurity_active = (hasattr(self.impurity_radiation_heat_sink, 'mode') and 
                          self.impurity_radiation_heat_sink.mode != runtime_params.Mode.ZERO)
        
        # Check if using Mavrin model
        using_mavrin = (hasattr(self.impurity_radiation_heat_sink, 'model_function_name') and 
                       self.impurity_radiation_heat_sink.model_function_name == 'impurity_radiation_mavrin_fit')
        
        # Only raise error if both are active (not in ZERO mode)
        if bremsstrahlung_active and impurity_active and using_mavrin:
            raise ValueError(
                "Both bremsstrahlung_heat_sink and impurity_radiation_heat_sink with Mavrin model "
                "should not be active at the same time to avoid double-counting radiation losses. "
                "Please set one of them to Mode.ZERO."
            )
    
    return self

  @property
  def source_model_config(self) -> dict[str, base.SourceModelBase]:
    return {
        k: v
        for k, v in self.__dict__.items()
        if isinstance(v, base.SourceModelBase)
    }

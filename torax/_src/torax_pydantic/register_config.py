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
"""Utilities for registering new pydantic configs."""
from torax._src.sources import base
from torax._src.sources import bremsstrahlung_heat_sink as bremsstrahlung_heat_sink_lib
from torax._src.sources import cyclotron_radiation_heat_sink as cyclotron_radiation_heat_sink_lib
from torax._src.sources import electron_cyclotron_source as electron_cyclotron_source_lib
from torax._src.sources import fusion_heat_source as fusion_heat_source_lib
from torax._src.sources import gas_puff_source as gas_puff_source_lib
from torax._src.sources import generic_current_source as generic_current_source_lib
from torax._src.sources import generic_ion_el_heat_source as generic_ion_el_heat_source_lib
from torax._src.sources import generic_particle_source as generic_particle_source_lib
from torax._src.sources import ion_cyclotron_source as ion_cyclotron_source_lib
from torax._src.sources import ohmic_heat_source as ohmic_heat_source_lib
from torax._src.sources import pellet_source as pellet_source_lib
from torax._src.sources import pydantic_model as sources_pydantic_model
from torax._src.sources.impurity_radiation_heat_sink import impurity_radiation_heat_sink as impurity_radiation_heat_sink_lib
from torax._src.sources.impurity_radiation_heat_sink import impurity_radiation_mavrin_fit as impurity_radiation_mavrin_fit_lib
from torax._src.torax_pydantic import model_config


def _validate_source_model_config(
    source_model_config_class: type[base.SourceModelBase],
    source_name: str,
):
  """Validates that the source model config is valid."""
  if source_name in ('qei', 'j_bootstrap'):
    raise ValueError(
        'Cannot register a new source model config for the qei or j_bootstrap'
        ' sources.'
    )

  source_model_config = source_model_config_class()
  if not hasattr(source_model_config, 'model_name'):
    raise ValueError(
        'The source model config must have a model_name attribute.'
    )
  model_name: str = source_model_config.model_name

  match source_name:
    case bremsstrahlung_heat_sink_lib.BremsstrahlungHeatSink.SOURCE_NAME:
      default_model_name = (
          bremsstrahlung_heat_sink_lib.DEFAULT_MODEL_FUNCTION_NAME
      )
    case (
        cyclotron_radiation_heat_sink_lib.CyclotronRadiationHeatSink.SOURCE_NAME
    ):
      default_model_name = (
          cyclotron_radiation_heat_sink_lib.DEFAULT_MODEL_FUNCTION_NAME
      )
    case electron_cyclotron_source_lib.ElectronCyclotronSource.SOURCE_NAME:
      default_model_name = (
          electron_cyclotron_source_lib.DEFAULT_MODEL_FUNCTION_NAME
      )
    case gas_puff_source_lib.GasPuffSource.SOURCE_NAME:
      default_model_name = (
          gas_puff_source_lib.DEFAULT_MODEL_FUNCTION_NAME
      )
    case generic_particle_source_lib.GenericParticleSource.SOURCE_NAME:
      default_model_name = (
          generic_particle_source_lib.DEFAULT_MODEL_FUNCTION_NAME
      )
    case pellet_source_lib.PelletSource.SOURCE_NAME:
      default_model_name = (
          pellet_source_lib.DEFAULT_MODEL_FUNCTION_NAME
      )
    case fusion_heat_source_lib.FusionHeatSource.SOURCE_NAME:
      default_model_name = (
          fusion_heat_source_lib.DEFAULT_MODEL_FUNCTION_NAME
      )
    case (
        generic_ion_el_heat_source_lib.GenericIonElectronHeatSource.SOURCE_NAME
    ):
      default_model_name = (
          generic_ion_el_heat_source_lib.DEFAULT_MODEL_FUNCTION_NAME
      )
    case impurity_radiation_heat_sink_lib.ImpurityRadiationHeatSink.SOURCE_NAME:
      default_model_name = (
          impurity_radiation_mavrin_fit_lib.DEFAULT_MODEL_FUNCTION_NAME
      )
    case ion_cyclotron_source_lib.IonCyclotronSource.SOURCE_NAME:
      default_model_name = (
          ion_cyclotron_source_lib.DEFAULT_MODEL_FUNCTION_NAME
      )
    case ohmic_heat_source_lib.OhmicHeatSource.SOURCE_NAME:
      default_model_name = (
          ohmic_heat_source_lib.DEFAULT_MODEL_FUNCTION_NAME
      )
    case generic_current_source_lib.GenericCurrentSource.SOURCE_NAME:
      default_model_name = (
          generic_current_source_lib.DEFAULT_MODEL_FUNCTION_NAME
      )
    case _:
      raise ValueError(f'The source name {source_name} is not supported.')

  if model_name == default_model_name:
    raise ValueError(
        f'The model function name {model_name} must be different from'
        f' the default model function name {default_model_name} for'
        f' the source {source_name}.'
    )


def register_source_model_config(
    source_model_config_class: type[base.SourceModelBase],
    source_name: str,
):
  """Update Pydantic schema to include a source model config.

  See torax.torax_pydantic.tests.register_config_test.py for an example of how
  to use this function and expected behavior.

  Args:
    source_model_config_class: The new source model config to register. This
      should be a subclass of SourceModelBase that implements the interface and
      has a unique `model_name`.
    source_name: The name of the source to register the model config against.
      This should be one of the fields in the Sources pydantic model. For the
      two "special" sources ("qei" and "j_bootstrap") registering a new
      implementation is not supported.
  """
  _validate_source_model_config(source_model_config_class, source_name)
  # Update the Sources pydantic model to be aware of the new config.
  sources_pydantic_model.Sources.model_fields[
      f'{source_name}'
  ].annotation |= source_model_config_class
  # Rebuild the pydantic schema for both the Sources and ToraxConfig models so
  # that uses of either will have access to the new config.
  sources_pydantic_model.Sources.model_rebuild(force=True)
  model_config.ToraxConfig.model_rebuild(force=True)

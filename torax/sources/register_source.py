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
"""Source registry.

This module contains a registry of sources and a utility for retrieving
registered sources.

To register a new source, use the `_register_new_source` helper and add to the
`_REGISTERED_SOURCES` dict. We register a source by telling TORAX what
class to build, the runtime associated with that source and (optionally) the
builder used to make the source class. If a builder is not provided, the
`_register_new_source` helper will create a default builder for you. The source
builder is used to build the source at runtime, and can be used to override
default runtime params with user provided ones from a config file.

All registered sources can be retrieved using the `get_registered_source`
function. This function takes in a source name and returns a `RegisteredSource`
dataclass containing the source class, source builder class, and default runtime
params class. TORAX uses this dataclass to instantiate the source at runtime
overriding any default runtime params with user provided ones from a config
file.

This is an internal feature of TORAX and the number of registered sources is
expected to grow over time as TORAX becomes more feature rich but ultimately be
finite.
"""

import dataclasses
from typing import Type

from torax.sources import bootstrap_current_source
from torax.sources import bremsstrahlung_heat_sink
from torax.sources import cyclotron_radiation_heat_sink
from torax.sources import electron_cyclotron_source
from torax.sources import fusion_heat_source
from torax.sources import gas_puff_source
from torax.sources import generic_current_source
from torax.sources import generic_ion_el_heat_source as ion_el_heat
from torax.sources import generic_particle_source
from torax.sources import ion_cyclotron_source
from torax.sources import ohmic_heat_source
from torax.sources import pellet_source
from torax.sources import qei_source
from torax.sources import runtime_params
from torax.sources import source
from torax.sources.impurity_radiation_heat_sink import impurity_radiation_constant_fraction
from torax.sources.impurity_radiation_heat_sink import impurity_radiation_heat_sink
from torax.sources.impurity_radiation_heat_sink import impurity_radiation_mavrin_fit


@dataclasses.dataclass(frozen=True)
class ModelFunction:
  source_profile_function: source.SourceProfileFunction | None
  runtime_params_class: Type[runtime_params.RuntimeParams]
  source_builder_class: source.SourceBuilderProtocol | None = None


@dataclasses.dataclass(frozen=True)
class SupportedSource:
  """Source that can be used in TORAX and any associated model functions."""

  source_class: Type[source.Source]
  model_functions: dict[str, ModelFunction]


_SUPPORTED_SOURCES = {
    bootstrap_current_source.BootstrapCurrentSource.SOURCE_NAME: SupportedSource(
        source_class=bootstrap_current_source.BootstrapCurrentSource,
        model_functions={
            bootstrap_current_source.BootstrapCurrentSource.DEFAULT_MODEL_FUNCTION_NAME: ModelFunction(
                source_profile_function=None,
                runtime_params_class=bootstrap_current_source.RuntimeParams,
            )
        },
    ),
    generic_current_source.GenericCurrentSource.SOURCE_NAME: SupportedSource(
        source_class=generic_current_source.GenericCurrentSource,
        model_functions={
            generic_current_source.GenericCurrentSource.DEFAULT_MODEL_FUNCTION_NAME: ModelFunction(
                source_profile_function=generic_current_source.calculate_generic_current,
                runtime_params_class=generic_current_source.RuntimeParams,
            )
        },
    ),
    electron_cyclotron_source.ElectronCyclotronSource.SOURCE_NAME: SupportedSource(
        source_class=electron_cyclotron_source.ElectronCyclotronSource,
        model_functions={
            electron_cyclotron_source.ElectronCyclotronSource.DEFAULT_MODEL_FUNCTION_NAME: ModelFunction(
                source_profile_function=electron_cyclotron_source.calc_heating_and_current,
                runtime_params_class=electron_cyclotron_source.RuntimeParams,
            )
        },
    ),
    generic_particle_source.GenericParticleSource.SOURCE_NAME: SupportedSource(
        source_class=generic_particle_source.GenericParticleSource,
        model_functions={
            generic_particle_source.GenericParticleSource.DEFAULT_MODEL_FUNCTION_NAME: ModelFunction(
                source_profile_function=generic_particle_source.calc_generic_particle_source,
                runtime_params_class=generic_particle_source.GenericParticleSourceRuntimeParams,
            )
        },
    ),
    gas_puff_source.GasPuffSource.SOURCE_NAME: SupportedSource(
        source_class=gas_puff_source.GasPuffSource,
        model_functions={
            gas_puff_source.GasPuffSource.DEFAULT_MODEL_FUNCTION_NAME: (
                ModelFunction(
                    source_profile_function=gas_puff_source.calc_puff_source,
                    runtime_params_class=gas_puff_source.GasPuffRuntimeParams,
                )
            )
        },
    ),
    pellet_source.PelletSource.SOURCE_NAME: SupportedSource(
        source_class=pellet_source.PelletSource,
        model_functions={
            pellet_source.PelletSource.DEFAULT_MODEL_FUNCTION_NAME: (
                ModelFunction(
                    source_profile_function=pellet_source.calc_pellet_source,
                    runtime_params_class=pellet_source.PelletRuntimeParams,
                )
            )
        },
    ),
    ion_el_heat.GenericIonElectronHeatSource.SOURCE_NAME: SupportedSource(
        source_class=ion_el_heat.GenericIonElectronHeatSource,
        model_functions={
            ion_el_heat.GenericIonElectronHeatSource.DEFAULT_MODEL_FUNCTION_NAME: ModelFunction(
                source_profile_function=ion_el_heat.default_formula,
                runtime_params_class=ion_el_heat.RuntimeParams,
            )
        },
    ),
    fusion_heat_source.FusionHeatSource.SOURCE_NAME: SupportedSource(
        source_class=fusion_heat_source.FusionHeatSource,
        model_functions={
            fusion_heat_source.FusionHeatSource.DEFAULT_MODEL_FUNCTION_NAME: ModelFunction(
                source_profile_function=fusion_heat_source.fusion_heat_model_func,
                runtime_params_class=fusion_heat_source.FusionHeatSourceRuntimeParams,
            )
        },
    ),
    qei_source.QeiSource.SOURCE_NAME: SupportedSource(
        source_class=qei_source.QeiSource,
        model_functions={
            qei_source.QeiSource.DEFAULT_MODEL_FUNCTION_NAME: ModelFunction(
                source_profile_function=None,
                runtime_params_class=qei_source.RuntimeParams,
            )
        },
    ),
    ohmic_heat_source.OhmicHeatSource.SOURCE_NAME: SupportedSource(
        source_class=ohmic_heat_source.OhmicHeatSource,
        model_functions={
            ohmic_heat_source.OhmicHeatSource.DEFAULT_MODEL_FUNCTION_NAME: (
                ModelFunction(
                    source_profile_function=ohmic_heat_source.ohmic_model_func,
                    runtime_params_class=ohmic_heat_source.OhmicRuntimeParams,
                )
            )
        },
    ),
    bremsstrahlung_heat_sink.BremsstrahlungHeatSink.SOURCE_NAME: SupportedSource(
        source_class=bremsstrahlung_heat_sink.BremsstrahlungHeatSink,
        model_functions={
            bremsstrahlung_heat_sink.BremsstrahlungHeatSink.DEFAULT_MODEL_FUNCTION_NAME: ModelFunction(
                source_profile_function=bremsstrahlung_heat_sink.bremsstrahlung_model_func,
                runtime_params_class=bremsstrahlung_heat_sink.RuntimeParams,
            )
        },
    ),
    ion_cyclotron_source.IonCyclotronSource.SOURCE_NAME: SupportedSource(
        source_class=ion_cyclotron_source.IonCyclotronSource,
        model_functions={
            ion_cyclotron_source.IonCyclotronSource.DEFAULT_MODEL_FUNCTION_NAME: ModelFunction(
                source_profile_function=None,
                runtime_params_class=ion_cyclotron_source.RuntimeParams,
                source_builder_class=ion_cyclotron_source.IonCyclotronSourceBuilder,
            )
        },
    ),
    cyclotron_radiation_heat_sink.CyclotronRadiationHeatSink.SOURCE_NAME: SupportedSource(
        source_class=cyclotron_radiation_heat_sink.CyclotronRadiationHeatSink,
        model_functions={
            cyclotron_radiation_heat_sink.CyclotronRadiationHeatSink.DEFAULT_MODEL_FUNCTION_NAME: ModelFunction(
                source_profile_function=cyclotron_radiation_heat_sink.cyclotron_radiation_albajar,
                runtime_params_class=cyclotron_radiation_heat_sink.RuntimeParams,
            )
        },
    ),
    impurity_radiation_heat_sink.ImpurityRadiationHeatSink.SOURCE_NAME: SupportedSource(
        source_class=impurity_radiation_heat_sink.ImpurityRadiationHeatSink,
        model_functions={
            impurity_radiation_heat_sink.ImpurityRadiationHeatSink.DEFAULT_MODEL_FUNCTION_NAME: ModelFunction(
                source_profile_function=impurity_radiation_mavrin_fit.impurity_radiation_mavrin_fit,
                runtime_params_class=impurity_radiation_mavrin_fit.RuntimeParams,
            ),
            impurity_radiation_constant_fraction.MODEL_FUNCTION_NAME: ModelFunction(
                source_profile_function=impurity_radiation_constant_fraction.radially_constant_fraction_of_Pin,
                runtime_params_class=impurity_radiation_constant_fraction.RuntimeParams,
            ),
        },
    ),
}


def get_supported_source(source_name: str) -> SupportedSource:
  """Used when building a simulation to get the supported source."""
  if source_name in _SUPPORTED_SOURCES:
    return _SUPPORTED_SOURCES[source_name]
  else:
    raise RuntimeError(f'Source:{source_name} has not been registered.')


def register_model_function(
    source_name: str,
    model_function_name: str,
    model_function: source.SourceProfileFunction,
    runtime_params_class: Type[runtime_params.RuntimeParams],
    source_builder_class: source.SourceBuilderProtocol | None = None,
) -> None:
  """Register a model function by adding to one of the supported sources in the registry."""
  if source_name not in _SUPPORTED_SOURCES:
    raise ValueError(f'Source:{source_name} not found under supported sources.')
  if model_function in _SUPPORTED_SOURCES[source_name].model_functions:
    raise ValueError(
        f'Model function:{model_function} has already been registered for'
        f' source:{source_name}.'
    )
  registered_source = _SUPPORTED_SOURCES[source_name]
  registered_source.model_functions[model_function_name] = ModelFunction(
      source_profile_function=model_function,
      runtime_params_class=runtime_params_class,
      source_builder_class=source_builder_class,
  )

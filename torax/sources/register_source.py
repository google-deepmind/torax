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
from torax.sources import electron_cyclotron_source
from torax.sources import electron_density_sources
from torax.sources import fusion_heat_source
from torax.sources import generic_current_source
from torax.sources import generic_ion_el_heat_source as ion_el_heat
from torax.sources import ion_cyclotron_source
from torax.sources import ohmic_heat_source
from torax.sources import qei_source
from torax.sources import runtime_params
from torax.sources import source


@dataclasses.dataclass(frozen=True)
class RegisteredSource:
  source_class: Type[source.Source]
  source_builder_class: source.SourceBuilderProtocol
  default_runtime_params_class: Type[runtime_params.RuntimeParams]


def _register_new_source(
    source_class: Type[source.Source],
    default_runtime_params_class: Type[runtime_params.RuntimeParams],
    source_builder_class: source.SourceBuilderProtocol | None = None,
    links_back: bool = False,
) -> RegisteredSource:
  """Register source class, default runtime params and (optional) builder for this source.

  Args:
    source_class: The source class.
    default_runtime_params_class: The default runtime params class.
    source_builder_class: The source builder class. If None, a default builder
      is created which uses the source class and default runtime params class to
      construct a builder for that source.
    links_back: Whether the source requires a reference to all the source
      models.

  Returns:
    A `RegisteredSource` dataclass containing the source class, source
    builder class, and default runtime params class.
  """
  if source_builder_class is None:
    builder_class = source.make_source_builder(
        source_class,
        runtime_params_type=default_runtime_params_class,
        links_back=links_back,
    )
  else:
    builder_class = source_builder_class

  return RegisteredSource(
      source_class=source_class,
      source_builder_class=builder_class,
      default_runtime_params_class=default_runtime_params_class,
  )


_REGISTERED_SOURCES = {
    bootstrap_current_source.BootstrapCurrentSource.SOURCE_NAME: (
        _register_new_source(
            source_class=bootstrap_current_source.BootstrapCurrentSource,
            default_runtime_params_class=bootstrap_current_source.RuntimeParams,
        )
    ),
    generic_current_source.GenericCurrentSource.SOURCE_NAME: (
        _register_new_source(
            source_class=generic_current_source.GenericCurrentSource,
            default_runtime_params_class=generic_current_source.RuntimeParams,
        )
    ),
    electron_cyclotron_source.ElectronCyclotronSource.SOURCE_NAME: _register_new_source(
        source_class=electron_cyclotron_source.ElectronCyclotronSource,
        default_runtime_params_class=electron_cyclotron_source.RuntimeParams,
    ),
    electron_density_sources.GenericParticleSource.SOURCE_NAME: _register_new_source(
        source_class=electron_density_sources.GenericParticleSource,
        default_runtime_params_class=electron_density_sources.GenericParticleSourceRuntimeParams,
    ),
    electron_density_sources.GasPuffSource.SOURCE_NAME: _register_new_source(
        source_class=electron_density_sources.GasPuffSource,
        default_runtime_params_class=electron_density_sources.GasPuffRuntimeParams,
    ),
    electron_density_sources.PelletSource.SOURCE_NAME: _register_new_source(
        source_class=electron_density_sources.PelletSource,
        default_runtime_params_class=electron_density_sources.PelletRuntimeParams,
    ),
    ion_el_heat.GenericIonElectronHeatSource.SOURCE_NAME: _register_new_source(
        source_class=ion_el_heat.GenericIonElectronHeatSource,
        default_runtime_params_class=ion_el_heat.RuntimeParams,
    ),
    fusion_heat_source.FusionHeatSource.SOURCE_NAME: _register_new_source(
        source_class=fusion_heat_source.FusionHeatSource,
        default_runtime_params_class=fusion_heat_source.FusionHeatSourceRuntimeParams,
    ),
    qei_source.QeiSource.SOURCE_NAME: _register_new_source(
        source_class=qei_source.QeiSource,
        default_runtime_params_class=qei_source.RuntimeParams,
    ),
    ohmic_heat_source.OhmicHeatSource.SOURCE_NAME: _register_new_source(
        source_class=ohmic_heat_source.OhmicHeatSource,
        default_runtime_params_class=ohmic_heat_source.OhmicRuntimeParams,
        links_back=True,
    ),
    bremsstrahlung_heat_sink.BremsstrahlungHeatSink.SOURCE_NAME: (
        _register_new_source(
            source_class=bremsstrahlung_heat_sink.BremsstrahlungHeatSink,
            default_runtime_params_class=bremsstrahlung_heat_sink.RuntimeParams,
        )
    ),
    ion_cyclotron_source.IonCyclotronSource.SOURCE_NAME: _register_new_source(
        source_class=ion_cyclotron_source.IonCyclotronSource,
        default_runtime_params_class=ion_cyclotron_source.RuntimeParams,
    ),
}


def get_registered_source(source_name: str) -> RegisteredSource:
  """Used when building a simulation to get the registered source."""
  if source_name in _REGISTERED_SOURCES:
    return _REGISTERED_SOURCES[source_name]
  else:
    raise RuntimeError(f'Source:{source_name} has not been registered.')

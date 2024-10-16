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
"""Utilities for registering sources.

This module contains a set of utilities for registering sources and retrieving
registered sources.

In TORAX we flexibly support different user provided sources to be active at
runtime. To do so, we use a registration mechanism such that users can register
their own sources and TORAX can look them up at runtime in the registry.

To register a new source, use the `register_new_source` function. This function
takes in a source name, source class, default runtime params class, and an
optional source builder class. The source name is used to identify the source
in the registry. The source class is the class of the source itself. The default
runtime params class is the class of the default runtime params for the source.
And the source builder class is an optional class which inherits from
`SourceBuilderProtocol`. If not provided, then a default source builder is
created which uses the source class and default runtime params class.

Once a source is registered, it can be retrieved using the
`get_registered_source` function. This function takes in a source name and
returns a `RegisteredSource` dataclass containing the source class, source
builder class, and default runtime params class. TORAX uses this dataclass to
instantiate the source at runtime overriding any default runtime params with
user provided ones from a config file.
"""
import dataclasses
from typing import Type

from torax.sources import bootstrap_current_source
from torax.sources import bremsstrahlung_heat_sink
from torax.sources import electron_density_sources
from torax.sources import external_current_source
from torax.sources import fusion_heat_source
from torax.sources import generic_ion_el_heat_source as ion_el_heat
from torax.sources import ohmic_heat_source
from torax.sources import qei_source
from torax.sources import runtime_params
from torax.sources import source

_REGISTERED_SOURCES = {}


@dataclasses.dataclass(frozen=True)
class RegisteredSource:
  source_class: Type[source.Source]
  source_builder_class: source.SourceBuilderProtocol
  default_runtime_params_class: Type[runtime_params.RuntimeParams]


def register_new_source(
    source_name: str,
    source_class: Type[source.Source],
    default_runtime_params_class: Type[runtime_params.RuntimeParams],
    source_builder_class: source.SourceBuilderProtocol | None = None,
    links_back: bool = False,
):
  """Register source class, default runtime params and (optional) builder for this source.

  Args:
    source_name: The name of the source.
    source_class: The source class.
    default_runtime_params_class: The default runtime params class.
    source_builder_class: The source builder class. If None, a default builder
      is created which uses the source class and default runtime params class to
      construct a builder for that source.
    links_back: Whether the source requires a reference to all the source
      models.
  """
  if source_name in _REGISTERED_SOURCES:
    raise ValueError(f'Source:{source_name} has already been registered.')

  if source_builder_class is None:
    builder_class = source.make_source_builder(
        source_class,
        runtime_params_type=default_runtime_params_class,
        links_back=links_back,
    )
  else:
    builder_class = source_builder_class

  _REGISTERED_SOURCES[source_name] = RegisteredSource(
      source_class=source_class,
      source_builder_class=builder_class,
      default_runtime_params_class=default_runtime_params_class,
  )


def get_registered_source(source_name: str) -> RegisteredSource:
  """Used when building a simulation to get the registered source."""
  if source_name in _REGISTERED_SOURCES:
    return _REGISTERED_SOURCES[source_name]
  else:
    raise RuntimeError(f'Source:{source_name} has not been registered.')


def register_torax_sources():
  """Register a set of sources commonly used in TORAX."""
  register_new_source(
      bootstrap_current_source.SOURCE_NAME,
      source_class=bootstrap_current_source.BootstrapCurrentSource,
      default_runtime_params_class=bootstrap_current_source.RuntimeParams,
  )
  register_new_source(
      external_current_source.SOURCE_NAME,
      external_current_source.ExternalCurrentSource,
      default_runtime_params_class=external_current_source.RuntimeParams,
  )
  register_new_source(
      electron_density_sources.GENERIC_PARTICLE_SOURCE_NAME,
      electron_density_sources.NBIParticleSource,
      default_runtime_params_class=electron_density_sources.NBIParticleRuntimeParams,
  )
  register_new_source(
      electron_density_sources.GAS_PUFF_SOURCE_NAME,
      electron_density_sources.GasPuffSource,
      default_runtime_params_class=electron_density_sources.GasPuffRuntimeParams,
  )
  register_new_source(
      electron_density_sources.PELLET_SOURCE_NAME,
      electron_density_sources.PelletSource,
      default_runtime_params_class=electron_density_sources.PelletRuntimeParams,
  )
  register_new_source(
      ion_el_heat.SOURCE_NAME,
      ion_el_heat.GenericIonElectronHeatSource,
      default_runtime_params_class=ion_el_heat.RuntimeParams,
  )
  register_new_source(
      fusion_heat_source.SOURCE_NAME,
      fusion_heat_source.FusionHeatSource,
      default_runtime_params_class=fusion_heat_source.FusionHeatSourceRuntimeParams
  )
  register_new_source(
      qei_source.SOURCE_NAME,
      qei_source.QeiSource,
      default_runtime_params_class=qei_source.RuntimeParams,
  )
  register_new_source(
      ohmic_heat_source.SOURCE_NAME,
      ohmic_heat_source.OhmicHeatSource,
      default_runtime_params_class=ohmic_heat_source.OhmicRuntimeParams,
      links_back=True,
  )
  register_new_source(
      bremsstrahlung_heat_sink.SOURCE_NAME,
      bremsstrahlung_heat_sink.BremsstrahlungHeatSink,
      default_runtime_params_class=bremsstrahlung_heat_sink.RuntimeParams,
  )


register_torax_sources()

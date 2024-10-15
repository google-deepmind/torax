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

"""Default collection of sources and their runtime params.

These are mainly useful for tests, but also they can serve as starting points
when defining new TORAX configs in general.

Many TORAX test files and example configurations use these defaults with minor
tweaks added on top.
"""

from typing import Any
from torax.sources import bootstrap_current_source
from torax.sources import bremsstrahlung_heat_sink
from torax.sources import electron_density_sources
from torax.sources import external_current_source
from torax.sources import fusion_heat_source
from torax.sources import generic_ion_el_heat_source as ion_el_heat
from torax.sources import ohmic_heat_source
from torax.sources import qei_source
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source
from torax.sources import source_models as source_models_lib


def get_default_runtime_params(
    source_name: str,
) -> runtime_params_lib.RuntimeParams:
  """Returns default RuntimeParams for the given source."""
  match source_name:
    case 'j_bootstrap':
      return bootstrap_current_source.RuntimeParams()
    case 'jext':
      return external_current_source.RuntimeParams()
    case 'nbi_particle_source':
      return electron_density_sources.NBIParticleRuntimeParams()
    case 'gas_puff_source':
      return electron_density_sources.GasPuffRuntimeParams()
    case 'pellet_source':
      return electron_density_sources.PelletRuntimeParams()
    case 'generic_ion_el_heat_source':
      return ion_el_heat.RuntimeParams()
    case 'fusion_heat_source':
      return fusion_heat_source.FusionHeatSourceRuntimeParams()
    case 'qei_source':
      return qei_source.RuntimeParams()
    case 'ohmic_heat_source':
      return ohmic_heat_source.OhmicRuntimeParams()
    case 'bremsstrahlung_heat_sink':
      return bremsstrahlung_heat_sink.RuntimeParams()
    case _:
      raise ValueError(f'Unknown source name: {source_name}')


def get_source_type(source_name: str) -> type[source.Source]:
  """Returns a constructor for the given source."""
  match source_name:
    case 'j_bootstrap':
      return bootstrap_current_source.BootstrapCurrentSource
    case 'jext':
      return external_current_source.ExternalCurrentSource
    case 'nbi_particle_source':
      return electron_density_sources.NBIParticleSource
    case 'gas_puff_source':
      return electron_density_sources.GasPuffSource
    case 'pellet_source':
      return electron_density_sources.PelletSource
    case 'generic_ion_el_heat_source':
      return ion_el_heat.GenericIonElectronHeatSource
    case 'fusion_heat_source':
      return fusion_heat_source.FusionHeatSource
    case 'qei_source':
      return qei_source.QeiSource
    case 'ohmic_heat_source':
      return ohmic_heat_source.OhmicHeatSource
    case 'bremsstrahlung_heat_sink':
      return bremsstrahlung_heat_sink.BremsstrahlungHeatSink
    case _:
      raise ValueError(f'Unknown source name: {source_name}')


def get_default_sources_builder() -> source_models_lib.SourceModelsBuilder:
  """Returns a SourceModelsBuilder containing default sources and runtime parameters.

  This set of sources and params are used by most of the TORAX test
  configurations, including ITER-inspired configs, with additional changes to
  their runtime configurations on top.

  If you plan to use them, please remember to update the default runtime
  parameters as needed. Here is an example of how to do so:

  .. code-block:: python

    default_sources: SourceModels = get_default_sources()
    # Turn off bootstrap current.
    default_sources.j_bootstrap.runtime_params.mode = runtime_params.Mode.ZERO
    # Change the Qei ion-electron heat exchange term.
    default_sources.qei_source.runtime_params.Qei_mult = 2.0
    # Turn off fusion power.
    default_sources.sources['fusion_heat_source'].runtime_params.mode = (
        runtime_params.Mode.ZERO
    )

  More examples are located in the test config files under
  `torax/tests/test_data`.
  """
  names = [
      # Current sources (for psi equation)
      'j_bootstrap',
      'jext',
      # Electron density sources/sink (for the ne equation).
      'nbi_particle_source',
      'gas_puff_source',
      'pellet_source',
      # Ion and electron heat sources (for the temp-ion and temp-el eqs).
      'generic_ion_el_heat_source',
      'fusion_heat_source',
      'qei_source',
      # Ohmic heat source
      'ohmic_heat_source',
      'bremsstrahlung_heat_sink',
  ]
  # pylint: disable=missing-kwoa
  # pytype: disable=missing-parameter
  source_models_builder = source_models_lib.SourceModelsBuilder(
      source_builders={
          name: get_source_builder_type(name)(
              runtime_params=get_default_runtime_params(name)
          )
          for name in names
      }
  )
  # pylint: enable=missing-kwoa
  # pytype: enable=missing-parameter
  return source_models_builder


def get_source_builder_type(source_name: str) -> Any:
  """Returns a builder type for the given source."""
  match source_name:
    case 'j_bootstrap':
      return bootstrap_current_source.BootstrapCurrentSourceBuilder
    case 'jext':
      return external_current_source.ExternalCurrentSourceBuilder

    case 'nbi_particle_source':
      return electron_density_sources.NBIParticleSourceBuilder

    case 'gas_puff_source':
      return electron_density_sources.GasPuffSourceBuilder

    case 'pellet_source':
      return electron_density_sources.PelletSourceBuilder

    case 'generic_ion_el_heat_source':
      return ion_el_heat.GenericIonElectronHeatSourceBuilder

    case 'fusion_heat_source':
      return fusion_heat_source.FusionHeatSourceBuilder

    case 'qei_source':
      return qei_source.QeiSourceBuilder

    case 'ohmic_heat_source':
      return ohmic_heat_source.OhmicHeatSourceBuilder

    case 'bremsstrahlung_heat_sink':
      return bremsstrahlung_heat_sink.BremsstrahlungHeatSinkBuilder

    case _:
      raise ValueError(f'Unknown source name: {source_name}')

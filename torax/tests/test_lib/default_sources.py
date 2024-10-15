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

"""Utilities to help with testing sources."""
from torax.sources import register_source
from torax.sources import source_models as source_models_lib


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
  source_builders = {}
  for name in names:
    registered_source = register_source.get_registered_source(name)
    runtime_params = registered_source.default_runtime_params_class()
    source_builders[name] = registered_source.source_builder_class(
        runtime_params=runtime_params
    )
  return source_models_lib.SourceModelsBuilder(source_builders)

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
from torax.sources import pydantic_model as source_pydantic_model


def get_default_sources() -> source_pydantic_model.Sources:
  """Returns default sources and configurations.

  This set of sources and params are used by most of the TORAX test
  configurations, including ITER-inspired configs, with additional changes to
  their runtime configurations on top.

  If you plan to use them, please remember to update the default configuration
  as needed. Here is an example of how to do so:

  .. code-block:: python

    default_sources = get_default_sources()
    sources_dict = default_sources.to_dict()
    sources_dict = sources_dict['source_model_config']
    sources_dict['qei_source']['Qei_mult'] = 0.0
    sources_dict['generic_ion_el_heat_source']['Ptot'] = 0.0
    sources_dict['fusion_heat_source']['mode'] = source_runtime_params.Mode.ZERO
    sources_dict['ohmic_heat_source']['mode'] = source_runtime_params.Mode.ZERO
    default_sources = sources_pydantic_model.Sources.from_dict(sources_dict)

  More examples are located in the test config files under
  `torax/tests/test_data`.
  """
  names = [
      # Current sources (for psi equation)
      'j_bootstrap',
      'generic_current_source',
      # Electron density sources/sink (for the ne equation).
      'generic_particle_source',
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
  sources = {}
  for name in names:
    sources[name] = {}
  return source_pydantic_model.Sources.from_dict(sources)

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

"""Gets a set of default sources and configurations."""
from typing import Any


def get_default_source_config() -> dict[str, Any]:
  """Returns default sources and configurations.

  This set of sources and params are used by most of the TORAX test
  configurations, including ITER-inspired configs, with additional changes to
  their runtime configurations on top.

  To use, load the dict, update as needed, and then convert to a Sources object:
  .. code-block:: python

    sources_dict = get_default_source_config()
    sources_dict = sources_dict['source_model_config']
    sources_dict['ei_exchange']['Qei_multiplier'] = 0.0
    sources_dict['generic_heat']['P_total'] = 0.0
    sources_dict['fusion']['mode'] = source_runtime_params.Mode.ZERO
    sources_dict['ohmic']['mode'] = source_runtime_params.Mode.ZERO
    default_sources = sources_pydantic_model.Sources.from_dict(sources_dict)
  """
  names = [
      # Current sources (for psi equation)
      'j_bootstrap',
      'generic_current',
      # Electron density sources/sink (for the n_e equation).
      'generic_particle',
      'gas_puff',
      'pellet',
      # Ion and electron heat sources (for the temp-ion and temp-el eqs).
      'generic_heat',
      'fusion',
      'ei_exchange',
      # Ohmic heat source
      'ohmic',
      'bremsstrahlung',
  ]
  return {name: {} for name in names}

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

from typing import Any, Dict
from torax.sources import source_models as source_models_lib


def get_default_source_config() -> Dict[str, Any]:
  """Returns a config dict containing default parameters for SourcesModelBuilder."""

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
  ]

  defaults = {
      name: source_models_lib.get_default_runtime_params(name) for name in names
  }

  return defaults


def get_default_sources_builder() -> source_models_lib.SourceModelsBuilder:
  """Returns a Builder for the default sources."""
  return source_models_lib.SourceModelsBuilder(get_default_source_config())


def get_default_sources() -> source_models_lib.SourceModels:
  """Returns a SourceModels containing default sources and runtime parameters.

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
  return source_models_lib.SourceModelsBuilder(get_default_source_config())()

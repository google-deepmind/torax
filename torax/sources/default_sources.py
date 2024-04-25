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

from torax.sources import bootstrap_current_source
from torax.sources import electron_density_sources
from torax.sources import external_current_source
from torax.sources import fusion_heat_source
from torax.sources import generic_ion_el_heat_source as ion_el_heat
from torax.sources import qei_source
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source_models as source_models_lib


def get_default_sources() -> source_models_lib.SourceModels:
  """Returns a SourceModels containing default sources and runtime parameters.

  This set of sources and params are used by most of the TORAX test
  configurations, including ITER-inpired configs, with additional changes to
  their runtime configurations on top.

  If you plan to use them, please remember to update the default runtime
  parameters as needed. Here is an example of how to do so:

  ```python
  default_sources: SourceModels = get_default_sources()
  # Turn off bootstrap current.
  default_sources.j_bootstrap.runtime_params.mode = runtime_params.Mode.ZERO
  # Change the Qei ion-electron heat exchange term.
  default_sources.qei_source.runtime_params.Qei_mult = 2.0
  # Turn off fusion power.
  default_sources.sources['fusion_heat_source'].runtime_params.mode = (
      runtime_params.Mode.ZERO
  )
  ```

  More examples are located in the test config files under
  `torax/tests/test_data`.
  """
  source_models = source_models_lib.SourceModels(
      sources={
          # Current sources (for psi equation)
          'j_bootstrap': bootstrap_current_source.BootstrapCurrentSource(
              runtime_params=bootstrap_current_source.RuntimeParams(
                  mode=runtime_params_lib.Mode.MODEL_BASED,
              ),
          ),
          'jext': external_current_source.ExternalCurrentSource(
              runtime_params=external_current_source.RuntimeParams(
                  mode=runtime_params_lib.Mode.FORMULA_BASED,
              ),
          ),
          # Electron density sources/sink (for the ne equation).
          'nbi_particle_source': electron_density_sources.NBIParticleSource(
              runtime_params=electron_density_sources.NBIParticleRuntimeParams(
                  mode=runtime_params_lib.Mode.FORMULA_BASED,
              ),
          ),
          'gas_puff_source': electron_density_sources.GasPuffSource(
              runtime_params=electron_density_sources.GasPuffRuntimeParams(
                  mode=runtime_params_lib.Mode.FORMULA_BASED,
              ),
          ),
          'pellet_source': electron_density_sources.PelletSource(
              runtime_params=electron_density_sources.PelletRuntimeParams(
                  mode=runtime_params_lib.Mode.FORMULA_BASED,
              ),
          ),
          # Ion and electron heat sources (for the temp-ion and temp-el eqs).
          'generic_ion_el_heat_source': (
              ion_el_heat.GenericIonElectronHeatSource(
                  runtime_params=ion_el_heat.RuntimeParams(
                      mode=runtime_params_lib.Mode.FORMULA_BASED,
                  ),
              )
          ),
          'fusion_heat_source': fusion_heat_source.FusionHeatSource(
              runtime_params=runtime_params_lib.RuntimeParams(
                  mode=runtime_params_lib.Mode.MODEL_BASED,
              ),
          ),
          'qei_source': qei_source.QeiSource(
              runtime_params=qei_source.RuntimeParams(
                  mode=runtime_params_lib.Mode.MODEL_BASED,
              ),
          ),
      }
  )
  # Add OhmicHeatSource after because it requires a pointer to the SourceModels.
  source_models.add_source(
      source_name='ohmic_heat_source',
      source=source_models_lib.OhmicHeatSource(
          source_models=source_models,
          runtime_params=runtime_params_lib.RuntimeParams(
              mode=runtime_params_lib.Mode.MODEL_BASED,
          ),
      ),
  )
  return source_models

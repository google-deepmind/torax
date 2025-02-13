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
from absl.testing import absltest
from absl.testing import parameterized
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
from torax.sources import register_source
from torax.sources import source as source_lib


class SourceTest(parameterized.TestCase):
  """Tests for the source registry."""

  @parameterized.parameters(
      bootstrap_current_source.BootstrapCurrentSource.SOURCE_NAME,
      bremsstrahlung_heat_sink.BremsstrahlungHeatSink.SOURCE_NAME,
      electron_cyclotron_source.ElectronCyclotronSource.SOURCE_NAME,
      electron_density_sources.GenericParticleSource.SOURCE_NAME,
      electron_density_sources.GasPuffSource.SOURCE_NAME,
      electron_density_sources.PelletSource.SOURCE_NAME,
      fusion_heat_source.FusionHeatSource.SOURCE_NAME,
      generic_current_source.GenericCurrentSource.SOURCE_NAME,
      ion_el_heat.GenericIonElectronHeatSource.SOURCE_NAME,
      ohmic_heat_source.OhmicHeatSource.SOURCE_NAME,
      qei_source.QeiSource.SOURCE_NAME,
  )
  def test_sources_in_registry_build_successfully(self, source_name: str):
    """Test that all sources in the registry build successfully."""
    registered_source = register_source.get_supported_source(source_name)
    source_class = registered_source.source_class
    model_function = registered_source.model_functions[
        source_class.DEFAULT_MODEL_FUNCTION_NAME
    ]
    source_builder_class = model_function.source_builder_class
    source_runtime_params_class = model_function.runtime_params_class
    if source_builder_class is None:
      source_builder_class = source_lib.make_source_builder(
          registered_source.source_class,
          runtime_params_type=source_runtime_params_class,
          model_func=model_function.source_profile_function,
      )
    source_runtime_params_class = model_function.runtime_params_class
    source_builder = source_builder_class()
    self.assertIsInstance(
        source_builder.runtime_params, source_runtime_params_class
    )
    source = source_builder()
    self.assertIsInstance(source, source_class)


if __name__ == "__main__":
  absltest.main()

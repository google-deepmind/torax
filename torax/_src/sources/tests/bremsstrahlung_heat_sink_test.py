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
from typing import Callable

from absl.testing import absltest
from absl.testing import parameterized
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.sources import bremsstrahlung_heat_sink
from torax._src.sources import source_models as source_models_lib
from torax._src.sources.tests import test_lib
from torax._src.test_utils import torax_refs


# pylint: disable=invalid-name


class BremsstrahlungHeatSinkTest(test_lib.SingleProfileSourceTestCase):
  """Tests for BremsstrahlungHeatSink."""

  def setUp(self):
    super().setUp(
        source_config_class=bremsstrahlung_heat_sink.BremsstrahlungHeatSinkConfig,
        source_name=bremsstrahlung_heat_sink.BremsstrahlungHeatSink.SOURCE_NAME,
    )

  @parameterized.parameters([
      dict(references_getter=torax_refs.circular_references),
      dict(references_getter=torax_refs.chease_references_Ip_from_chease),
      dict(
          references_getter=torax_refs.chease_references_Ip_from_runtime_params
      ),
  ])
  def test_compare_against_known(
      self, references_getter: Callable[[], torax_refs.References]
  ):
    references = references_getter()

    dynamic_runtime_params_slice, geo = references.get_dynamic_slice_and_geo()
    static_runtime_params_slice = (
        build_runtime_params.build_static_params_from_config(references.config)
    )
    source_models = source_models_lib.SourceModels(
        sources=references.config.sources,
        neoclassical=references.config.neoclassical,
    )
    core_profiles = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_runtime_params_slice,
        geo=geo,
        source_models=source_models,
    )

    P_brem_total, P_brems_profile = (
        bremsstrahlung_heat_sink.calc_bremsstrahlung(
            core_profiles,
            geo,
            dynamic_runtime_params_slice.plasma_composition.Z_eff_face,
            dynamic_runtime_params_slice.numerics.density_reference,
        )
    )

    self.assertIsNotNone(P_brem_total)
    self.assertIsNotNone(P_brems_profile)

    P_brem_total_stott, P_brems_profile_stott = (
        bremsstrahlung_heat_sink.calc_bremsstrahlung(
            core_profiles,
            geo,
            dynamic_runtime_params_slice.plasma_composition.Z_eff_face,
            dynamic_runtime_params_slice.numerics.density_reference,
            use_relativistic_correction=True,
        )
    )

    self.assertIsNotNone(P_brem_total_stott)
    self.assertIsNotNone(P_brems_profile_stott)

    # Expect the relativistic correction to increase the total power.
    self.assertGreater(P_brem_total_stott, P_brem_total)


if __name__ == '__main__':
  absltest.main()

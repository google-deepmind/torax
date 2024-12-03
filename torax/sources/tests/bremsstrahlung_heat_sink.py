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

# pylint: disable=invalid-name

"""Tests for radiation_source."""

from typing import Callable

from absl.testing import absltest
from absl.testing import parameterized
from torax import core_profile_setters
from torax.sources import bremsstrahlung_heat_sink
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source_models as source_models_lib
from torax.sources.tests import test_lib
from torax.tests.test_lib import torax_refs


class BremsstrahlungHeatSinkTest(test_lib.SingleProfileSourceTestCase):
  """Tests for BremsstrahlungHeatSink."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass(
        source_class=bremsstrahlung_heat_sink.BremsstrahlungHeatSink,
        runtime_params_class=bremsstrahlung_heat_sink.RuntimeParams,
        unsupported_modes=[
            runtime_params_lib.Mode.FORMULA_BASED,
        ],
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

    runtime_params = references.runtime_params
    geo_provider = references.geometry_provider

    source_models_builder = source_models_lib.SourceModelsBuilder()
    dynamic_runtime_params_slice, geo = (
        torax_refs.build_consistent_dynamic_runtime_params_slice_and_geometry(
            runtime_params,
            geo_provider,
            sources=source_models_builder.runtime_params,
        )
    )
    source_models = source_models_builder()
    core_profiles = core_profile_setters.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        source_models=source_models,
    )

    P_brem_total, P_brems_profile = (
        bremsstrahlung_heat_sink.calc_bremsstrahlung(
            core_profiles,
            geo,
            dynamic_runtime_params_slice.plasma_composition.Zeff_face,
            dynamic_runtime_params_slice.numerics.nref,
        )
    )

    self.assertIsNotNone(P_brem_total)
    self.assertIsNotNone(P_brems_profile)

    P_brem_total_stott, P_brems_profile_stott = (
        bremsstrahlung_heat_sink.calc_bremsstrahlung(
            core_profiles,
            geo,
            dynamic_runtime_params_slice.plasma_composition.Zeff_face,
            dynamic_runtime_params_slice.numerics.nref,
            use_relativistic_correction=True,
        )
    )

    self.assertIsNotNone(P_brem_total_stott)
    self.assertIsNotNone(P_brems_profile_stott)

    # Expect the relativistic correction to increase the total power.
    self.assertGreater(P_brem_total_stott, P_brem_total)


if __name__ == '__main__':
  absltest.main()

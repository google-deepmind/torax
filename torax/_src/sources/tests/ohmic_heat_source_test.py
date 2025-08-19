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
from unittest import mock

from absl.testing import absltest
from torax._src.config import runtime_params_slice
from torax._src.sources import ohmic_heat_source
from torax._src.sources.tests import test_lib
from torax._src.torax_pydantic import torax_pydantic


class OhmicHeatSourceTest(test_lib.SingleProfileSourceTestCase):
  """Tests for OhmicHeatSource."""

  def setUp(self):
    super().setUp(
        source_config_class=ohmic_heat_source.OhmicHeatSourceConfig,
        source_name=ohmic_heat_source.OhmicHeatSource.SOURCE_NAME,
        needs_source_models=True,
    )

  def test_raises_error_if_calculated_source_profiles_is_none(self):
    source = ohmic_heat_source.OhmicHeatSource(
        model_func=ohmic_heat_source.ohmic_model_func
    )
    source_config = self._source_config_class.from_dict({})
    torax_pydantic.set_grid(source_config, torax_pydantic.Grid1D(nx=4,))
    dynamic_runtime_params_slice = mock.create_autospec(
        runtime_params_slice.DynamicRuntimeParamsSlice,
        instance=True,
        sources={
            self._source_name: source_config.build_dynamic_params(t=0.0)
        },
    )
    with self.assertRaisesRegex(
        ValueError,
        'calculated_source_profiles is a required argument for'
        ' ohmic_model_func. This can occur if this source function is used in'
        ' an explicit source.',
    ):
      source.get_value(
          mock.ANY,
          dynamic_runtime_params_slice,
          mock.ANY,
          mock.ANY,
          calculated_source_profiles=None,
          conductivity=mock.ANY,
      )

  def test_raises_error_if_conductivity_is_none(self):
    source = ohmic_heat_source.OhmicHeatSource(
        model_func=ohmic_heat_source.ohmic_model_func
    )
    source_config = self._source_config_class.from_dict({})
    torax_pydantic.set_grid(source_config, torax_pydantic.Grid1D(nx=4,))
    dynamic_runtime_params_slice = mock.create_autospec(
        runtime_params_slice.DynamicRuntimeParamsSlice,
        instance=True,
        sources={
            self._source_name: source_config.build_dynamic_params(t=0.0)
        },
    )
    with self.assertRaisesRegex(
        ValueError,
        'conductivity is a required argument for'
        ' ohmic_model_func. This can occur if this source function is used in'
        ' an explicit source.',
    ):
      source.get_value(
          mock.ANY,
          dynamic_runtime_params_slice,
          mock.ANY,
          mock.ANY,
          calculated_source_profiles=mock.ANY,
          conductivity=None,
      )


if __name__ == '__main__':
  absltest.main()

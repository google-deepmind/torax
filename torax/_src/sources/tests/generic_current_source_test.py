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
from torax._src.sources import generic_current_source
from torax._src.sources.tests import test_lib


class GenericCurrentSourceTest(test_lib.SingleProfileSourceTestCase):
  """Tests for GenericCurrentSource."""

  source_config_class = generic_current_source.GenericCurrentSourceConfig
  source_name = generic_current_source.GenericCurrentSource.SOURCE_ID
  extra_source_config_kwargs = {
      'fraction_of_total_current': 0.2,
      'gaussian_width': 0.05,
      'gaussian_location': 0.4,
  }

  @parameterized.named_parameters(
      dict(
          testcase_name='relative_warns_on_i_generic',
          use_absolute_current=False,
          expected_warning=(
              'I_generic is set but will be ignored because'
              ' use_absolute_current is False'
          ),
      ),
      dict(
          testcase_name='absolute_warns_on_fraction_of_total_current',
          use_absolute_current=True,
          expected_warning=(
              'fraction_of_total_current is set but will be ignored because'
              ' use_absolute_current is True'
          ),
      ),
  )
  def test_warns_when_unused_amplitude_is_set(
      self,
      use_absolute_current: bool,
      expected_warning: str,
  ):
    with self.assertLogs(level='WARNING') as logs:
      generic_current_source.GenericCurrentSourceConfig.from_dict({
          'use_absolute_current': use_absolute_current,
          'fraction_of_total_current': 0.2,
          'I_generic': 3.0e6,
          'gaussian_width': 0.05,
          'gaussian_location': 0.4,
      })
    self.assertTrue(any(expected_warning in msg for msg in logs.output))


class GenericCurrentSourceAbsoluteTest(test_lib.SingleProfileSourceTestCase):
  """Tests for GenericCurrentSource with absolute current."""

  source_config_class = generic_current_source.GenericCurrentSourceConfig
  source_name = generic_current_source.GenericCurrentSource.SOURCE_ID
  extra_source_config_kwargs = {
      'use_absolute_current': True,
      'I_generic': 3.0e6,
      'gaussian_width': 0.05,
      'gaussian_location': 0.4,
  }


if __name__ == '__main__':
  absltest.main()

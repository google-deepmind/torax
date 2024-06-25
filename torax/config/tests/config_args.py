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

"""Tests for torax.config.config_args."""

import os

from absl.testing import absltest
from absl.testing import parameterized
from torax.config import config_args
import xarray as xr


class ConfigArgsTest(parameterized.TestCase):
  """Unit tests for the `torax.config.config_args` module."""

  def setUp(self):
    super().setUp()
    src_dir = absltest.TEST_SRCDIR.value
    torax_dir = 'torax/'
    test_data_dir = os.path.join(src_dir, torax_dir, "tests/test_data")
    self.ds = xr.load_dataset(
        os.path.join(test_data_dir, "test_iterhybrid_rampup.nc")
    )

  def test_load_time_interpolated_array_smoke(self):
    config_args.load_time_interpolated_array(self.ds, "temp_ion")

  def test_load_time_interpolated_scalar_smoke(self):
    config_args.load_time_interpolated_scalar(self.ds, "temp_el_right_bc")


if __name__ == "__main__":
  absltest.main()

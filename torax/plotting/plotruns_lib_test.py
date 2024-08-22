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

"""Unit tests for torax.plotting.plotruns_lib."""

import os
from absl.testing import absltest
from torax.plotting import plotruns_lib
from torax.tests.test_lib import paths


class PlotrunsLibTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    self.test_data_dir = paths.test_data_dir()

  def test_data_loading(self):
    plotruns_lib.load_data(
        os.path.join(self.test_data_dir, "test_iterhybrid_rampup.nc")
    )


if __name__ == "__main__":
  absltest.main()

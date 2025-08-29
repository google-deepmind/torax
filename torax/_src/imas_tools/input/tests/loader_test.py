# Copyright 2025 DeepMind Technologies Limited
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


import os

from absl.testing import absltest
from absl.testing import parameterized
from imas import ids_toplevel
from torax._src.config import config_loader
from torax._src.imas_tools.input import loader
from torax._src.test_utils import sim_test_case


class IMASLoaderTest(sim_test_case.SimTestCase):
  """Unit tests for torax.torax_imastools.input.core_profiles.py"""

  @parameterized.parameters([
      dict(
          ids_name=loader.CORE_PROFILES,
          path="core_profiles_ddv4_iterhybrid_rampup_conditions.nc",
      ),
      dict(
          ids_name=loader.EQUILIBRIUM,
          path="ITERhybrid_COCOS17_IDS_ddv4.nc",
      ),
  ])
  def test_load_imas_from_netCDF(
      self,
      ids_name,
      path,
  ):
    """Unit tests to check the IMAS loader can load an IMAS IDS from a netCDF
    file."""
    dir = os.path.join(config_loader.torax_path(), "data/third_party/imas_data")
    ids_in = loader.load_imas_data(path, ids_name, dir)
    assert isinstance(ids_in, ids_toplevel.IDSToplevel)


if __name__ == "__main__":
  absltest.main()

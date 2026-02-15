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
from absl.testing import absltest
import numpy as np
from torax._src.imas_tools.input import loader
from torax._src.imas_tools.input import validation


# pylint: disable=invalid-name
class IMASLoaderTest(absltest.TestCase):

  def test_validate_profile_conditions_from_IMAS_raises_on_missing_quantities(
      self,
  ):
    path = "core_profiles_ddv4_iterhybrid_rampup_conditions.nc"
    ids_name = "core_profiles"
    ids_in = loader.load_imas_data(path, ids_name)
    for profile in ids_in.profiles_1d:
      profile.grid.rho_tor_norm = np.array([])
    with self.assertRaises(ValueError):
      validation.validate_profile_conditions_from_IMAS(ids_in)

  def test_validate_profile_conditions_from_IMAS_warns(self):
    path = "core_profiles_ddv4_iterhybrid_rampup_conditions.nc"
    ids_name = "core_profiles"
    ids_in = loader.load_imas_data(path, ids_name)
    self.assertFalse(ids_in.global_quantities.v_loop.has_value)
    with self.assertLogs(level="WARNING") as logs:
      validation.validate_profile_conditions_from_IMAS(ids_in)
    self.assertIn("The IDS is missing the global_quantities.v_loop quantity.",
                  logs.output[0])

  def test_validate_plasma_composition_from_IMAS_raises_on_empty_ions(self):
    path = "core_profiles_ddv4_iterhybrid_rampup_conditions.nc"
    ids_name = "core_profiles"
    ids_in = loader.load_imas_data(path, ids_name)
    # The input IDS has no ion information.
    with self.assertRaises(ValueError):
      validation.validate_plasma_composition_from_IMAS(ids_in)


if __name__ == "__main__":
  absltest.main()

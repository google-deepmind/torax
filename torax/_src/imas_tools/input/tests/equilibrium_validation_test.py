# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Required geometry-field checks for IMAS equilibrium input."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax._src.imas_tools.input import equilibrium as imas_equilibrium
from torax._src.imas_tools.input import loader
from torax._src.imas_tools.input import validation


class EquilibriumGeometryValidationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.equilibrium = loader.load_imas_data(
        'ITERhybrid_COCOS17_IDS_ddv4.nc', 'equilibrium'
    )

  def test_valid_equilibrium_is_accepted(self):
    validation.validate_equilibrium_geometry_from_IMAS(self.equilibrium)

  @parameterized.parameters('gm7', 'gm3', 'f', 'r_inboard', 'gm4', 'phi')
  def test_missing_required_profile_has_field_specific_error(self, field):
    setattr(
        self.equilibrium.time_slice[0].profiles_1d,
        field,
        np.array([]),
    )
    with self.assertRaisesRegex(ValueError, f'profiles_1d.{field}'):
      validation.validate_equilibrium_geometry_from_IMAS(self.equilibrium)

  def test_missing_profile_fails_before_geometry_arithmetic(self):
    self.equilibrium.time_slice[0].profiles_1d.gm7 = np.array([])
    with self.assertRaisesRegex(ValueError, 'profiles_1d.gm7'):
      imas_equilibrium._geometry_from_single_slice(
          self.equilibrium, np.linspace(0.0, 1.0, 12)
      )

  def test_missing_volume_is_reported_when_derivative_is_absent(self):
    profiles = self.equilibrium.time_slice[0].profiles_1d
    profiles.dvolume_dpsi = np.array([])
    profiles.volume = np.array([])
    with self.assertRaisesRegex(ValueError, 'profiles_1d.volume'):
      validation.validate_equilibrium_geometry_from_IMAS(self.equilibrium)

  def test_optional_derivative_can_use_existing_volume(self):
    profiles = self.equilibrium.time_slice[0].profiles_1d
    self.assertTrue(profiles.volume.has_value)
    profiles.dvolume_dpsi = np.array([])
    validation.validate_equilibrium_geometry_from_IMAS(self.equilibrium)

  def test_optional_gm9_is_not_required(self):
    self.equilibrium.time_slice[0].profiles_1d.gm9 = np.array([])
    validation.validate_equilibrium_geometry_from_IMAS(self.equilibrium)

  def test_validation_selects_requested_slice(self):
    equilibrium = loader.load_imas_data(
        'ITERhybrid_rampup_11_time_slices_COCOS17_IDS_ddv4.nc',
        'equilibrium',
    )
    validation.validate_equilibrium_geometry_from_IMAS(equilibrium, 0)
    equilibrium.time_slice[5].profiles_1d.gm7 = np.array([])
    with self.assertRaisesRegex(ValueError, 'profiles_1d.gm7'):
      validation.validate_equilibrium_geometry_from_IMAS(equilibrium, 5)


if __name__ == '__main__':
  absltest.main()

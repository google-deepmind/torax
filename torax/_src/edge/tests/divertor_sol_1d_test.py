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
import numpy as np
from torax._src.edge import divertor_sol_1d
from torax._src.edge import extended_lengyel_defaults

# pylint: disable=invalid-name

_RTOL = 1e-6


class DivertorSOL1DTest(parameterized.TestCase):
  """Testing the DivertorSOL1D class properties.

  All reference values taken from the second inner loop of the reference case in
  https://github.com/cfs-energy/extended-lengyel
  """

  def setUp(self):
    super().setUp()
    self.divertor_sol_1d = divertor_sol_1d.DivertorSOL1D(
        q_parallel=5.06193577e8,
        divertor_Z_eff=2.291360670810858,
        target_electron_temp=2.34,
        SOL_conduction_fraction=extended_lengyel_defaults.SOL_CONDUCTION_FRACTION,
        divertor_broadening_factor=extended_lengyel_defaults.DIVERTOR_BROADENING_FACTOR,
        divertor_parallel_length=5.0,
        parallel_connection_length=20.0,
        separatrix_mach_number=extended_lengyel_defaults.SEPARATRIX_MACH_NUMBER,
        separatrix_electron_density=3.3e19,
        separatrix_ratio_of_ion_to_electron_temp=extended_lengyel_defaults.SEPARATRIX_RATIO_ION_TO_ELECTRON_TEMP,
        separatrix_ratio_of_electron_to_ion_density=extended_lengyel_defaults.SEPARATRIX_RATIO_ELECTRON_TO_ION_DENSITY,
        average_ion_mass=2.0,
        sheath_heat_transmission_factor=extended_lengyel_defaults.SHEATH_HEAT_TRANSMISSION_FACTOR,
        target_mach_number=extended_lengyel_defaults.TARGET_MACH_NUMBER,
        target_ratio_of_ion_to_electron_temp=extended_lengyel_defaults.TARGET_RATIO_ION_TO_ELECTRON_TEMP,
        target_ratio_of_electron_to_ion_density=extended_lengyel_defaults.TARGET_RATIO_ELECTRON_TO_ION_DENSITY,
        toroidal_flux_expansion=extended_lengyel_defaults.TOROIDAL_FLUX_EXPANSION,
    )

  def test_kappa_e(self):
    expected_value = 1751.6010938527386
    np.testing.assert_allclose(
        self.divertor_sol_1d.kappa_e,
        expected_value,
        rtol=_RTOL,
    )

  def test_electron_temp_at_cc_interface(self):
    expected_value = 6.167578954082415
    np.testing.assert_allclose(
        self.divertor_sol_1d.electron_temp_at_cc_interface,
        expected_value,
        rtol=_RTOL,
    )

  def test_divertor_entrance_electron_temp(self):
    expected_value = 60.13510639676952
    np.testing.assert_allclose(
        self.divertor_sol_1d.divertor_entrance_electron_temp,
        expected_value,
        rtol=_RTOL,
    )

  def test_separatrix_electron_temp(self):
    expected_value = 116.09239718114556
    np.testing.assert_allclose(
        self.divertor_sol_1d.separatrix_electron_temp,
        expected_value,
        rtol=_RTOL,
    )

  def test_required_power_loss(self):
    expected_value = 0.9662169852278072
    np.testing.assert_allclose(
        self.divertor_sol_1d.required_power_loss,
        expected_value,
        rtol=_RTOL,
    )


if __name__ == '__main__':
  absltest.main()

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


class DivertorSOL1DInverseModeTest(parameterized.TestCase):
  """Testing the DivertorSOL1D class properties.

  All reference values taken from the second loop of the reference case in
  https://github.com/cfs-energy/extended-lengyel
  """

  def setUp(self):
    super().setUp()
    self._RTOL = 1e-6
    self.divertor_sol_1d = divertor_sol_1d.DivertorSOL1D(
        q_parallel=3.39611623e8,
        c_z_prefactor=0.059314229517142096,
        kappa_e=1751.6010938527386,
        alpha_t=0.0,
        main_ion_charge=1.0,
        seed_impurity_weights={'N': 1.0, 'Ar': 0.05},
        fixed_impurity_concentrations={'He': 0.01},
        ne_tau=extended_lengyel_defaults.NE_TAU,
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

  def test_electron_temp_at_cc_interface(self):
    expected_value = 6.167578954082415
    np.testing.assert_allclose(
        self.divertor_sol_1d.electron_temp_at_cc_interface,
        expected_value,
        rtol=self._RTOL,
    )

  def test_divertor_entrance_electron_temp(self):
    expected_value = 53.65683926205252
    np.testing.assert_allclose(
        self.divertor_sol_1d.divertor_entrance_electron_temp,
        expected_value,
        rtol=self._RTOL,
    )

  def test_separatrix_electron_temp(self):
    expected_value = 103.58141942945846
    np.testing.assert_allclose(
        self.divertor_sol_1d.separatrix_electron_temp,
        expected_value,
        rtol=self._RTOL,
    )

  def test_required_power_loss(self):
    expected_value = 0.9550726743297632
    np.testing.assert_allclose(
        self.divertor_sol_1d.required_power_loss,
        expected_value,
        rtol=self._RTOL,
    )

  def test_divertor_Z_eff(self):
    expected_value = 2.2881883214145797
    np.testing.assert_allclose(
        self.divertor_sol_1d.divertor_Z_eff,
        expected_value,
        rtol=self._RTOL,
    )


class DivertorSOL1DForwardModeTest(parameterized.TestCase):
  """Testing the DivertorSOL1D class properties.

  All reference values taken from the second loop of the reference case in
  https://github.com/cfs-energy/extended-lengyel
  """

  def setUp(self):
    super().setUp()
    self._RTOL = 5e-5
    self.divertor_sol_1d = divertor_sol_1d.DivertorSOL1D(
        q_parallel=5.061935771095335e8,
        c_z_prefactor=0.0,
        kappa_e=1931.8277173925928,
        alpha_t=0.0,
        seed_impurity_weights={},
        fixed_impurity_concentrations={
            'He': 0.01,
            'N': 0.038397305226362526,
            'Ar': 0.0019198652613181264,
        },
        main_ion_charge=1.0,
        ne_tau=extended_lengyel_defaults.NE_TAU,
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

  def test_calc_target_electron_temp(self):
    previous_target_electron_temp = 2.0
    expected_value = 10.181228774214071
    parallel_heat_flux_at_cc_interface = 1.1088918473707701e8
    np.testing.assert_allclose(
        divertor_sol_1d.calc_target_electron_temp(
            self.divertor_sol_1d,
            parallel_heat_flux_at_cc_interface,
            previous_target_electron_temp,
        ),
        expected_value,
        rtol=self._RTOL,
    )


if __name__ == '__main__':
  absltest.main()

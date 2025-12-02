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
    params = divertor_sol_1d.ExtendedLengyelParameters(
        # Dummy values for unused parameters in these specific tests.
        major_radius=1.0,
        minor_radius=1.0,
        separatrix_average_poloidal_field=1.0,
        fieldline_pitch_at_omp=1.0,
        cylindrical_safety_factor=1.0,
        power_crossing_separatrix=1.0,
        ratio_bpol_omp_to_bpol_avg=extended_lengyel_defaults.RATIO_BPOL_OMP_TO_BPOL_AVG,
        fraction_of_P_SOL_to_divertor=extended_lengyel_defaults.FRACTION_OF_PSOL_TO_DIVERTOR,
        angle_of_incidence_target=extended_lengyel_defaults.ANGLE_OF_INCIDENCE_TARGET,
        T_wall=extended_lengyel_defaults.T_WALL,
        ratio_of_molecular_to_ion_mass=extended_lengyel_defaults.RATIO_MOLECULAR_TO_ION_MASS,
        # Parameters from reference case
        main_ion_charge=1.0,
        mean_ion_charge_state=1.0,
        seed_impurity_weights={'N': 1.0, 'Ar': 0.05},
        fixed_impurity_concentrations={'He': 0.01},
        ne_tau=extended_lengyel_defaults.NE_TAU,
        SOL_conduction_fraction=extended_lengyel_defaults.SOL_CONDUCTION_FRACTION,
        divertor_broadening_factor=extended_lengyel_defaults.DIVERTOR_BROADENING_FACTOR,
        connection_length_divertor=5.0,
        connection_length_target=20.0,
        mach_separatrix=extended_lengyel_defaults.MACH_SEPARATRIX,
        separatrix_electron_density=3.3e19,
        T_i_T_e_ratio_separatrix=extended_lengyel_defaults.T_I_T_E_RATIO_SEPARATRIX,
        n_e_n_i_ratio_separatrix=extended_lengyel_defaults.N_E_N_I_RATIO_SEPARATRIX,
        average_ion_mass=2.0,
        sheath_heat_transmission_factor=extended_lengyel_defaults.SHEATH_HEAT_TRANSMISSION_FACTOR,
        mach_target=extended_lengyel_defaults.MACH_TARGET,
        T_i_T_e_ratio_target=extended_lengyel_defaults.T_I_T_E_RATIO_TARGET,
        n_e_n_i_ratio_target=extended_lengyel_defaults.N_E_N_I_RATIO_TARGET,
        toroidal_flux_expansion=extended_lengyel_defaults.TOROIDAL_FLUX_EXPANSION,
    )
    state = divertor_sol_1d.ExtendedLengyelState(
        q_parallel=3.39611623e8,
        c_z_prefactor=0.059314229517142096,
        kappa_e=1751.6010938527386,
        alpha_t=0.0,
        T_e_target=2.34,
    )
    self.divertor_sol_1d = divertor_sol_1d.DivertorSOL1D(
        params=params,
        state=state,
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
    params = divertor_sol_1d.ExtendedLengyelParameters(
        separatrix_average_poloidal_field=0.28506577,
        fieldline_pitch_at_omp=5.14589459864493,
        power_crossing_separatrix=5.5e6,
        minor_radius=0.5,
        major_radius=1.65,
        cylindrical_safety_factor=3.7290303009853,
        ratio_bpol_omp_to_bpol_avg=extended_lengyel_defaults.RATIO_BPOL_OMP_TO_BPOL_AVG,
        fraction_of_P_SOL_to_divertor=extended_lengyel_defaults.FRACTION_OF_PSOL_TO_DIVERTOR,
        angle_of_incidence_target=extended_lengyel_defaults.ANGLE_OF_INCIDENCE_TARGET,
        T_wall=extended_lengyel_defaults.T_WALL,
        ratio_of_molecular_to_ion_mass=extended_lengyel_defaults.RATIO_MOLECULAR_TO_ION_MASS,
        # Parameters from reference case
        seed_impurity_weights={},
        fixed_impurity_concentrations={
            'He': 0.01,
            'N': 0.038397305226362526,
            'Ar': 0.0019198652613181264,
        },
        main_ion_charge=1.0,
        mean_ion_charge_state=1.0,
        ne_tau=extended_lengyel_defaults.NE_TAU,
        SOL_conduction_fraction=extended_lengyel_defaults.SOL_CONDUCTION_FRACTION,
        divertor_broadening_factor=extended_lengyel_defaults.DIVERTOR_BROADENING_FACTOR,
        connection_length_divertor=5.0,
        connection_length_target=20.0,
        mach_separatrix=extended_lengyel_defaults.MACH_SEPARATRIX,
        separatrix_electron_density=3.3e19,
        T_i_T_e_ratio_separatrix=extended_lengyel_defaults.T_I_T_E_RATIO_SEPARATRIX,
        n_e_n_i_ratio_separatrix=extended_lengyel_defaults.N_E_N_I_RATIO_SEPARATRIX,
        average_ion_mass=2.0,
        sheath_heat_transmission_factor=extended_lengyel_defaults.SHEATH_HEAT_TRANSMISSION_FACTOR,
        mach_target=extended_lengyel_defaults.MACH_TARGET,
        T_i_T_e_ratio_target=extended_lengyel_defaults.T_I_T_E_RATIO_TARGET,
        n_e_n_i_ratio_target=extended_lengyel_defaults.N_E_N_I_RATIO_TARGET,
        toroidal_flux_expansion=extended_lengyel_defaults.TOROIDAL_FLUX_EXPANSION,
    )
    state = divertor_sol_1d.ExtendedLengyelState(
        q_parallel=5.061935771095335e8,
        c_z_prefactor=0.0,
        kappa_e=1931.8277173925928,
        alpha_t=0.0,
        T_e_target=2.0,
    )
    self.divertor_sol_1d = divertor_sol_1d.DivertorSOL1D(
        params=params,
        state=state,
    )

  def test_calc_T_e_target(self):
    expected_value = 10.181228774214071
    parallel_heat_flux_at_cc_interface = 1.1088918473707701e8
    np.testing.assert_allclose(
        divertor_sol_1d.calc_T_e_target(
            self.divertor_sol_1d,
            parallel_heat_flux_at_cc_interface,
        ),
        expected_value,
        rtol=self._RTOL,
    )

  def test_calc_kappa_e(self):
    expected_value = 1751.6010938527386
    Z_eff = 2.291360670810858
    np.testing.assert_allclose(
        divertor_sol_1d.calc_kappa_e(Z_eff),
        expected_value,
        rtol=1e-5,
    )

  def test_calc_q_parallel(self):
    alpha_t = 0.0
    separatrix_election_temp = 100.0  # [eV]

    # reference value from the first loop of the reference case in
    # https://github.com/cfs-energy/extended-lengyel
    expected_q_parallel = 506.193577e6

    calculated_q_parallel = divertor_sol_1d.calc_q_parallel(
        params=self.divertor_sol_1d.params,
        separatrix_electron_temp=separatrix_election_temp,
        alpha_t=alpha_t,
    )

    np.testing.assert_allclose(
        calculated_q_parallel,
        expected_q_parallel,
        rtol=1e-5,
    )

  def test_calc_alpha_t(self):
    """Test calc_alpha_t against reference values."""

    # Inputs and output from the first loop of the reference case in
    # https://github.com/cfs-energy/extended-lengyel
    separatrix_electron_temp = 106.2293618373  # eV
    separatrix_Z_eff = 2.329589485913357  # dimensionless

    expected_alpha_t = 0.4020393753155751

    calculated_alpha_t = divertor_sol_1d.calc_alpha_t(
        params=self.divertor_sol_1d.params,
        separatrix_electron_temp=separatrix_electron_temp,
        separatrix_Z_eff=separatrix_Z_eff,
    )

    np.testing.assert_allclose(
        calculated_alpha_t,
        expected_alpha_t,
        rtol=1e-5,
    )


if __name__ == '__main__':
  absltest.main()

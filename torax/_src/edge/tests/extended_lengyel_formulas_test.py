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
import numpy as np
from torax._src.edge import extended_lengyel_defaults
from torax._src.edge import extended_lengyel_formulas

# pylint: disable=invalid-name


class ExtendedLengyelFormulasTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.target_electron_temp = 2.34  # [eV]
    self.magnetic_field_on_axis = 2.5  # [T]
    self.plasma_current = 1e6  # [A]
    self.major_radius = 1.65  # [m]
    self.minor_radius = 0.5  # [m]
    self.elongation_psi95 = 1.6  # [dimensionless]
    self.triangularity_psi95 = 0.3  # [dimensionless]

  def test_calc_alpha_t(self):
    """Test calc_alpha_t against reference values."""

    # Inputs and output from the first loop of the reference case in
    # https://github.com/cfs-energy/extended-lengyel
    separatrix_electron_density = 3.3e19  # m^-3
    separatrix_electron_temp = 0.1062293618373  # keV
    cylindrical_safety_factor = 3.7290303009853  # dimensionless
    major_radius = self.major_radius  # m
    average_ion_mass = 2.0  # [amu]
    Z_eff = 2.329589485913357  # dimensionless
    mean_ion_charge_state = 1.0  # elementary charge
    ion_to_electron_temp_ratio = 1.0  # dimensionless

    expected_alpha_t = 0.4020393753155751

    calculated_alpha_t = extended_lengyel_formulas.calc_alpha_t(
        separatrix_electron_density=separatrix_electron_density,
        separatrix_electron_temp=separatrix_electron_temp,
        cylindrical_safety_factor=cylindrical_safety_factor,
        major_radius=major_radius,
        average_ion_mass=average_ion_mass,
        Z_eff=Z_eff,
        mean_ion_charge_state=mean_ion_charge_state,
        ion_to_electron_temp_ratio=ion_to_electron_temp_ratio,
    )

    np.testing.assert_allclose(
        calculated_alpha_t,
        expected_alpha_t,
        rtol=1e-5,
    )

  def test_calc_momentum_loss_in_convection_layer(self):
    # reference value from the first loop of the reference case in
    # https://github.com/cfs-energy/extended-lengyel
    expected_momentum_loss = 0.5364587873343747
    calculated_momentum_loss = (
        extended_lengyel_formulas.calc_momentum_loss_in_convection_layer(
            target_electron_temp=self.target_electron_temp
        )
    )
    np.testing.assert_allclose(
        calculated_momentum_loss,
        expected_momentum_loss,
        rtol=1e-6,
    )

  def test_calc_density_ratio_in_convection_layer(self):
    # reference value from the first loop of the reference case in
    # https://github.com/cfs-energy/extended-lengyel
    expected_density_ratio = 0.6108818435013572
    calculated_density_ratio = (
        extended_lengyel_formulas.calc_density_ratio_in_convection_layer(
            target_electron_temp=self.target_electron_temp
        )
    )
    np.testing.assert_allclose(
        calculated_density_ratio,
        expected_density_ratio,
        rtol=1e-6,
    )

  def test_calc_power_loss_in_convection_layer(self):
    # reference value from the first loop of the reference case in
    # https://github.com/cfs-energy/extended-lengyel
    expected_power_loss = 0.6791789837814304
    calculated_power_loss = (
        extended_lengyel_formulas.calc_power_loss_in_convection_layer(
            target_electron_temp=self.target_electron_temp
        )
    )
    np.testing.assert_allclose(
        calculated_power_loss,
        expected_power_loss,
        rtol=1e-6,
    )

  def test_calc_shaping_factor(self):
    # reference value from the first loop of the reference case in
    # https://github.com/cfs-energy/extended-lengyel
    expected_value = 1.4031849486079875
    calculated_value = extended_lengyel_formulas.calc_shaping_factor(
        elongation_psi95=self.elongation_psi95,
        triangularity_psi95=self.triangularity_psi95,
    )
    np.testing.assert_allclose(
        calculated_value,
        expected_value,
        rtol=1e-5,
    )

  def test_calc_separatrix_average_poloidal_field(self):
    # reference value from the first loop of the reference case in
    # https://github.com/cfs-energy/extended-lengyel
    expected_value = 0.2850657717047037
    shaping_factor = 1.4031849486079875
    calculated_value = (
        extended_lengyel_formulas.calc_separatrix_average_poloidal_field(
            shaping_factor=shaping_factor,
            minor_radius=self.minor_radius,
            plasma_current=self.plasma_current,
        )
    )
    np.testing.assert_allclose(
        calculated_value,
        expected_value,
        rtol=1e-5,
    )

  def test_calc_cylindrical_safety_factor(self):
    # reference value from the first loop of the reference case in
    # https://github.com/cfs-energy/extended-lengyel
    expected_value = 3.7290303009852943
    shaping_factor = 1.4031849486079875
    separatrix_average_poloidal_field = 0.2850657717047037
    calculated_value = extended_lengyel_formulas.calc_cylindrical_safety_factor(
        magnetic_field_on_axis=self.magnetic_field_on_axis,
        separatrix_average_poloidal_field=separatrix_average_poloidal_field,
        shaping_factor=shaping_factor,
        minor_radius=self.minor_radius,
        major_radius=self.major_radius,
    )
    np.testing.assert_allclose(
        calculated_value,
        expected_value,
        rtol=1e-5,
    )

  def test_calc_fieldline_pitch_at_omp(self):
    # reference value from the first loop of the reference case in
    # https://github.com/cfs-energy/extended-lengyel
    expected_value = 5.14589459864493
    calculated_value = extended_lengyel_formulas.calc_fieldline_pitch_at_omp(
        magnetic_field_on_axis=self.magnetic_field_on_axis,
        plasma_current=self.plasma_current,
        major_radius=self.major_radius,
        minor_radius=self.minor_radius,
        elongation_psi95=self.elongation_psi95,
        triangularity_psi95=self.triangularity_psi95,
        ratio_of_upstream_to_average_poloidal_field=extended_lengyel_defaults.RATIO_UPSTREAM_TO_AVG_BPOL,
    )
    np.testing.assert_allclose(
        calculated_value,
        expected_value,
        rtol=1e-5,
    )

  def test_calc_electron_temp_at_cc_interface(self):
    # reference value from the first loop of the reference case in
    # https://github.com/cfs-energy/extended-lengyel
    expected_value = 6.167578954082415
    calculated_value = (
        extended_lengyel_formulas.calc_electron_temp_at_cc_interface(
            target_electron_temp=self.target_electron_temp
        )
    )
    np.testing.assert_allclose(
        calculated_value,
        expected_value,
        rtol=1e-5,
    )


if __name__ == '__main__':
  absltest.main()

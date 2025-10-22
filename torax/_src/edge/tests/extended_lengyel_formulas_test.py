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

  def test_calc_momentum_loss_in_convection_layer(self):
    # reference value from the first loop of the reference case in
    # https://github.com/cfs-energy/extended-lengyel
    expected_momentum_loss = 0.5364587873343747
    target_electron_temp = 2.34  # [eV]

    calculated_momentum_loss = (
        extended_lengyel_formulas.calc_momentum_loss_in_convection_layer(
            target_electron_temp=target_electron_temp
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
    target_electron_temp = 2.34  # [eV]

    calculated_density_ratio = (
        extended_lengyel_formulas.calc_density_ratio_in_convection_layer(
            target_electron_temp=target_electron_temp
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
    target_electron_temp = 2.34  # [eV]

    calculated_power_loss = (
        extended_lengyel_formulas.calc_power_loss_in_convection_layer(
            target_electron_temp=target_electron_temp
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
    elongation_psi95 = 1.6  # [dimensionless]
    triangularity_psi95 = 0.3  # [dimensionless]

    calculated_value = extended_lengyel_formulas.calc_shaping_factor(
        elongation_psi95=elongation_psi95,
        triangularity_psi95=triangularity_psi95,
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
    minor_radius = 0.5  # [m]
    plasma_current = 1e6  # [A]

    calculated_value = (
        extended_lengyel_formulas.calc_separatrix_average_poloidal_field(
            shaping_factor=shaping_factor,
            minor_radius=minor_radius,
            plasma_current=plasma_current,
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
    magnetic_field_on_axis = 2.5  # [T]
    minor_radius = 0.5  # [m]
    major_radius = 1.65  # [m]

    calculated_value = extended_lengyel_formulas.calc_cylindrical_safety_factor(
        magnetic_field_on_axis=magnetic_field_on_axis,
        separatrix_average_poloidal_field=separatrix_average_poloidal_field,
        shaping_factor=shaping_factor,
        minor_radius=minor_radius,
        major_radius=major_radius,
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
    magnetic_field_on_axis = 2.5  # [T]
    plasma_current = 1e6  # [A]
    major_radius = 1.65  # [m]
    minor_radius = 0.5  # [m]
    elongation_psi95 = 1.6  # [dimensionless]
    triangularity_psi95 = 0.3  # [dimensionless]

    calculated_value = extended_lengyel_formulas.calc_fieldline_pitch_at_omp(
        magnetic_field_on_axis=magnetic_field_on_axis,
        plasma_current=plasma_current,
        major_radius=major_radius,
        minor_radius=minor_radius,
        elongation_psi95=elongation_psi95,
        triangularity_psi95=triangularity_psi95,
        ratio_of_upstream_to_average_poloidal_field=extended_lengyel_defaults.RATIO_UPSTREAM_TO_AVG_BPOL,
    )
    np.testing.assert_allclose(
        calculated_value,
        expected_value,
        rtol=1e-5,
    )

  def test_calc_Z_eff(self):

    c_z = 0.059314229517142054
    T_e = 0.05502789988290978
    Z_i = 1.0
    seed_impurity_weights = {'N': 1.0, 'Ar': 0.05}
    fixed_impurity_concentrations = {'He': 0.01}

    # reference value from the first loop of the reference case in
    # https://github.com/cfs-energy/extended-lengyel
    expected_Z_eff = 2.291360670810858

    calculated_Z_eff = extended_lengyel_formulas.calc_Z_eff(
        c_z=c_z,
        T_e=T_e,
        Z_i=Z_i,
        ne_tau=extended_lengyel_defaults.NE_TAU,
        seed_impurity_weights=seed_impurity_weights,
        fixed_impurity_concentrations=fixed_impurity_concentrations,
    )

    np.testing.assert_allclose(
        calculated_Z_eff,
        expected_Z_eff,
        rtol=1e-5,
    )

if __name__ == '__main__':
  absltest.main()

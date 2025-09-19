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
from torax._src.edge import extended_lengyel_formulas

# pylint: disable=invalid-name


class ExtendedLengyelFormulasTest(absltest.TestCase):

  def test_calc_alpha_t(self):
    """Test calc_alpha_t against reference values."""

    # Inputs and output from the first loop of the reference case in
    # https://github.com/cfs-energy/extended-lengyel
    separatrix_electron_density = 3.3e19  # m^-3
    separatrix_electron_temp = 0.1062293618373  # keV
    cylindrical_safety_factor = 3.7290303009853  # dimensionless
    major_radius = 1.65  # m
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
        err_msg='alpha_t calculation does not match the reference value.',
    )


if __name__ == '__main__':
  absltest.main()

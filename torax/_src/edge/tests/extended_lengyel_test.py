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
from torax._src.edge import divertor_sol_1d
from torax._src.edge import extended_lengyel
from torax._src.edge import extended_lengyel_defaults

# pylint: disable=invalid-name


class ExtendedLengyelTest(absltest.TestCase):

  def test_solve_for_c_z(self):
    """Test _solve_for_c_z against reference values."""

    # The plasma state is based on the second inner loop of
    # the reference case in https://github.com/cfs-energy/extended-lengyel

    sol_state = divertor_sol_1d.DivertorSOL1D(
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

    calculated_c_z, status = extended_lengyel._solve_for_c_z(
        q_parallel=sol_state.q_parallel,
        divertor_sol_1d=sol_state,
        seed_impurity_weights={'N': 1.0, 'Ar': 0.05},
        fixed_impurity_concentrations={'He': 0.01},
        ne_tau=extended_lengyel_defaults.NE_TAU,
    )

    expected_c_z = 0.06323862137705387

    self.assertEqual(status, extended_lengyel.SolveCzStatus.SUCCESS)
    np.testing.assert_allclose(
        calculated_c_z,
        expected_c_z,
        rtol=1e-4,
    )


if __name__ == '__main__':
  absltest.main()

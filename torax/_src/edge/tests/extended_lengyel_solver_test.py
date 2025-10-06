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
from torax._src.edge import extended_lengyel_defaults
from torax._src.edge import extended_lengyel_solvers

# pylint: disable=invalid-name


class ExtendedLengyelSolverTest(absltest.TestCase):

  def test_solve_for_c_z(self):
    """Test _solve_for_c_z against reference values."""

    # The plasma state is based on the second loop of
    # the reference case in https://github.com/cfs-energy/extended-lengyel

    params = divertor_sol_1d.ExtendedLengyelParameters(
        # Dummy values for unused parameters in these specific tests.
        major_radius=1.0,
        minor_radius=1.0,
        separatrix_average_poloidal_field=1.0,
        fieldline_pitch_at_omp=1.0,
        cylindrical_safety_factor=1.0,
        power_crossing_separatrix=1.0,
        ratio_of_upstream_to_average_poloidal_field=(
            extended_lengyel_defaults.RATIO_UPSTREAM_TO_AVG_BPOL
        ),
        fraction_of_P_SOL_to_divertor=(
            extended_lengyel_defaults.FRACTION_OF_PSOL_TO_DIVERTOR
        ),
        target_angle_of_incidence=(
            extended_lengyel_defaults.TARGET_ANGLE_OF_INCIDENCE
        ),
        wall_temperature=extended_lengyel_defaults.WALL_TEMPERATURE,
        ratio_of_molecular_to_ion_mass=(
            extended_lengyel_defaults.RATIO_MOLECULAR_TO_ION_MASS
        ),
        # Parameters from reference case
        seed_impurity_weights={'N': 1.0, 'Ar': 0.05},
        fixed_impurity_concentrations={'He': 0.01},
        main_ion_charge=1.0,
        ne_tau=extended_lengyel_defaults.NE_TAU,
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
    state = divertor_sol_1d.ExtendedLengyelState(
        q_parallel=3.39611622588553e8,
        c_z_prefactor=0.059314229517142096,
        kappa_e=1751.6010938527386,
        alpha_t=0.0,
        target_electron_temp=2.34,
    )
    sol_model = divertor_sol_1d.DivertorSOL1D(
        params=params,
        state=state,
    )

    calculated_c_z, status = extended_lengyel_solvers._solve_for_c_z_prefactor(
        sol_model=sol_model,
    )
    expected_c_z = 0.03487637336277587

    self.assertEqual(status, extended_lengyel_solvers.SolveStatus.SUCCESS)
    np.testing.assert_allclose(
        calculated_c_z,
        expected_c_z,
        rtol=5e-4,
    )

  def test_solve_for_qcc(self):
    """Test _solve_for_qcc."""

    # The plasma state is based on the first loop of the forward model reference
    # case in https://github.com/cfs-energy/extended-lengyel.

    params = divertor_sol_1d.ExtendedLengyelParameters(
        # Dummy values for unused parameters in these specific tests.
        major_radius=1.0,
        minor_radius=1.0,
        separatrix_average_poloidal_field=1.0,
        fieldline_pitch_at_omp=1.0,
        cylindrical_safety_factor=1.0,
        power_crossing_separatrix=1.0,
        ratio_of_upstream_to_average_poloidal_field=(
            extended_lengyel_defaults.RATIO_UPSTREAM_TO_AVG_BPOL
        ),
        fraction_of_P_SOL_to_divertor=(
            extended_lengyel_defaults.FRACTION_OF_PSOL_TO_DIVERTOR
        ),
        target_angle_of_incidence=(
            extended_lengyel_defaults.TARGET_ANGLE_OF_INCIDENCE
        ),
        wall_temperature=extended_lengyel_defaults.WALL_TEMPERATURE,
        ratio_of_molecular_to_ion_mass=(
            extended_lengyel_defaults.RATIO_MOLECULAR_TO_ION_MASS
        ),
        # Parameters from reference case
        seed_impurity_weights={},
        fixed_impurity_concentrations={
            'He': 0.01,
            'N': 0.038397305226362526,
            'Ar': 0.0019198652613181264,
        },
        main_ion_charge=1.0,
        ne_tau=extended_lengyel_defaults.NE_TAU,
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
    state = divertor_sol_1d.ExtendedLengyelState(
        q_parallel=5.061935771095335e8,
        c_z_prefactor=0.0,
        kappa_e=1931.8277173925928,
        alpha_t=0.0,
        target_electron_temp=2.34,
    )
    sol_model = divertor_sol_1d.DivertorSOL1D(
        params=params,
        state=state,
    )

    calculated_qcc, status = extended_lengyel_solvers._solve_for_qcc(
        sol_model=sol_model,
    )
    expected_qcc = 1.11662e08
    self.assertEqual(status, extended_lengyel_solvers.SolveStatus.SUCCESS)
    np.testing.assert_allclose(calculated_qcc, expected_qcc, rtol=5e-4)

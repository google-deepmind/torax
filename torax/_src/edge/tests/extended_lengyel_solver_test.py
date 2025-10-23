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
from torax._src.edge import extended_lengyel_formulas
from torax._src.edge import extended_lengyel_solvers
from torax._src.solver import jax_root_finding

# pylint: disable=invalid-name


class ExtendedLengyelSolverInverseTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # The plasma state is based on the second loop of
    # the reference case in https://github.com/cfs-energy/extended-lengyel
    elongation_psi95 = 1.6
    triangularity_psi95 = 0.3
    plasma_current = 1e6
    minor_radius = 0.5
    major_radius = 1.65
    magnetic_field_on_axis = 2.5

    shaping_factor = extended_lengyel_formulas.calc_shaping_factor(
        elongation_psi95=elongation_psi95,
        triangularity_psi95=triangularity_psi95,
    )
    separatrix_average_poloidal_field = (
        extended_lengyel_formulas.calc_separatrix_average_poloidal_field(
            plasma_current=plasma_current,
            minor_radius=minor_radius,
            shaping_factor=shaping_factor,
        )
    )
    cylindrical_safety_factor = (
        extended_lengyel_formulas.calc_cylindrical_safety_factor(
            magnetic_field_on_axis=magnetic_field_on_axis,
            separatrix_average_poloidal_field=separatrix_average_poloidal_field,
            shaping_factor=shaping_factor,
            minor_radius=minor_radius,
            major_radius=major_radius,
        )
    )
    fieldline_pitch_at_omp = extended_lengyel_formulas.calc_fieldline_pitch_at_omp(
        magnetic_field_on_axis=magnetic_field_on_axis,
        plasma_current=plasma_current,
        major_radius=major_radius,
        minor_radius=minor_radius,
        elongation_psi95=elongation_psi95,
        triangularity_psi95=triangularity_psi95,
        ratio_of_upstream_to_average_poloidal_field=extended_lengyel_defaults.RATIO_UPSTREAM_TO_AVG_BPOL,
    )

    self.params = divertor_sol_1d.ExtendedLengyelParameters(
        major_radius=major_radius,
        minor_radius=minor_radius,
        separatrix_average_poloidal_field=separatrix_average_poloidal_field,
        fieldline_pitch_at_omp=fieldline_pitch_at_omp,
        cylindrical_safety_factor=cylindrical_safety_factor,
        power_crossing_separatrix=5.5e6,
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
        seed_impurity_weights={'N': 1.0, 'Ar': 0.05},
        fixed_impurity_concentrations={'He': 0.01},
        main_ion_charge=1.0,
        mean_ion_charge_state=1.0,
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

  def test_successful_solve_for_c_z(self):
    state = divertor_sol_1d.ExtendedLengyelState(
        q_parallel=3.39611622588553e8,
        c_z_prefactor=0.059314229517142096,
        kappa_e=1751.6010938527386,
        alpha_t=0.0,
        target_electron_temp=2.34,
    )
    sol_model = divertor_sol_1d.DivertorSOL1D(
        params=self.params,
        state=state,
    )

    calculated_c_z, status = extended_lengyel_solvers._solve_for_c_z_prefactor(
        sol_model=sol_model,
    )
    expected_c_z = 0.03487637336277587

    self.assertEqual(status, extended_lengyel_solvers.PhysicsOutcome.SUCCESS)
    np.testing.assert_allclose(
        calculated_c_z,
        expected_c_z,
        rtol=5e-4,
    )

  def test_unsuccessful_solve_for_c_z(self):
    # The input q_parallel (heat flux) is set so low that the power loss in the
    # cc region is sufficient to reach the target temperature for even no seeded
    # impurities, meaning that only "negative" c_z could satisfy the equation,
    # which is unphysical.

    state = divertor_sol_1d.ExtendedLengyelState(
        q_parallel=1e3,
        c_z_prefactor=0.059314229517142096,
        kappa_e=1751.6010938527386,
        alpha_t=0.0,
        target_electron_temp=2.34,
    )
    sol_model = divertor_sol_1d.DivertorSOL1D(
        params=self.params,
        state=state,
    )

    calculated_c_z, status = extended_lengyel_solvers._solve_for_c_z_prefactor(
        sol_model=sol_model,
    )
    expected_c_z = 0.0

    self.assertEqual(
        status, extended_lengyel_solvers.PhysicsOutcome.C_Z_PREFACTOR_NEGATIVE
    )
    np.testing.assert_allclose(
        calculated_c_z,
        expected_c_z,
    )

  def test_inverse_unsuccessful_newton_solve_but_successful_hybrid_solve(self):
    # The initial guess state is deliberately set far from the solution, by
    # having too low a q_parallel. But the hybrid solver should still converge
    # to the solution while Newton-Raphson fails.
    state = divertor_sol_1d.ExtendedLengyelState(
        q_parallel=1e3,
        c_z_prefactor=0.0,
        kappa_e=1751.6010938527386,
        alpha_t=0.1,
        target_electron_temp=2.34,
    )
    initial_sol_model = divertor_sol_1d.DivertorSOL1D(
        params=self.params,
        state=state,
    )
    _, status = extended_lengyel_solvers.inverse_mode_newton_solver(
        initial_sol_model=initial_sol_model,
    )
    assert isinstance(status.numerics_outcome, jax_root_finding.RootMetadata)
    self.assertEqual(status.numerics_outcome.error, 1)

    final_sol_model, status = (
        extended_lengyel_solvers.inverse_mode_hybrid_solver(
            initial_sol_model=initial_sol_model,
        )
    )
    assert isinstance(status.numerics_outcome, jax_root_finding.RootMetadata)
    self.assertEqual(status.numerics_outcome.error, 0)
    np.testing.assert_allclose(
        final_sol_model.seed_impurity_concentrations['N'],
        0.038397305226362526,
        rtol=1e-3,
    )
    np.testing.assert_allclose(
        final_sol_model.seed_impurity_concentrations['Ar'],
        0.0019198652613181264,
        rtol=1e-3,
    )


class ExtendedLengyelSolverForwardTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # The plasma state is based on the first loop of the forward model reference
    # case in https://github.com/cfs-energy/extended-lengyel.
    elongation_psi95 = 1.6
    triangularity_psi95 = 0.3
    plasma_current = 1e6
    minor_radius = 0.5
    major_radius = 1.65
    magnetic_field_on_axis = 2.5

    shaping_factor = extended_lengyel_formulas.calc_shaping_factor(
        elongation_psi95=elongation_psi95,
        triangularity_psi95=triangularity_psi95,
    )
    separatrix_average_poloidal_field = (
        extended_lengyel_formulas.calc_separatrix_average_poloidal_field(
            plasma_current=plasma_current,
            minor_radius=minor_radius,
            shaping_factor=shaping_factor,
        )
    )
    cylindrical_safety_factor = (
        extended_lengyel_formulas.calc_cylindrical_safety_factor(
            magnetic_field_on_axis=magnetic_field_on_axis,
            separatrix_average_poloidal_field=separatrix_average_poloidal_field,
            shaping_factor=shaping_factor,
            minor_radius=minor_radius,
            major_radius=major_radius,
        )
    )
    fieldline_pitch_at_omp = extended_lengyel_formulas.calc_fieldline_pitch_at_omp(
        magnetic_field_on_axis=magnetic_field_on_axis,
        plasma_current=plasma_current,
        major_radius=major_radius,
        minor_radius=minor_radius,
        elongation_psi95=elongation_psi95,
        triangularity_psi95=triangularity_psi95,
        ratio_of_upstream_to_average_poloidal_field=extended_lengyel_defaults.RATIO_UPSTREAM_TO_AVG_BPOL,
    )

    self.params = divertor_sol_1d.ExtendedLengyelParameters(
        # Dummy values for unused parameters in these specific tests.
        major_radius=major_radius,
        minor_radius=minor_radius,
        separatrix_average_poloidal_field=separatrix_average_poloidal_field,
        fieldline_pitch_at_omp=fieldline_pitch_at_omp,
        cylindrical_safety_factor=cylindrical_safety_factor,
        power_crossing_separatrix=5.5e6,
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
        mean_ion_charge_state=1.0,
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

  def test_successful_solve_for_qcc(self):
    state = divertor_sol_1d.ExtendedLengyelState(
        q_parallel=5.061935771095335e8,
        c_z_prefactor=0.0,
        kappa_e=1931.8277173925928,
        alpha_t=0.0,
        target_electron_temp=2.34,
    )
    sol_model = divertor_sol_1d.DivertorSOL1D(
        params=self.params,
        state=state,
    )

    calculated_qcc, status = extended_lengyel_solvers._solve_for_qcc(
        sol_model=sol_model,
    )
    expected_qcc = 1.11662e08
    self.assertEqual(status, extended_lengyel_solvers.PhysicsOutcome.SUCCESS)
    np.testing.assert_allclose(calculated_qcc, expected_qcc, rtol=5e-4)

  def test_unsuccessful_solve_for_qcc(self):
    # The input q_parallel (heat flux) is modified to be too low, and
    # inconsistent with the rest of the state such that no power balance is
    # reached and unphysical negative heat flux is "predicted" at the
    # convective-conduction transition point. This is a sign of full detachment
    # in the divertor.

    state = divertor_sol_1d.ExtendedLengyelState(
        q_parallel=1e3,
        c_z_prefactor=0.0,
        kappa_e=1931.8277173925928,
        alpha_t=0.0,
        target_electron_temp=2.34,
    )
    sol_model = divertor_sol_1d.DivertorSOL1D(
        params=self.params,
        state=state,
    )

    calculated_qcc, status = extended_lengyel_solvers._solve_for_qcc(
        sol_model=sol_model,
    )
    self.assertEqual(
        status, extended_lengyel_solvers.PhysicsOutcome.Q_CC_SQUARED_NEGATIVE
    )
    np.testing.assert_allclose(calculated_qcc, 0.0)

  def test_forward_unsuccessful_newton_solve_but_successful_hybrid_solve(self):
    # The initial guess state is deliberately set far from the solution, by
    # having too low a q_parallel. But the hybrid solver should still converge
    # to the solution while Newton-Raphson fails.
    state = divertor_sol_1d.ExtendedLengyelState(
        q_parallel=1e3,
        c_z_prefactor=0.0,
        kappa_e=1751.6010938527386,
        alpha_t=0.1,
        target_electron_temp=2.0,
    )
    initial_sol_model = divertor_sol_1d.DivertorSOL1D(
        params=self.params,
        state=state,
    )
    _, status = extended_lengyel_solvers.forward_mode_newton_solver(
        initial_sol_model=initial_sol_model,
    )
    assert isinstance(status.numerics_outcome, jax_root_finding.RootMetadata)
    self.assertEqual(status.numerics_outcome.error, 1)

    final_sol_model, status = (
        extended_lengyel_solvers.forward_mode_hybrid_solver(
            initial_sol_model=initial_sol_model,
        )
    )
    assert isinstance(status.numerics_outcome, jax_root_finding.RootMetadata)
    self.assertEqual(status.numerics_outcome.error, 0)
    np.testing.assert_allclose(
        final_sol_model.state.target_electron_temp,
        2.34,
        rtol=1e-3,
    )


if __name__ == '__main__':
  absltest.main()

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

"""Tests for the ExtendedLengyelModel integration."""

from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax._src import math_utils
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.edge import extended_lengyel_enums
from torax._src.edge import extended_lengyel_model
from torax._src.edge import extended_lengyel_solvers
from torax._src.edge import extended_lengyel_standalone
from torax._src.edge import pydantic_model
from torax._src.fvm import cell_variable
from torax._src.geometry import standard_geometry
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.orchestration import run_simulation
from torax._src.sources import generic_ion_el_heat_source
from torax._src.sources import source_profiles
from torax._src.test_utils import sim_test_case

# pylint: disable=invalid-name


class ExtendedLengyelModelTest(absltest.TestCase):

  def test_call_inverse_mode(self):
    """Tests that ExtendedLengyelModel.__call__ works correctly in inverse mode.

    This test uses the same physical parameters as the standalone test
    `test_run_extended_lengyel_model_inverse_mode_fixed_step` in
    `extended_lengyel_standalone_test.py` and verifies it matches the expected
    outputs when called through the model adapter with TORAX state objects.
    """

    # --- 1. Setup Mock TORAX State Objects ---
    # Set up TORAX state objects with the same physical parameters as the
    # standalone test, i.e.
    # 'power_crossing_separatrix': 5.5e6,
    # 'separatrix_electron_density': 3.3e19,
    # 'main_ion_charge': 1.0,
    # 'mean_ion_charge_state': 1.0,
    # 'magnetic_field_on_axis': 2.5,
    # 'plasma_current': 1.0e6,
    # 'major_radius': 1.65,
    # 'minor_radius': 0.5,
    # 'elongation_psi95': 1.6,
    # 'triangularity_psi95': 0.3,
    # 'average_ion_mass': 2.0,

    n_rho = 10
    mock_geo = mock.MagicMock(spec=standard_geometry.StandardGeometry)
    # Mock geo quantities that are needed for the test.
    mock_geo.R_major = np.array(1.65)
    mock_geo.a_minor = np.array(0.5)
    mock_geo.B_0 = np.array(2.5)
    mock_geo.elongation_face = np.array([1.6] * (n_rho + 1))
    mock_geo.delta_face = np.array([0.3] * (n_rho + 1))
    mock_geo.rho_face_norm = np.linspace(0, 1.0, (n_rho + 1))
    mock_geo.rho_norm = np.linspace(
        1.0 / (2 * n_rho), 1.0 - 1.0 / (2 * n_rho), n_rho
    )
    mock_geo.rho = mock_geo.rho_norm * mock_geo.a_minor
    mock_geo.drho_norm = np.ones(n_rho) / n_rho
    mock_geo.vpr = np.linspace(0, 100, n_rho)
    mock_geo.connection_length_target = None
    mock_geo.connection_length_divertor = None
    mock_geo.target_angle_of_incidence = None
    mock_geo.R_target = None
    mock_geo.R_OMP = None
    mock_geo.B_pol_OMP = None
    mock_geo.diverted = np.bool(True)

    # Mock CoreProfiles
    mock_core_profiles = mock.MagicMock(spec=state.CoreProfiles)
    mock_core_profiles.Ip_profile_face = np.linspace(0.0, 1.0e6, (n_rho + 1))
    mock_core_profiles.A_i = np.array(2.0)
    mock_core_profiles.A_impurity_face = np.array([10.0] * (n_rho + 1))
    mock_core_profiles.Z_i_face = np.array([1.0] * (n_rho + 1))

    mock_psi = mock.MagicMock(spec=cell_variable.CellVariable)
    mock_psi.face_value.return_value = np.linspace(0, 1.0, (n_rho + 1))
    mock_core_profiles.psi = mock_psi

    mock_n_e = mock.MagicMock(spec=cell_variable.CellVariable)
    mock_n_e.face_value.return_value = np.array([3.3e19] * (n_rho + 1))
    mock_core_profiles.n_e = mock_n_e

    mock_n_i = mock.MagicMock(spec=cell_variable.CellVariable)
    mock_n_i.face_value.return_value = np.array([3.3e19] * (n_rho + 1))
    mock_core_profiles.n_i = mock_n_i

    mock_n_impurity = mock.MagicMock(spec=cell_variable.CellVariable)
    mock_n_impurity.face_value.return_value = np.zeros((n_rho + 1))
    mock_core_profiles.n_impurity = mock_n_impurity

    # Set up CoreSources to give P_SOL_total = 5.5e6 W.
    target_P_SOL = 5.5e6

    ion_heat, el_heat = generic_ion_el_heat_source.calc_generic_heat_source(
        geo=mock_geo,
        gaussian_location=0.5,
        gaussian_width=0.2,
        P_total=target_P_SOL,
        electron_heat_fraction=0.7,
        absorption_fraction=1.0,
    )

    mock_core_sources = source_profiles.SourceProfiles(
        bootstrap_current=bootstrap_current_base.BootstrapCurrent.zeros(
            mock_geo
        ),
        qei=source_profiles.QeiInfo.zeros(mock_geo),
        T_e={'generic_heat': el_heat},
        T_i={'generic_heat': ion_heat},
    )

    # Verify that the mock sources integrate to the target power.
    dP_e_drho = mock_core_sources.total_sources('T_e', mock_geo)
    dP_i_drho = mock_core_sources.total_sources('T_i', mock_geo)

    integrated_P_SOL = math_utils.cell_integration(
        dP_e_drho + dP_i_drho, mock_geo
    )
    np.testing.assert_allclose(
        integrated_P_SOL,
        target_P_SOL,
        rtol=1e-6,
        err_msg='Mock sources do not integrate to the expected target P_SOL.',
    )

    # --- 2. Configure Edge Model ---
    # Provide the remaining parameters via the edge config.
    edge_config = pydantic_model.ExtendedLengyelConfig(
        model_name='extended_lengyel',
        computation_mode=extended_lengyel_enums.ComputationMode.INVERSE,
        solver_mode=extended_lengyel_enums.SolverMode.FIXED_STEP,
        target_electron_temp=2.34,
        parallel_connection_length=20.0,
        divertor_parallel_length=5.0,
        seed_impurity_weights={'N': 1.0, 'Ar': 0.05},
        fixed_impurity_concentrations={'He': 0.01},
        enrichment_factor={'N': 1.0, 'Ar': 1.0, 'He': 1.0},
    )
    edge_runtime_params = edge_config.build_runtime_params(t=0.0)

    # Mock full RuntimeParams to hold the edge params
    mock_runtime_params = mock.MagicMock(spec=runtime_params_lib.RuntimeParams)
    mock_runtime_params.edge = edge_runtime_params

    # --- 3. Run Model and Assert ---
    model = edge_config.build_edge_model()
    outputs = model(
        mock_runtime_params,
        mock_geo,
        mock_core_profiles,
        mock_core_sources,
    )

    # Expected values from standalone reference case in:
    # https://github.com/cfs-energy/extended-lengyel
    _RTOL = 5e-4
    expected_outputs = {
        'neutral_pressure_in_divertor': 1.737773924511501,
        'alpha_t': 0.35908862950459736,
        'q_parallel': 3.64822996e8,
        'heat_flux_perp_to_target': 7.92853e5,
        'separatrix_electron_temp': 0.1028445648,  # in keV
        'separatrix_Z_eff': 1.8621973566614212,
        'seed_impurity_concentrations': {
            'N': 0.038397305226362526,
            'Ar': 0.0019198652613181264,
        },
    }

    self.assertEqual(
        outputs.solver_status.physics_outcome,
        extended_lengyel_solvers.PhysicsOutcome.SUCCESS,
    )
    self.assertEqual(
        outputs.solver_status.numerics_outcome,
        extended_lengyel_solvers.FixedStepOutcome.SUCCESS,
    )
    for key, value in expected_outputs.items():
      if key == 'seed_impurity_concentrations':
        assert isinstance(value, dict)
        for impurity, conc in value.items():
          self.assertIn(impurity, outputs.seed_impurity_concentrations)
          np.testing.assert_allclose(
              outputs.seed_impurity_concentrations[impurity],
              conc,
              rtol=_RTOL,
              err_msg=f'Impurity concentration for {impurity} does not match.',
          )
      else:
        np.testing.assert_allclose(getattr(outputs, key), value, rtol=_RTOL)


class ExtendedLengyelModelValidationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.model = extended_lengyel_model.ExtendedLengyelModel()

    self.mock_core_profiles = mock.MagicMock(spec=state.CoreProfiles)
    mock_psi = mock.MagicMock(spec=cell_variable.CellVariable)
    mock_psi.face_value.return_value = np.array([0.0, 0.5, 1.0])
    mock_psi.face_grad.return_value = np.array([0.0, 0.5, 1.0])
    self.mock_core_profiles.psi = mock_psi

    self.mock_runtime_params = mock.MagicMock(
        spec=runtime_params_lib.RuntimeParams
    )

  def _create_edge_params(self, **kwargs):
    # Defaults for required fields to avoid clutter in tests
    defaults = {
        'computation_mode': extended_lengyel_enums.ComputationMode.FORWARD,
        'solver_mode': extended_lengyel_enums.SolverMode.FIXED_STEP,
        'update_temperatures': True,
        'fixed_step_iterations': 1,
        'newton_raphson_iterations': 1,
        'newton_raphson_tol': 1e-5,
        'ne_tau': 0.0,
        'divertor_broadening_factor': 1.0,
        'ratio_bpol_omp_to_bpol_avg': 1.0,
        'sheath_heat_transmission_factor': 1.0,
        'fraction_of_P_SOL_to_divertor': 1.0,
        'SOL_conduction_fraction': 1.0,
        'ratio_of_molecular_to_ion_mass': 1.0,
        'wall_temperature': 1.0,
        'separatrix_mach_number': 1.0,
        'separatrix_ratio_of_ion_to_electron_temp': 1.0,
        'separatrix_ratio_of_electron_to_ion_density': 1.0,
        'target_ratio_of_ion_to_electron_temp': 1.0,
        'target_ratio_of_electron_to_ion_density': 1.0,
        'target_mach_number': 1.0,
        'seed_impurity_weights': {},
        'fixed_impurity_concentrations': {},
        'enrichment_factor': {},
        'target_electron_temp': None,
        # Geometric params default to None to test validation
        'parallel_connection_length': None,
        'divertor_parallel_length': None,
        'toroidal_flux_expansion': None,
        'target_angle_of_incidence': None,
    }
    defaults.update(kwargs)
    return extended_lengyel_model.RuntimeParams(**defaults)

  def test_resolve_geo_params_precedence_geo_over_config(self):
    mock_geo = mock.MagicMock(spec=standard_geometry.StandardGeometry)
    mock_geo.connection_length_target = 100.0
    mock_geo.connection_length_divertor = 10.0
    mock_geo.target_angle_of_incidence = 5.0
    mock_geo.R_target = 4.0
    mock_geo.R_OMP = 2.0
    mock_geo.B_pol_OMP = 1.0
    mock_geo.diverted = True
    mock_geo.g2_face = np.array([0.0, 0.5, 1.0])
    mock_geo.vpr_face = np.array([0.0, 0.5, 1.0])

    # Config also provides values (conflicting)
    edge_params = self._create_edge_params(
        parallel_connection_length=30.0,
        divertor_parallel_length=20.0,
        target_angle_of_incidence=1.0,
        toroidal_flux_expansion=3.0,
    )

    with self.assertLogs(level='WARNING') as logs:
      resolved = extended_lengyel_model._resolve_geometric_parameters(
          mock_geo, self.mock_core_profiles, edge_params
      )

    # Verify values from Geometry are used
    self.assertEqual(resolved.parallel_connection_length, 100.0)
    self.assertEqual(resolved.divertor_parallel_length, 10.0)
    self.assertEqual(resolved.target_angle_of_incidence, 5.0)
    self.assertEqual(resolved.toroidal_flux_expansion, 2.0)

    # Verify warnings were logged for overrides
    self.assertTrue(
        any('parallel_connection_length' in r.message for r in logs.records)
    )
    self.assertTrue(
        any('divertor_parallel_length' in r.message for r in logs.records)
    )

  def test_resolve_geo_params_fallback_to_config(self):
    """Test fallback to Config values when Geometry values are missing, with warning."""
    mock_geo = mock.MagicMock(spec=standard_geometry.StandardGeometry)
    mock_geo.connection_length_target = None
    mock_geo.connection_length_divertor = None
    mock_geo.target_angle_of_incidence = None
    mock_geo.R_target = None
    mock_geo.R_OMP = None
    mock_geo.B_pol_OMP = None
    mock_geo.diverted = None

    edge_params = self._create_edge_params(
        parallel_connection_length=50.0,
        divertor_parallel_length=5.0,
        target_angle_of_incidence=2.0,
        toroidal_flux_expansion=1.5,
    )

    with self.assertLogs(level='WARNING') as logs:
      resolved = extended_lengyel_model._resolve_geometric_parameters(
          mock_geo, self.mock_core_profiles, edge_params
      )

    # Verify values from Config are used
    self.assertEqual(resolved.parallel_connection_length, 50.0)
    self.assertEqual(resolved.target_angle_of_incidence, 2.0)

    # Verify warnings were logged for fallbacks
    self.assertTrue(
        any('not found in Geometry' in r.message for r in logs.records)
    )

  def test_resolve_geo_params_missing_raises_error(self):
    """Test ValueError is raised when a required parameter is missing from both."""
    mock_geo = mock.MagicMock(spec=standard_geometry.StandardGeometry)
    mock_geo.connection_length_target = None
    mock_geo.connection_length_divertor = None
    mock_geo.target_angle_of_incidence = None
    mock_geo.R_target = None
    mock_geo.R_OMP = None
    mock_geo.B_pol_OMP = None
    mock_geo.diverted = None

    # Config also missing values (set to None in _create_edge_params default)
    edge_params = self._create_edge_params()

    with self.assertRaisesRegex(
        ValueError, "Parameter 'parallel_connection_length' must be provided"
    ):
      extended_lengyel_model._resolve_geometric_parameters(
          mock_geo, self.mock_core_profiles, edge_params
      )

  def test_diverted_status_logic(self):
    """Test logic for divertor_broadening_factor based on diverted status."""
    mock_geo = mock.MagicMock(spec=standard_geometry.StandardGeometry)
    mock_geo.connection_length_target = 10.0
    mock_geo.connection_length_divertor = 1.0
    mock_geo.target_angle_of_incidence = 1.0
    mock_geo.R_target = 1.0
    mock_geo.R_OMP = 1.0
    mock_geo.B_pol_OMP = 1.0
    mock_geo.g2_face = np.array([0.0, 0.5, 1.0])
    mock_geo.vpr_face = np.array([0.0, 0.5, 1.0])

    edge_params = self._create_edge_params(divertor_broadening_factor=3.0)

    # Case 1: Diverted = True -> Use configured broadening
    mock_geo.diverted = True
    resolved = extended_lengyel_model._resolve_geometric_parameters(
        mock_geo, self.mock_core_profiles, edge_params
    )
    self.assertEqual(resolved.divertor_broadening_factor, 3.0)

    # Case 2: Diverted = False -> Broadening forced to 1.0
    mock_geo.diverted = False
    resolved = extended_lengyel_model._resolve_geometric_parameters(
        mock_geo, self.mock_core_profiles, edge_params
    )
    self.assertEqual(resolved.divertor_broadening_factor, 1.0)


class ExtendedLengyelModelCouplingTest(sim_test_case.SimTestCase):

  def test_edge_model_coupling_smoke(self):
    """Smoke test to ensure edge model is called and outputs are stored."""
    torax_config = self._get_torax_config(
        'test_iterhybrid_predictor_corrector.py'
    )
    torax_config.update_fields({
        'edge': {
            'model_name': 'extended_lengyel',
            'computation_mode': extended_lengyel_enums.ComputationMode.FORWARD,
            'fixed_impurity_concentrations': {'Ne': 5e-2},
            'enrichment_factor': {'Ne': 1.0},
            'parallel_connection_length': 50.0,
            'divertor_parallel_length': 10.0,
            'toroidal_flux_expansion': 4.0,
            'target_angle_of_incidence': 3.0,
        }
    })

    # Run for just a few steps
    torax_config.update_fields({
        'numerics.t_final': (
            torax_config.numerics.t_initial
            + 5 * torax_config.numerics.fixed_dt.value[0]
        )
    })

    _, state_history = run_simulation.run_simulation(torax_config)

    self.assertEqual(state_history.sim_error, state.SimError.NO_ERROR)

    # Basic sanity checks on output values
    for edge_output in state_history._edge_outputs:
      self.assertIsNotNone(edge_output)
      self.assertIsInstance(
          edge_output,
          extended_lengyel_standalone.ExtendedLengyelOutputs,
      )
      # Basic sanity check that the target temperature did not converge
      # to near-zero values, which is a common failure mode.
      self.assertGreater(edge_output.separatrix_electron_temp, 1e-2)

  @parameterized.named_parameters(
      ('updates_enabled', True, 2.0),
      ('updates_disabled', False, 2.0),
      ('non_unity_ratio', True, 3.0),
  )
  def test_boundary_condition_updates(
      self, update_temperatures, ion_to_electron_ratio
  ):
    """Tests that boundary conditions are correctly updated based on edge model."""

    initial_Te_bc = 0.5
    initial_Ti_bc = 0.5

    torax_config = self._get_torax_config(
        'test_iterhybrid_predictor_corrector.py'
    )
    torax_config.update_fields({
        'profile_conditions.T_e_right_bc': initial_Te_bc,
        'profile_conditions.T_i_right_bc': initial_Ti_bc,
        'edge': {
            'model_name': 'extended_lengyel',
            'computation_mode': extended_lengyel_enums.ComputationMode.FORWARD,
            'fixed_impurity_concentrations': {'Ne': 5e-2},
            'enrichment_factor': {'Ne': 1.0},
            'parallel_connection_length': 50.0,
            'divertor_parallel_length': 10.0,
            'toroidal_flux_expansion': 4.0,
            'target_angle_of_incidence': 3.0,
            # Test parameters
            'update_temperatures': update_temperatures,
            'target_ratio_of_ion_to_electron_temp': ion_to_electron_ratio,
        },
    })

    # Run for a few steps to allow updates to happen
    torax_config.update_fields({
        'numerics.t_final': (
            torax_config.numerics.t_initial
            + 3 * torax_config.numerics.fixed_dt.value[0]
        )
    })

    _, state_history = run_simulation.run_simulation(torax_config)
    self.assertEqual(state_history.sim_error, state.SimError.NO_ERROR)

    # Check the last time step
    final_state = state_history.core_profiles[-1]
    final_edge_output = state_history._edge_outputs[-1]
    self.assertIsNotNone(final_edge_output)

    if update_temperatures:
      # BCs should match edge model output
      expected_Te_bc = final_edge_output.separatrix_electron_temp
      expected_Ti_bc = expected_Te_bc * ion_to_electron_ratio
      np.testing.assert_allclose(
          final_state.T_e.right_face_constraint, expected_Te_bc, rtol=1e-5
      )
      np.testing.assert_allclose(
          final_state.T_i.right_face_constraint, expected_Ti_bc, rtol=1e-5
      )
      # Sanity check that it actually changed from initial
      with self.assertRaises(AssertionError):
        np.testing.assert_allclose(
            final_state.T_e.right_face_constraint, initial_Te_bc, rtol=1e-5
        )
    else:
      # BCs should remain at initial prescribed values
      np.testing.assert_allclose(
          final_state.T_e.right_face_constraint, initial_Te_bc, rtol=1e-5
      )
      np.testing.assert_allclose(
          final_state.T_i.right_face_constraint, initial_Ti_bc, rtol=1e-5
      )


if __name__ == '__main__':
  absltest.main()

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
import numpy as np
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.edge import extended_lengyel_enums
from torax._src.edge import extended_lengyel_solvers
from torax._src.edge import pydantic_model
from torax._src.fvm import cell_variable
from torax._src.geometry import standard_geometry
from torax._src.output_tools import post_processing

# pylint: disable=invalid-name


class ExtendedLengyelModelTest(absltest.TestCase):

  def test_call_inverse_mode(self):
    """Tests that ExtendedLengyelModel.__call__ works correctly in inverse mode.

    This test uses the same physical parameters as the standalone test
    `test_run_extended_lengyel_model_inverse_mode_fixed_step` in
    `extended_lengyel_standalone_test.py` and verifies it matches the expected
    outputs when called through the model adapter with TOR AX state objects.
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

    mock_geo = mock.MagicMock(spec=standard_geometry.StandardGeometry)
    mock_geo.R_major = np.array(1.65)
    mock_geo.a_minor = np.array(0.5)
    mock_geo.B_0 = np.array(2.5)
    mock_geo.elongation_face = np.array([1.5, 1.6, 1.7])
    mock_geo.delta_face = np.array([0.1, 0.3, 0.5])

    # Mock CoreProfiles
    psi_face = np.array([0.0, 0.95, 1.0])
    mock_core_profiles = mock.MagicMock(spec=state.CoreProfiles)
    mock_core_profiles.Ip_profile_face = np.array([0.0, 0.95e6, 1.0e6])
    mock_core_profiles.A_i = np.array(2.0)
    mock_core_profiles.A_impurity_face = np.array([10.0, 10.0, 10.0])
    mock_core_profiles.Z_i_face = np.array([1.0, 1.0, 1.0])

    mock_psi = mock.MagicMock(spec=cell_variable.CellVariable)
    mock_psi.face_value.return_value = psi_face
    mock_core_profiles.psi = mock_psi

    mock_n_e = mock.MagicMock(spec=cell_variable.CellVariable)
    mock_n_e.face_value.return_value = np.array([0.0, 3.0e19, 3.3e19])
    mock_core_profiles.n_e = mock_n_e

    mock_n_i = mock.MagicMock(spec=cell_variable.CellVariable)
    mock_n_i.face_value.return_value = np.array([0.0, 3.0e19, 3.3e19])
    mock_core_profiles.n_i = mock_n_i

    mock_n_impurity = mock.MagicMock(spec=cell_variable.CellVariable)
    mock_n_impurity.face_value.return_value = np.array([0.0, 0.0, 0.0])
    mock_core_profiles.n_impurity = mock_n_impurity

    mock_post_processed_outputs = mock.MagicMock(
        spec=post_processing.PostProcessedOutputs
    )
    mock_post_processed_outputs.P_SOL_total = np.array(5.5e6)

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
    mock_runtime_params = mock.MagicMock(
        spec=runtime_params_slice.RuntimeParams
    )
    mock_runtime_params.edge = edge_runtime_params

    # --- 3. Run Model and Assert ---
    model = edge_config.build_edge_model()
    outputs = model(
        mock_runtime_params,
        mock_geo,
        mock_core_profiles,
        mock_post_processed_outputs,
    )

    # Expected values from standalone test
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


if __name__ == '__main__':
  absltest.main()

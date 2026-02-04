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
import jax.numpy as jnp
import numpy as np
from torax._src import math_utils
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles.plasma_composition import electron_density_ratios
from torax._src.core_profiles.plasma_composition import plasma_composition as plasma_composition_lib
from torax._src.edge import extended_lengyel_enums
from torax._src.edge import extended_lengyel_model
from torax._src.edge import extended_lengyel_solvers
from torax._src.edge import extended_lengyel_standalone
from torax._src.edge import pydantic_model
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.geometry import standard_geometry
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.orchestration import run_simulation
from torax._src.sources import generic_ion_el_heat_source
from torax._src.sources import source_profiles
from torax._src.test_utils import sim_test_case

# pylint: disable=invalid-name


class ExtendedLengyelModelTest(parameterized.TestCase):

  def test_call_inverse_mode(self):
    """Tests that ExtendedLengyelModel.__call__ works correctly in inverse mode.

    This test uses the same physical parameters as the standalone test
    `test_run_extended_lengyel_model_inverse_mode_fixed_point` in
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
    mock_geo.geometry_type = geometry.GeometryType.FBT
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
    mock_geo.angle_of_incidence_target = None
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
        solver_mode=extended_lengyel_enums.SolverMode.FIXED_POINT,
        impurity_sot=extended_lengyel_model.FixedImpuritySourceOfTruth.CORE,
        T_e_target=2.34,
        connection_length_target=20.0,
        connection_length_divertor=5.0,
        angle_of_incidence_target=3.0,
        ratio_bpol_omp_to_bpol_avg=4.0 / 3.0,
        toroidal_flux_expansion=1.0,
        seed_impurity_weights={'N': 1.0, 'Ar': 0.05},
        fixed_impurity_concentrations={'He': 0.01},
        enrichment_factor={'N': 1.0, 'Ar': 1.0, 'He': 1.0},
        use_enrichment_model=False,
    )
    edge_runtime_params = edge_config.build_runtime_params(t=0.0)

    # Mock full RuntimeParams to hold the edge params
    mock_runtime_params = mock.MagicMock(spec=runtime_params_lib.RuntimeParams)
    mock_runtime_params.edge = edge_runtime_params
    mock_plasma_composition = mock.MagicMock(
        spec=plasma_composition_lib.RuntimeParams
    )
    mock_impurity_params = mock.MagicMock(
        spec=electron_density_ratios.RuntimeParams
    )
    # The standalone test has fixed He concentration of 0.01.
    # With enrichment=1.0, the core ratio at the edge should be 0.01.
    mock_impurity_params.n_e_ratios_face = {
        'He': np.array([0.01] * (n_rho + 1))
    }
    mock_plasma_composition.impurity = mock_impurity_params
    mock_runtime_params.plasma_composition = mock_plasma_composition

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
        'pressure_neutral_divertor': 1.737773924511501,
        'alpha_t': 0.35908862950459736,
        'q_parallel': 3.64822996e8,
        'q_perpendicular_target': 7.92853e5,
        'T_e_separatrix': 0.1028445648,  # in keV
        'Z_eff_separatrix': 1.8621973566614212,
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
        extended_lengyel_solvers.FixedPointOutcome.SUCCESS,
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

  @mock.patch.object(
      extended_lengyel_standalone, 'run_extended_lengyel_standalone'
  )
  def test_fixed_impurity_update_from_core(self, mock_run_standalone):
    """Tests that fixed impurities are updated from core when source_of_truth is CORE."""
    # Setup parameters
    n_rho = 10
    c_core_edge = 0.01
    enrichment = 2.0
    expected_c_edge = c_core_edge * enrichment

    # 1. Mock Geometry
    mock_geo = mock.MagicMock(spec=standard_geometry.StandardGeometry)
    mock_geo.geometry_type = geometry.GeometryType.FBT
    mock_geo.rho_face_norm = np.linspace(0, 1.0, (n_rho + 1))
    mock_geo.rho_norm = np.linspace(0, 1, n_rho)
    mock_geo.drho_norm = np.ones(n_rho) / n_rho
    # Need minimal attributes for _resolve_geometric_parameters
    mock_geo.connection_length_target = None
    mock_geo.connection_length_divertor = None
    mock_geo.angle_of_incidence_target = None
    mock_geo.R_target = None
    mock_geo.R_OMP = None
    mock_geo.B_pol_OMP = None
    mock_geo.diverted = True
    # Need attributes for calcs
    mock_geo.R_major = np.array(1.65)
    mock_geo.a_minor = np.array(0.5)
    mock_geo.B_0 = np.array(2.5)
    mock_geo.elongation_face = np.array([1.6] * (n_rho + 1))
    mock_geo.delta_face = np.array([0.3] * (n_rho + 1))
    mock_geo.vpr = np.ones(n_rho)

    # 2. Mock CoreProfiles
    mock_core_profiles = mock.MagicMock(spec=state.CoreProfiles)
    mock_psi = mock.MagicMock(spec=cell_variable.CellVariable)
    mock_n_e = mock.MagicMock(spec=cell_variable.CellVariable)
    mock_n_i = mock.MagicMock(spec=cell_variable.CellVariable)
    mock_n_impurity = mock.MagicMock(spec=cell_variable.CellVariable)
    mock_psi.face_value.return_value = np.linspace(0, 1.0, n_rho + 1)
    mock_n_e.face_value.return_value = np.ones(n_rho + 1) * 1e19
    mock_n_i.face_value.return_value = np.ones(n_rho + 1) * 1e19
    mock_n_impurity.face_value.return_value = np.zeros(n_rho + 1)
    mock_core_profiles.psi = mock_psi
    mock_core_profiles.n_e = mock_n_e
    mock_core_profiles.n_i = mock_n_i
    mock_core_profiles.n_impurity = mock_n_impurity
    mock_core_profiles.Z_i_face = np.ones(n_rho + 1)
    mock_core_profiles.A_i = np.array(2.0)
    mock_core_profiles.A_impurity_face = np.ones(n_rho + 1) * 4.0
    mock_core_profiles.Ip_profile_face = np.ones(n_rho + 1) * 1e6

    # 3. Mock CoreSources
    mock_core_sources = mock.MagicMock(spec=source_profiles.SourceProfiles)
    mock_core_sources.total_sources.return_value = np.zeros(n_rho)

    # 4. Configure Edge params
    edge_config = pydantic_model.ExtendedLengyelConfig(
        model_name='extended_lengyel',
        computation_mode=extended_lengyel_enums.ComputationMode.FORWARD,
        impurity_sot=extended_lengyel_model.FixedImpuritySourceOfTruth.CORE,
        fixed_impurity_concentrations={
            'He': 0.05
        },  # Stale value, should be ignored
        enrichment_factor={'He': enrichment},
        seed_impurity_weights={},
        # Other required fields
        connection_length_target=10.0,
        connection_length_divertor=5.0,
        angle_of_incidence_target=1.0,
        toroidal_flux_expansion=1.0,
        ratio_bpol_omp_to_bpol_avg=1.0,
        use_enrichment_model=False,
        diverted=True,
    )
    edge_params = edge_config.build_runtime_params(t=0.0)

    # 5. Construct RuntimeParams with plasma composition
    # We use a real dataclass for impurity params to satisfy isinstance check
    impurity_params = electron_density_ratios.RuntimeParams(
        n_e_ratios={},  # Only face used in the logic
        n_e_ratios_face={'He': jnp.linspace(0, c_core_edge, n_rho + 1)},
        A_avg=mock.MagicMock(),
        A_avg_face=mock.MagicMock(),
        Z_override=None,
    )

    mock_runtime_params = mock.MagicMock(spec=runtime_params_lib.RuntimeParams)
    mock_plasma_composition = mock.MagicMock(
        spec=plasma_composition_lib.RuntimeParams
    )
    mock_plasma_composition.impurity = impurity_params
    mock_runtime_params.plasma_composition = mock_plasma_composition
    mock_runtime_params.edge = edge_params

    # 6. Run
    model = extended_lengyel_model.ExtendedLengyelModel()
    model(mock_runtime_params, mock_geo, mock_core_profiles, mock_core_sources)

    # 7. Assert
    _, kwargs = mock_run_standalone.call_args
    passed_fixed_impurities = kwargs['fixed_impurity_concentrations']

    self.assertIn('He', passed_fixed_impurities)
    np.testing.assert_allclose(
        passed_fixed_impurities['He'],
        expected_c_edge,
        err_msg='Fixed impurity concentration was not updated from core.',
    )

  @mock.patch.object(
      extended_lengyel_standalone, 'run_extended_lengyel_standalone'
  )
  def test_initial_guess_from_previous_step(self, mock_run_standalone):
    """Tests that initial guess is derived from previous_edge_outputs when enabled."""
    n_rho = 10

    # 1. Mock Geometry
    mock_geo = mock.MagicMock(spec=standard_geometry.StandardGeometry)
    mock_geo.geometry_type = geometry.GeometryType.CHEASE
    mock_geo.rho_face_norm = np.linspace(0, 1.0, (n_rho + 1))
    mock_geo.rho_norm = np.linspace(0, 1, n_rho)
    mock_geo.drho_norm = np.ones(n_rho) / n_rho
    mock_geo.connection_length_target = None
    mock_geo.connection_length_divertor = None
    mock_geo.angle_of_incidence_target = None
    mock_geo.R_target = None
    mock_geo.R_OMP = None
    mock_geo.B_pol_OMP = None
    mock_geo.diverted = None
    mock_geo.R_major = np.array(1.65)
    mock_geo.a_minor = np.array(0.5)
    mock_geo.B_0 = np.array(2.5)
    mock_geo.elongation_face = np.array([1.6] * (n_rho + 1))
    mock_geo.delta_face = np.array([0.3] * (n_rho + 1))
    mock_geo.vpr = np.ones(n_rho)

    # 2. Mock CoreProfiles
    mock_core_profiles = mock.MagicMock(spec=state.CoreProfiles)
    for attr in ['psi', 'n_e', 'n_i', 'n_impurity']:
      m = mock.MagicMock(spec=cell_variable.CellVariable)
      m.face_value.return_value = np.ones(n_rho + 1)
      setattr(mock_core_profiles, attr, m)
    mock_core_profiles.Z_i_face = np.ones(n_rho + 1)
    mock_core_profiles.A_i = np.array(2.0)
    mock_core_profiles.A_impurity_face = np.ones(n_rho + 1)
    mock_core_profiles.Ip_profile_face = np.ones(n_rho + 1)

    # 3. Mock CoreSources
    mock_core_sources = mock.MagicMock(spec=source_profiles.SourceProfiles)
    mock_core_sources.total_sources.return_value = np.zeros(n_rho)

    # 4. Create previous_edge_outputs with known values
    previous_alpha_t = 0.42
    previous_kappa_e = 2500.0
    previous_T_e_sep_keV = 0.1  # Will be converted to 100 eV
    previous_T_e_target = 3.5
    previous_edge_outputs = extended_lengyel_standalone.ExtendedLengyelOutputs(
        q_parallel=jnp.array(1e8),
        q_perpendicular_target=jnp.array(1e6),
        T_e_separatrix=jnp.array(previous_T_e_sep_keV),
        T_e_target=jnp.array(previous_T_e_target),
        pressure_neutral_divertor=jnp.array(1.0),
        alpha_t=jnp.array(previous_alpha_t),
        kappa_e=jnp.array(previous_kappa_e),
        c_z_prefactor=jnp.array(0.0),
        Z_eff_separatrix=jnp.array(1.5),
        seed_impurity_concentrations={},
        solver_status=extended_lengyel_solvers.ExtendedLengyelSolverStatus(
            physics_outcome=extended_lengyel_solvers.PhysicsOutcome.SUCCESS,
            numerics_outcome=extended_lengyel_solvers.FixedPointOutcome.SUCCESS,
        ),
        calculated_enrichment={},
    )

    # 5. Configure Edge with use_previous_step_as_guess=True (default)
    edge_config = pydantic_model.ExtendedLengyelConfig(
        model_name='extended_lengyel',
        computation_mode=extended_lengyel_enums.ComputationMode.FORWARD,
        impurity_sot=extended_lengyel_model.FixedImpuritySourceOfTruth.CORE,
        fixed_impurity_concentrations={'He': 0.05},
        seed_impurity_weights={},
        connection_length_target=10.0,
        connection_length_divertor=5.0,
        angle_of_incidence_target=1.0,
        toroidal_flux_expansion=1.0,
        ratio_bpol_omp_to_bpol_avg=1.0,
        use_enrichment_model=False,
        enrichment_factor={'He': 1.0},
        diverted=True,
        initial_guess=pydantic_model.InitialGuessConfig(
            use_previous_step_as_guess=True,
        ),
    )
    edge_params = edge_config.build_runtime_params(t=0.0)

    # 6. Construct RuntimeParams
    mock_runtime_params = mock.MagicMock(spec=runtime_params_lib.RuntimeParams)
    mock_runtime_params.edge = edge_params
    impurity_params = electron_density_ratios.RuntimeParams(
        n_e_ratios={},
        n_e_ratios_face={'He': np.ones(n_rho + 1) * 0.01},
        A_avg=mock.MagicMock(),
        A_avg_face=mock.MagicMock(),
        Z_override=None,
    )
    mock_plasma_composition = mock.MagicMock(
        spec=plasma_composition_lib.RuntimeParams
    )
    mock_plasma_composition.impurity = impurity_params
    mock_runtime_params.plasma_composition = mock_plasma_composition

    # 7. Run with previous_edge_outputs
    model = extended_lengyel_model.ExtendedLengyelModel()
    model(
        mock_runtime_params,
        mock_geo,
        mock_core_profiles,
        mock_core_sources,
        previous_edge_outputs=previous_edge_outputs,
    )

    # 8. Assert initial_guess was passed with values from previous_edge_outputs
    _, kwargs = mock_run_standalone.call_args
    passed_initial_guess = kwargs['initial_guess']

    np.testing.assert_allclose(
        passed_initial_guess.alpha_t,
        previous_alpha_t,
        err_msg='alpha_t was not taken from previous_edge_outputs.',
    )
    np.testing.assert_allclose(
        passed_initial_guess.kappa_e,
        previous_kappa_e,
        err_msg='kappa_e was not taken from previous_edge_outputs.',
    )
    np.testing.assert_allclose(
        passed_initial_guess.T_e_separatrix,
        previous_T_e_sep_keV * 1e3,  # keV to eV conversion
        err_msg='T_e_separatrix was not correctly converted from keV to eV.',
    )
    np.testing.assert_allclose(
        passed_initial_guess.T_e_target,
        previous_T_e_target,
        err_msg='T_e_target was not taken from previous_edge_outputs.',
    )

  @parameterized.named_parameters(
      (
          'diverted',
          True,  # diverted
          3.0,  # expected broadening
      ),
      (
          'limited',
          False,  # diverted
          1.0,  # expected broadening
      ),
  )
  @mock.patch.object(
      extended_lengyel_standalone, 'run_extended_lengyel_standalone'
  )
  def test_broadening_limited_logic(
      self,
      diverted,
      expected_broadening,
      mock_run_standalone,
  ):
    """Tests correct parameter passing for limited vs diverted plasmas."""
    n_rho = 10
    _CONFIG_BROADENING = 3.0

    # 1. Mock Geometry
    # We simulate a non-FBT geometry so that 'diverted' is read from
    # the edge configuration rather than the geometry object.
    mock_geo = mock.MagicMock(spec=standard_geometry.StandardGeometry)
    mock_geo.geometry_type = geometry.GeometryType.CHEASE  # Not FBT

    # Mock minimal attributes required for geometric resolution
    mock_geo.rho_face_norm = np.linspace(0, 1.0, (n_rho + 1))
    mock_geo.rho_norm = np.linspace(0, 1, n_rho)
    mock_geo.drho_norm = np.ones(n_rho) / n_rho
    mock_geo.connection_length_target = None
    mock_geo.connection_length_divertor = None
    mock_geo.angle_of_incidence_target = None
    mock_geo.R_target = None
    mock_geo.R_OMP = None
    mock_geo.B_pol_OMP = None
    # Ensure the geometry doesn't override the config
    mock_geo.diverted = None

    # Mock attributes required for physics calculations before the call
    mock_geo.R_major = np.array(1.65)
    mock_geo.a_minor = np.array(0.5)
    mock_geo.B_0 = np.array(2.5)
    mock_geo.elongation_face = np.array([1.6] * (n_rho + 1))
    mock_geo.delta_face = np.array([0.3] * (n_rho + 1))
    mock_geo.vpr = np.ones(n_rho)

    # 2. Mock CoreProfiles
    mock_core_profiles = mock.MagicMock(spec=state.CoreProfiles)
    # Setup minimal profiles to avoid index errors
    for attr in ['psi', 'n_e', 'n_i', 'n_impurity']:
      m = mock.MagicMock(spec=cell_variable.CellVariable)
      m.face_value.return_value = np.ones(n_rho + 1)
      setattr(mock_core_profiles, attr, m)
    mock_core_profiles.Z_i_face = np.ones(n_rho + 1)
    mock_core_profiles.A_i = np.array(2.0)
    mock_core_profiles.A_impurity_face = np.ones(n_rho + 1)
    mock_core_profiles.Ip_profile_face = np.ones(n_rho + 1)

    # 3. Mock CoreSources
    mock_core_sources = mock.MagicMock(spec=source_profiles.SourceProfiles)
    mock_core_sources.total_sources.return_value = np.zeros(n_rho)

    edge_config = pydantic_model.ExtendedLengyelConfig(
        model_name='extended_lengyel',
        computation_mode=extended_lengyel_enums.ComputationMode.FORWARD,
        impurity_sot=extended_lengyel_model.FixedImpuritySourceOfTruth.CORE,
        fixed_impurity_concentrations={'He': 0.05},
        seed_impurity_weights={},
        connection_length_target=10.0,
        connection_length_divertor=5.0,
        angle_of_incidence_target=1.0,
        toroidal_flux_expansion=1.0,
        ratio_bpol_omp_to_bpol_avg=1.0,
        use_enrichment_model=True,
        diverted=diverted,
        divertor_broadening_factor=_CONFIG_BROADENING,
    )
    edge_params = edge_config.build_runtime_params(t=0.0)

    # 5. Construct RuntimeParams
    mock_runtime_params = mock.MagicMock(spec=runtime_params_lib.RuntimeParams)
    mock_runtime_params.edge = edge_params

    # Mock plasma composition for the impurity check inside the model
    impurity_params = electron_density_ratios.RuntimeParams(
        n_e_ratios={},
        n_e_ratios_face={'He': np.ones(n_rho + 1) * 0.01},
        A_avg=mock.MagicMock(),
        A_avg_face=mock.MagicMock(),
        Z_override=None,
    )
    mock_plasma_composition = mock.MagicMock(
        spec=plasma_composition_lib.RuntimeParams
    )
    mock_plasma_composition.impurity = impurity_params
    mock_runtime_params.plasma_composition = mock_plasma_composition

    # 6. Run
    model = extended_lengyel_model.ExtendedLengyelModel()
    model(mock_runtime_params, mock_geo, mock_core_profiles, mock_core_sources)

    # 7. Assert inputs passed to the standalone runner
    _, kwargs = mock_run_standalone.call_args

    # Check divertor broadening factor logic.
    # If limited (diverted=False), this should be forced to 1.0.
    np.testing.assert_allclose(
        kwargs['divertor_broadening_factor'],
        expected_broadening,
        err_msg=(
            f'Broadening factor mismatch. diverted={diverted}, '
            f'config={_CONFIG_BROADENING}, expected={expected_broadening}'
        ),
    )


class ExtendedLengyelModelValidationTest(parameterized.TestCase):

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
        'solver_mode': extended_lengyel_enums.SolverMode.FIXED_POINT,
        'impurity_sot': extended_lengyel_model.FixedImpuritySourceOfTruth.CORE,
        'update_temperatures': True,
        'update_impurities': True,
        'fixed_point_iterations': 1,
        'newton_raphson_iterations': 1,
        'newton_raphson_tol': 1e-5,
        'ne_tau': 0.0,
        'divertor_broadening_factor': 1.0,
        'ratio_bpol_omp_to_bpol_avg': 1.0,
        'sheath_heat_transmission_factor': 1.0,
        'fraction_of_P_SOL_to_divertor': 1.0,
        'SOL_conduction_fraction': 1.0,
        'ratio_of_molecular_to_ion_mass': 1.0,
        'T_wall': 1.0,
        'mach_separatrix': 1.0,
        'T_i_T_e_ratio_separatrix': 1.0,
        'n_e_n_i_ratio_separatrix': 1.0,
        'T_i_T_e_ratio_target': 1.0,
        'n_e_n_i_ratio_target': 1.0,
        'mach_target': 1.0,
        'seed_impurity_weights': {},
        'fixed_impurity_concentrations': {},
        'enrichment_factor': {},
        'T_e_target': None,
        # Geometric params default to None to test validation
        'connection_length_target': None,
        'connection_length_divertor': None,
        'toroidal_flux_expansion': None,
        'angle_of_incidence_target': None,
        'use_enrichment_model': False,
        'enrichment_model_multiplier': 1.0,
        'diverted': None,
        'initial_guess': extended_lengyel_model.InitialGuessRuntimeParams(
            alpha_t=0.0,
            alpha_t_provided=False,
            kappa_e=0.0,
            kappa_e_provided=False,
            T_e_separatrix=0.0,
            T_e_separatrix_provided=False,
            T_e_target=0.0,
            T_e_target_provided=False,
            c_z_prefactor=0.0,
            c_z_prefactor_provided=False,
            use_previous_step_as_guess=False,
        ),
    }
    defaults.update(kwargs)
    return extended_lengyel_model.RuntimeParams(**defaults)

  def test_resolve_geo_params_precedence_geo_over_config(self):
    mock_geo = mock.MagicMock(spec=standard_geometry.StandardGeometry)
    mock_geo.connection_length_target = jnp.array(100.0)
    mock_geo.connection_length_divertor = jnp.array(10.0)
    mock_geo.angle_of_incidence_target = jnp.array(5.0)
    mock_geo.R_target = jnp.array(4.0)
    mock_geo.R_OMP = jnp.array(2.0)
    mock_geo.B_pol_OMP = jnp.array(1.0)
    mock_geo.diverted = jnp.array(True)
    mock_geo.g2_face = np.array([0.0, 0.5, 1.0])
    mock_geo.vpr_face = np.array([0.0, 0.5, 1.0])

    # Config also provides values (conflicting)
    edge_params = self._create_edge_params(
        connection_length_target=30.0,
        connection_length_divertor=20.0,
        angle_of_incidence_target=1.0,
        toroidal_flux_expansion=3.0,
    )

    resolved = extended_lengyel_model._resolve_geometric_parameters(
        mock_geo, self.mock_core_profiles, edge_params, diverted=True
    )

    # Verify values from Geometry are used
    self.assertEqual(resolved.connection_length_target, 100.0)
    self.assertEqual(resolved.connection_length_divertor, 10.0)
    self.assertEqual(resolved.angle_of_incidence_target, 5.0)
    self.assertEqual(resolved.toroidal_flux_expansion, 2.0)

  def test_resolve_geo_params_fallback_to_config(self):
    """Test fallback to Config values when Geometry values are missing, with warning."""
    mock_geo = mock.MagicMock(spec=standard_geometry.StandardGeometry)
    mock_geo.connection_length_target = None
    mock_geo.connection_length_divertor = None
    mock_geo.angle_of_incidence_target = None
    mock_geo.R_target = None
    mock_geo.R_OMP = None
    mock_geo.B_pol_OMP = None
    mock_geo.diverted = None

    edge_params = self._create_edge_params(
        connection_length_target=50.0,
        connection_length_divertor=5.0,
        angle_of_incidence_target=2.0,
        toroidal_flux_expansion=1.5,
    )

    with self.assertLogs(level='WARNING') as logs:
      resolved = extended_lengyel_model._resolve_geometric_parameters(
          mock_geo, self.mock_core_profiles, edge_params, diverted=True
      )

    # Verify values from Config are used
    self.assertEqual(resolved.connection_length_target, 50.0)
    self.assertEqual(resolved.angle_of_incidence_target, 2.0)

    # Verify warnings were logged for fallbacks
    self.assertTrue(
        any('not found in Geometry' in r.message for r in logs.records)
    )

  def test_resolve_geo_params_missing_raises_error(self):
    """Test ValueError is raised when a required parameter is missing from both."""
    mock_geo = mock.MagicMock(spec=standard_geometry.StandardGeometry)
    mock_geo.connection_length_target = None
    mock_geo.connection_length_divertor = None
    mock_geo.angle_of_incidence_target = None
    mock_geo.R_target = None
    mock_geo.R_OMP = None
    mock_geo.B_pol_OMP = None
    mock_geo.diverted = None

    # Config also missing values (set to None in _create_edge_params default)
    edge_params = self._create_edge_params()

    with self.assertRaisesRegex(
        ValueError, "Parameter 'connection_length_target' must be provided"
    ):
      extended_lengyel_model._resolve_geometric_parameters(
          mock_geo, self.mock_core_profiles, edge_params, diverted=True
      )

  @parameterized.named_parameters(
      ('zero_value', 0.0),
      ('nan_value', np.nan),
  )
  def test_resolve_geo_params_fallback_on_invalid_geo(self, invalid_value):
    """Test fallback to Config when Geometry value is invalid (0 or NaN)."""
    mock_geo = mock.MagicMock(spec=standard_geometry.StandardGeometry)
    mock_geo.connection_length_target = jnp.array(invalid_value)
    mock_geo.connection_length_divertor = jnp.array(10.0)  # Valid
    # Other geo params None to trigger other paths (or minimal path)
    mock_geo.angle_of_incidence_target = jnp.array(5.0)
    mock_geo.R_target = None
    mock_geo.R_OMP = None
    mock_geo.B_pol_OMP = None
    mock_geo.diverted = jnp.array(True)

    edge_params = self._create_edge_params(
        connection_length_target=50.0,
        toroidal_flux_expansion=1.5,
    )

    resolved = extended_lengyel_model._resolve_geometric_parameters(
        mock_geo, self.mock_core_profiles, edge_params, diverted=True
    )

    # Verify fallback to config value for invalid geo param
    self.assertEqual(resolved.connection_length_target, 50.0)
    # Verify valid geo param is still used
    self.assertEqual(resolved.connection_length_divertor, 10.0)

  @parameterized.named_parameters(
      ('zero_value', 0.0),
      ('nan_value', np.nan),
  )
  def test_resolve_geo_params_raises_on_invalid_geo_no_fallback(
      self, invalid_value
  ):
    """Test Error is raised when Geometry value is invalid and no Config fallback."""
    mock_geo = mock.MagicMock(spec=standard_geometry.StandardGeometry)
    mock_geo.connection_length_target = jnp.array(invalid_value)
    # Other params to satisfy requirements or just fail on the first one
    mock_geo.connection_length_divertor = jnp.array(10.0)
    mock_geo.angle_of_incidence_target = jnp.array(5.0)
    mock_geo.R_target = None
    mock_geo.R_OMP = None
    mock_geo.B_pol_OMP = None
    mock_geo.diverted = jnp.array(True)

    edge_params = self._create_edge_params(
        # connection_length_target is None by default in _create_edge_params
        toroidal_flux_expansion=1.5,
    )

    with self.assertRaisesRegex(
        RuntimeError, "Geometry parameter 'connection_length_target' is invalid"
    ):
      extended_lengyel_model._resolve_geometric_parameters(
          mock_geo, self.mock_core_profiles, edge_params, diverted=True
      )


class ExtendedLengyelModelCouplingTest(sim_test_case.SimTestCase):

  def test_edge_model_coupling_smoke(self):
    """Smoke test to ensure edge model is called and outputs are stored."""
    torax_config = self._get_torax_config(
        'test_iterhybrid_predictor_corrector.py'
    )
    torax_config.update_fields({
        'plasma_composition.impurity': {
            'impurity_mode': 'n_e_ratios',
            'species': {'Ne': 0.01},
        },
        'edge': {
            'model_name': 'extended_lengyel',
            'computation_mode': extended_lengyel_enums.ComputationMode.FORWARD,
            'fixed_impurity_concentrations': {'Ne': 5e-2},
            'enrichment_factor': {'Ne': 1.0},
            'connection_length_target': 50.0,
            'connection_length_divertor': 10.0,
            'toroidal_flux_expansion': 4.0,
            'angle_of_incidence_target': 3.0,
            'ratio_bpol_omp_to_bpol_avg': 4.0 / 3.0,
            'use_enrichment_model': False,
            'diverted': True,
        },
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
      self.assertGreater(edge_output.T_e_separatrix, 1e-2)

  @parameterized.named_parameters(
      ('updates_enabled', True, 2.0),
      ('updates_disabled', False, 2.0),
      ('non_unity_ratio', True, 3.0),
  )
  def test_temperature_boundary_condition_updates(
      self, update_temperatures, ion_to_electron_ratio
  ):
    """Tests that boundary conditions are correctly updated based on edge model."""

    initial_Te_bc = 0.5
    initial_Ti_bc = 0.5

    torax_config = self._get_torax_config(
        'test_iterhybrid_predictor_corrector.py'
    )
    torax_config.update_fields({
        'plasma_composition.impurity': {
            'impurity_mode': 'n_e_ratios',
            'species': {'Ne': 0.01},
        },
        'profile_conditions.T_e_right_bc': initial_Te_bc,
        'profile_conditions.T_i_right_bc': initial_Ti_bc,
        'edge': {
            'model_name': 'extended_lengyel',
            'computation_mode': extended_lengyel_enums.ComputationMode.FORWARD,
            'fixed_impurity_concentrations': {'Ne': 5e-2},
            'enrichment_factor': {'Ne': 1.0},
            'connection_length_target': 50.0,
            'connection_length_divertor': 10.0,
            'toroidal_flux_expansion': 4.0,
            'angle_of_incidence_target': 3.0,
            'ratio_bpol_omp_to_bpol_avg': 4.0 / 3.0,
            # Test parameters
            'update_temperatures': update_temperatures,
            'T_i_T_e_ratio_target': ion_to_electron_ratio,
            'use_enrichment_model': False,
            'diverted': True,
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
      expected_Te_bc = final_edge_output.T_e_separatrix
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

  def test_inverse_mode_updates_impurities(self):
    """Tests that INVERSE mode updates core impurity ratios with shape scaling."""
    # Initial profile has shape: 0.01 at center, 0.02 at edge.
    initial_N_ratio_axis = 0.001
    initial_N_ratio_lcfs = 0.002
    initial_He3_ratio_axis = 0.01
    initial_He3_ratio_lcfs = 0.03
    # Dictionary defining linear profile in rho_norm
    initial_N_ratio_dict = {
        0: initial_N_ratio_axis,
        1: initial_N_ratio_lcfs,
    }

    He3_ratio_dict = {
        0: initial_He3_ratio_axis,
        1: initial_He3_ratio_lcfs,
    }

    enrichment = 5.0

    torax_config = self._get_torax_config(
        'test_iterhybrid_predictor_corrector.py'
    )

    # Use n_e_ratios mode for impurities
    torax_config.update_fields({
        'plasma_composition': {
            'main_ion': 'D',
            'impurity': {
                'impurity_mode': 'n_e_ratios',
                'species': {
                    'N': initial_N_ratio_dict,
                    'He3': He3_ratio_dict,
                },
            },
        },
        'sources.impurity_radiation': {
            'model_name': 'mavrin_fit',
        },
        'edge': {
            'model_name': 'extended_lengyel',
            'computation_mode': extended_lengyel_enums.ComputationMode.INVERSE,
            'T_e_target': 2.5,
            'seed_impurity_weights': {'N': 1.0},
            'fixed_impurity_concentrations': {'He3': 0.06},
            'enrichment_factor': {'N': enrichment, 'He3': 1.0},
            'connection_length_target': 60.0,
            'connection_length_divertor': 15.0,
            'toroidal_flux_expansion': 4.0,
            'angle_of_incidence_target': 3.0,
            'ratio_bpol_omp_to_bpol_avg': 4.0 / 3.0,
            'use_enrichment_model': False,
            'diverted': True,
        },
    })

    # Run for a few steps
    torax_config.update_fields({
        'numerics.t_final': (
            torax_config.numerics.t_initial
            + 3 * torax_config.numerics.fixed_dt.value[0]
        )
    })

    xr_outputs, state_history = run_simulation.run_simulation(torax_config)

    final_edge_output = state_history._edge_outputs[-1]
    N_edge_conc = final_edge_output.seed_impurity_concentrations['N']

    # Check that core impurity ratios were updated correctly. Only N is updated,
    # He3 is fixed. Impurity ratios are on the cell grid.
    n_e_final = xr_outputs.profiles.n_e.values[-1, :]
    calculated_N_ratio_final = (
        xr_outputs.profiles.n_impurity_species.sel(impurity_symbol='N').values[
            -1, :
        ]
        / n_e_final[1:-1]
    )
    calculated_He3_ratio_final = (
        xr_outputs.profiles.n_impurity_species.sel(
            impurity_symbol='He3'
        ).values[-1, :]
        / n_e_final[1:-1]
    )

    N_ratio_face = np.linspace(initial_N_ratio_axis, initial_N_ratio_lcfs, 26)
    N_ratio_cell = 0.5 * (N_ratio_face[:-1] + N_ratio_face[1:])
    He3_ratio_face = np.linspace(
        initial_He3_ratio_axis, initial_He3_ratio_lcfs, 26
    )
    He3_ratio_cell = 0.5 * (He3_ratio_face[:-1] + He3_ratio_face[1:])

    # On cell grid, scaled from shape in runtime_params
    expected_N_ratio_final = (
        N_edge_conc / enrichment * N_ratio_cell / initial_N_ratio_lcfs
    )
    # Not expected to change
    expected_He3_ratio_final = He3_ratio_cell

    np.testing.assert_allclose(
        calculated_N_ratio_final, expected_N_ratio_final, rtol=1e-4
    )
    np.testing.assert_allclose(
        calculated_He3_ratio_final, expected_He3_ratio_final, rtol=1e-4
    )


class ExtendedLengyelEnrichmentFactorTest(sim_test_case.SimTestCase):

  def setUp(self):
    super().setUp()
    self._CONFIG_ENRICHMENT = 2.0
    self._N_E_VALUE = 3e18
    self._EDGE_NE_VALUE = 1.5e17

    self.torax_config = self._get_torax_config(
        'test_iterhybrid_predictor_corrector.py'
    )
    self.torax_config.update_fields({
        'plasma_composition.impurity': {
            'impurity_mode': 'n_e_ratios',
            'species': {'Ne': 0.01},
        },
        'profile_conditions.n_e_right_bc': self._N_E_VALUE,
        'profile_conditions.n_e': self._N_E_VALUE,
        'profile_conditions.n_e_nbar_is_fGW': False,
        'profile_conditions.n_e_right_bc_is_fGW': False,
        'profile_conditions.normalize_n_e_to_nbar': False,
        'numerics.evolve_density': False,
        'edge': {
            'model_name': 'extended_lengyel',
            'impurity_sot': (
                extended_lengyel_model.FixedImpuritySourceOfTruth.EDGE
            ),
            'computation_mode': extended_lengyel_enums.ComputationMode.FORWARD,
            'fixed_impurity_concentrations': {
                'Ne': self._EDGE_NE_VALUE / self._N_E_VALUE
            },
            'enrichment_factor': {'Ne': self._CONFIG_ENRICHMENT},
            'connection_length_target': 50.0,
            'connection_length_divertor': 10.0,
            'toroidal_flux_expansion': 4.0,
            'angle_of_incidence_target': 3.0,
            'ratio_bpol_omp_to_bpol_avg': 4.0 / 3.0,
            'diverted': True,
            # 'use_enrichment_model' and 'diverted' are overridden in tests.
        },
    })

    # Run for just a few steps
    self.torax_config.update_fields({
        'numerics.t_final': (
            self.torax_config.numerics.t_initial
            + 5 * self.torax_config.numerics.fixed_dt.value[0]
        )
    })

  def test_use_model_diverted(self):
    """Enrichment from Kallenbach model for diverted geometry."""
    self.torax_config.update_fields({
        'edge.use_enrichment_model': True,
        'edge.diverted': True,
    })
    outputs, _ = run_simulation.run_simulation(self.torax_config)
    calculated_enrichment = outputs.edge.calculated_enrichment.values[0]
    core_impurity_value = outputs.profiles.n_impurity.values[:, -1]

    for i in range(len(calculated_enrichment)):
      if i == 0:
        continue  # Skip the first time step due to initialization effects.
      # Should be Kallenbach. We don't know exact value but it is > 1.0.
      self.assertGreater(calculated_enrichment[i], 1.0)
      # And not the config value as this output is physics-derived.
      self.assertNotAlmostEqual(
          calculated_enrichment[i], self._CONFIG_ENRICHMENT
      )
      np.testing.assert_allclose(
          core_impurity_value[i],
          self._EDGE_NE_VALUE / calculated_enrichment[i],
          rtol=1e-5,
      )

  def test_no_model_diverted(self):
    """Enrichment from config for diverted geometry."""
    self.torax_config.update_fields({
        'edge.use_enrichment_model': False,
        'edge.diverted': True,
    })
    outputs, _ = run_simulation.run_simulation(self.torax_config)
    core_impurity_value = outputs.profiles.n_impurity.values[:, -1]
    for i in range(len(core_impurity_value)):
      if i == 0:
        continue
      # Takes enrichment factor from config.
      np.testing.assert_allclose(
          core_impurity_value[i],
          self._EDGE_NE_VALUE / self._CONFIG_ENRICHMENT,
          rtol=1e-5,
      )

  def test_use_model_limited(self):
    """Enrichment is 1.0 for limited geometry when using model."""
    self.torax_config.update_fields({
        'edge.use_enrichment_model': True,
        'edge.diverted': False,
    })
    outputs, _ = run_simulation.run_simulation(self.torax_config)
    calculated_enrichment = outputs.edge.calculated_enrichment.values[0]
    core_impurity_value = outputs.profiles.n_impurity.values[:, -1]
    for i in range(len(calculated_enrichment)):
      if i == 0:
        continue
      # Physics model sets 1.0 for limited
      self.assertEqual(calculated_enrichment[i], 1.0)
      np.testing.assert_allclose(
          core_impurity_value[i],
          self._EDGE_NE_VALUE,
          rtol=1e-5,
      )

  def test_no_model_limited(self):
    """Enrichment from config for limited geometry."""
    self.torax_config.update_fields({
        'edge.use_enrichment_model': False,
        'edge.diverted': False,
    })
    outputs, _ = run_simulation.run_simulation(self.torax_config)
    core_impurity_value = outputs.profiles.n_impurity.values[:, -1]
    for i in range(len(core_impurity_value)):
      if i == 0:
        continue
      # Takes enrichment factor from config.
      np.testing.assert_allclose(
          core_impurity_value[i],
          self._EDGE_NE_VALUE / self._CONFIG_ENRICHMENT,
          rtol=1e-5,
      )


if __name__ == '__main__':
  absltest.main()

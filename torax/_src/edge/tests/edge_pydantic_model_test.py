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
import dataclasses
from absl.testing import absltest
import numpy as np
from torax._src.edge import extended_lengyel_defaults
from torax._src.edge import extended_lengyel_enums
from torax._src.edge import extended_lengyel_solvers
from torax._src.edge import extended_lengyel_standalone
from torax._src.edge import pydantic_model
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config


# pylint: disable=invalid-name
class ExtendedLengyelPydanticModelTest(absltest.TestCase):

  def test_extended_lengyel_defaults(self):
    """Checks default values for the extended lengyel config."""
    config = pydantic_model.ExtendedLengyelConfig()
    self.assertEqual(config.model_name, 'extended_lengyel')
    self.assertEqual(
        config.computation_mode, extended_lengyel_enums.ComputationMode.FORWARD
    )
    self.assertEqual(
        config.solver_mode, extended_lengyel_enums.SolverMode.HYBRID
    )

  def test_fixed_step_iterations_default(self):
    # Default solver_mode is HYBRID
    config_hybrid = pydantic_model.ExtendedLengyelConfig()
    self.assertEqual(
        config_hybrid.fixed_step_iterations,
        extended_lengyel_defaults.HYBRID_FIXED_STEP_ITERATIONS,
    )
    # Explicitly set solver_mode to FIXED_STEP
    config_fixed = pydantic_model.ExtendedLengyelConfig(
        solver_mode=extended_lengyel_enums.SolverMode.FIXED_STEP
    )
    self.assertEqual(
        config_fixed.fixed_step_iterations,
        extended_lengyel_defaults.FIXED_STEP_ITERATIONS,
    )
    # User can override
    config_override = pydantic_model.ExtendedLengyelConfig(
        fixed_step_iterations=100
    )
    self.assertEqual(config_override.fixed_step_iterations, 100)

  def test_torax_config_integration(self):
    """Ensures ToraxConfig can parse the new edge field."""
    config_dict = default_configs.get_default_config_dict()
    config_dict['geometry'] = {
        'geometry_type': 'chease',
        'geometry_file': 'ITER_hybrid_citrin_equil_cheasedata.mat2cols',
    }
    config_dict['edge'] = {
        'model_name': 'extended_lengyel',
        'computation_mode': 'inverse',
        'target_electron_temp': 2.34,
        'seed_impurity_weights': {'N': 1.0, 'Ar': 0.05},
        'enrichment_factor': {'N': 1.0, 'Ar': 1.0,},
    }
    torax_config = model_config.ToraxConfig.from_dict(config_dict)
    self.assertIsNotNone(torax_config.edge)
    self.assertIsInstance(
        torax_config.edge, pydantic_model.ExtendedLengyelConfig
    )
    # Extra assert to make pytype happy.
    assert isinstance(torax_config.edge, pydantic_model.ExtendedLengyelConfig)
    self.assertEqual(
        torax_config.edge.computation_mode,
        extended_lengyel_enums.ComputationMode.INVERSE,
    )

  def test_torax_config_no_edge(self):
    """Ensures ToraxConfig works fine without an edge config."""
    config_dict = default_configs.get_default_config_dict()
    # No 'edge' key in config_dict
    torax_config = model_config.ToraxConfig.from_dict(config_dict)
    self.assertIsNone(torax_config.edge)

  def test_enrichment_factor_key_validation_valid(self):
    pydantic_model.ExtendedLengyelConfig(
        computation_mode=extended_lengyel_enums.ComputationMode.INVERSE,
        target_electron_temp=10.0,
        seed_impurity_weights={'N': 1.0},
        fixed_impurity_concentrations={'He': 0.01},
        enrichment_factor={'N': 2.0, 'He': 1.5},
    )

  def test_enrichment_factor_key_validation_missing_key(self):
    with self.assertRaisesRegex(
        ValueError,
        'enrichment_factor is missing keys',
    ):
      pydantic_model.ExtendedLengyelConfig(
          computation_mode=extended_lengyel_enums.ComputationMode.INVERSE,
          target_electron_temp=10.0,
          seed_impurity_weights={'N': 1.0},
          fixed_impurity_concentrations={'He': 0.01},
          enrichment_factor={'N': 2.0},
      )

  def test_enrichment_factor_key_validation_extra_key(self):
    with self.assertRaisesRegex(
        ValueError,
        'enrichment_factor has extra keys',
    ):
      pydantic_model.ExtendedLengyelConfig(
          computation_mode=extended_lengyel_enums.ComputationMode.INVERSE,
          target_electron_temp=10.0,
          seed_impurity_weights={'N': 1.0},
          fixed_impurity_concentrations={'He': 0.01},
          enrichment_factor={'N': 2.0, 'He': 1.5, 'Ar': 2.0},
      )

  def test_computation_mode_forward_valid_config(self):
    pydantic_model.ExtendedLengyelConfig(
        computation_mode='forward',
        target_electron_temp=None,
        seed_impurity_weights={},
    )

  def test_computation_mode_forward_raises_on_target_temp(self):
    with self.assertRaisesRegex(
        ValueError,
        'target_electron_temp must not be provided for forward computation'
        ' mode.',
    ):
      pydantic_model.ExtendedLengyelConfig(
          computation_mode='forward',
          target_electron_temp=10.0,
      )

  def test_computation_mode_forward_raises_on_seed_impurities(self):
    with self.assertRaisesRegex(
        ValueError,
        'seed_impurity_weights must not be provided for forward computation'
        ' mode.',
    ):
      pydantic_model.ExtendedLengyelConfig(
          computation_mode='forward',
          seed_impurity_weights={'N': 1.0},
          enrichment_factor={'N': 1.0},
      )

  def test_computation_mode_inverse_valid_config(self):
    # This should not raise an error.
    pydantic_model.ExtendedLengyelConfig(
        computation_mode='inverse',
        target_electron_temp=10.0,
        seed_impurity_weights={'N': 1.0},
        enrichment_factor={'N': 1.0},
    )

  def test_computation_mode_inverse_raises_on_missing_target_temp(self):
    with self.assertRaisesRegex(
        ValueError,
        'target_electron_temp must be provided for inverse computation mode.',
    ):
      pydantic_model.ExtendedLengyelConfig(
          computation_mode='inverse',
          target_electron_temp=None,
          seed_impurity_weights={'N': 1.0},
          enrichment_factor={'N': 1.0},
      )

  def test_computation_mode_inverse_raises_on_empty_seed_impurities(self):
    with self.assertRaisesRegex(
        ValueError,
        'seed_impurity_weights must be provided for inverse computation mode.',
    ):
      pydantic_model.ExtendedLengyelConfig(
          computation_mode='inverse',
          target_electron_temp=10.0,
          seed_impurity_weights={},
          enrichment_factor={},
      )

  def test_computation_mode_inverse_raises_on_none_seed_impurities(self):
    with self.assertRaisesRegex(
        ValueError,
        'seed_impurity_weights must be provided for inverse computation mode.',
    ):
      pydantic_model.ExtendedLengyelConfig(
          computation_mode='inverse',
          target_electron_temp=10.0,
          seed_impurity_weights=None,
          enrichment_factor={},
      )

  def test_optional_params_are_none_in_runtime_params_if_not_provided(self):
    config = pydantic_model.ExtendedLengyelConfig()
    runtime_params = config.build_runtime_params(t=0.0)
    self.assertIsNone(runtime_params.parallel_connection_length)
    self.assertIsNone(runtime_params.divertor_parallel_length)
    self.assertIsNone(runtime_params.target_electron_temp)

  def test_run_standalone_from_pydantic_config(self):
    """Tests that standalone can be run from a pydantic config."""
    # This test is based on
    # extended_lengyel_standalone_test.test_run_extended_lengyel_model_inverse_mode_fixed_step
    # --- Expected output values ---
    # Reference values from running the reference case in:
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
    # Inputs that would be configured in the TORAX config.
    config_dict_inputs = {
        'model_name': 'extended_lengyel',
        'target_electron_temp': 2.34,
        'seed_impurity_weights': {'N': 1.0, 'Ar': 0.05},
        'fixed_impurity_concentrations': {'He': 0.01},
        'parallel_connection_length': 20.0,
        'divertor_parallel_length': 5.0,
        'computation_mode': 'inverse',
        'solver_mode': 'fixed_step',
        'enrichment_factor': {'N': 1.0, 'Ar': 1.0, 'He': 1.0},
    }
    # Inputs that would come from the TORAX state at runtime.
    dynamic_inputs = {
        'power_crossing_separatrix': 5.5e6,
        'separatrix_electron_density': 3.3e19,
        'main_ion_charge': 1.0,
        'mean_ion_charge_state': 1.0,
        'magnetic_field_on_axis': 2.5,
        'plasma_current': 1.0e6,
        'major_radius': 1.65,
        'minor_radius': 0.5,
        'elongation_psi95': 1.6,
        'triangularity_psi95': 0.3,
        'average_ion_mass': 2.0,
    }
    # Build runtime params from config
    config = pydantic_model.ExtendedLengyelConfig.from_dict(config_dict_inputs)
    runtime_params = config.build_runtime_params(t=0.0)
    # Combine dynamic inputs and runtime params to call standalone function
    runtime_params_dict = dataclasses.asdict(runtime_params)
    runtime_params_dict.pop('enrichment_factor')
    runtime_params_dict.pop('update_temperatures')
    kwargs = {**dynamic_inputs, **runtime_params_dict}
    # Run the model
    outputs = extended_lengyel_standalone.run_extended_lengyel_standalone(
        **kwargs
    )
    # --- Assertions ---
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

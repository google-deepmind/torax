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
import re
from absl.testing import absltest
import numpy as np
import pydantic
from torax._src.edge.extended_lengyel import extended_lengyel_defaults
from torax._src.edge.extended_lengyel import extended_lengyel_enums
from torax._src.edge.extended_lengyel import extended_lengyel_solvers
from torax._src.edge.extended_lengyel import extended_lengyel_standalone
from torax._src.edge.extended_lengyel import pydantic_model
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config


# pylint: disable=invalid-name
class ExtendedLengyelPydanticModelTest(absltest.TestCase):

  def test_extended_lengyel_defaults(self):
    """Checks default values for the extended lengyel config."""
    config = pydantic_model.ExtendedLengyelConfig.from_dict({})
    self.assertEqual(config.model_name, 'extended_lengyel')
    self.assertEqual(
        config.computation_mode, extended_lengyel_enums.ComputationMode.FORWARD
    )
    self.assertEqual(
        config.solver_mode, extended_lengyel_enums.SolverMode.HYBRID
    )

  def test_initial_guess_config_defaults(self):
    """Checks default values for the initial guess config."""
    config = pydantic_model.ExtendedLengyelConfig.from_dict({})
    self.assertEqual(
        config.initial_guess.alpha_t.get_value(0.0),
        extended_lengyel_defaults.DEFAULT_ALPHA_T_INIT,
    )
    self.assertEqual(
        config.initial_guess.kappa_e.get_value(0.0),
        extended_lengyel_defaults.DEFAULT_KAPPA_E_INIT,
    )
    self.assertEqual(
        config.initial_guess.T_e_separatrix.get_value(0.0),
        extended_lengyel_defaults.DEFAULT_T_E_SEPARATRIX_INIT,
    )
    self.assertEqual(
        config.initial_guess.T_e_target.get_value(0.0),
        extended_lengyel_defaults.DEFAULT_T_E_TARGET_INIT_FORWARD,
    )
    self.assertEqual(
        config.initial_guess.c_z_prefactor.get_value(0.0),
        extended_lengyel_defaults.DEFAULT_C_Z_PREFACTOR_INIT,
    )
    self.assertTrue(config.initial_guess.use_previous_step_as_guess)

  def test_fixed_point_iterations_default(self):
    # Default solver_mode is HYBRID
    config_hybrid = pydantic_model.ExtendedLengyelConfig.from_dict({})
    self.assertEqual(
        config_hybrid.fixed_point_iterations,
        extended_lengyel_defaults.HYBRID_FIXED_POINT_ITERATIONS,
    )
    # Explicitly set solver_mode to FIXED_POINT
    config_fixed = pydantic_model.ExtendedLengyelConfig.from_dict({
        'solver_mode': extended_lengyel_enums.SolverMode.FIXED_POINT,
    })
    self.assertEqual(
        config_fixed.fixed_point_iterations,
        extended_lengyel_defaults.FIXED_POINT_ITERATIONS,
    )
    # User can override
    config_override = pydantic_model.ExtendedLengyelConfig.from_dict({
        'fixed_point_iterations': 100,
    })
    self.assertEqual(config_override.fixed_point_iterations, 100)

  def test_fixed_point_relaxation_factor(self):
    config = pydantic_model.ExtendedLengyelConfig.from_dict({})
    self.assertEqual(
        config.fixed_point_relaxation_factor,
        extended_lengyel_defaults.FIXED_POINT_RELAXATION_FACTOR,
    )
    config_override = pydantic_model.ExtendedLengyelConfig.from_dict(
        {'fixed_point_relaxation_factor': 0.6}
    )
    self.assertEqual(config_override.fixed_point_relaxation_factor, 0.6)
    with self.assertRaises(ValueError):
      pydantic_model.ExtendedLengyelConfig.from_dict(
          {'fixed_point_relaxation_factor': 0.0}
      )
    with self.assertRaises(ValueError):
      pydantic_model.ExtendedLengyelConfig.from_dict(
          {'fixed_point_relaxation_factor': 1.1}
      )

  def test_torax_config_integration(self):
    """Ensures ToraxConfig can parse the new edge field."""
    config_dict = default_configs.get_default_config_dict()
    config_dict['geometry'] = {
        'geometry_type': 'chease',
        'geometry_file': 'iterhybrid.mat2cols',
    }
    config_dict['edge'] = {
        'model_name': 'extended_lengyel',
        'computation_mode': 'inverse',
        'T_e_target': 2.34,
        'seed_impurity_weights': {'N': 1.0, 'Ar': 0.05},
        'use_enrichment_model': False,
        'enrichment_factor': {
            'N': 1.0,
            'Ar': 1.0,
        },
        'diverted': True,
        'initial_guess': {'alpha_t': 0.2},
    }
    config_dict['plasma_composition'] = {
        'impurity': {
            'impurity_mode': 'n_e_ratios',
            'species': {'N': 1e-2, 'Ar': 1e-3},
        }
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
    assert torax_config.edge.initial_guess.alpha_t is not None
    self.assertEqual(
        torax_config.edge.initial_guess.alpha_t.get_value(0.0), 0.2
    )

  def test_torax_config_no_edge(self):
    """Ensures ToraxConfig works fine without an edge config."""
    config_dict = default_configs.get_default_config_dict()
    # No 'edge' key in config_dict
    torax_config = model_config.ToraxConfig.from_dict(config_dict)
    self.assertIsNone(torax_config.edge)

  def test_enrichment_factor_key_validation_valid(self):
    pydantic_model.ExtendedLengyelConfig.from_dict({
        'use_enrichment_model': False,
        'computation_mode': extended_lengyel_enums.ComputationMode.INVERSE,
        'T_e_target': 10.0,
        'seed_impurity_weights': {'N': 1.0},
        'fixed_impurity_concentrations': {'He4': 0.01},
        'enrichment_factor': {'N': 2.0, 'He4': 1.5},
    })

  def test_enrichment_factor_key_validation_missing_key(self):
    with self.assertRaisesRegex(
        ValueError,
        'enrichment_factor is missing keys',
    ):
      pydantic_model.ExtendedLengyelConfig.from_dict({
          'use_enrichment_model': False,
          'computation_mode': extended_lengyel_enums.ComputationMode.INVERSE,
          'T_e_target': 10.0,
          'seed_impurity_weights': {'N': 1.0},
          'fixed_impurity_concentrations': {'He4': 0.01},
          'enrichment_factor': {'N': 2.0},
      })

  def test_enrichment_factor_key_validation_extra_key(self):
    with self.assertRaisesRegex(
        ValueError,
        'enrichment_factor has extra keys',
    ):
      pydantic_model.ExtendedLengyelConfig.from_dict({
          'use_enrichment_model': False,
          'computation_mode': extended_lengyel_enums.ComputationMode.INVERSE,
          'T_e_target': 10.0,
          'seed_impurity_weights': {'N': 1.0},
          'fixed_impurity_concentrations': {'He4': 0.01},
          'enrichment_factor': {'N': 2.0, 'He4': 1.5, 'Ar': 2.0},
      })

  def test_computation_mode_forward_valid_config(self):
    pydantic_model.ExtendedLengyelConfig.from_dict({
        'computation_mode': 'forward',
        'T_e_target': None,
        'seed_impurity_weights': {},
    })

  def test_computation_mode_forward_raises_on_target_temp(self):
    with self.assertRaisesRegex(
        ValueError,
        'T_e_target must not be provided for forward computation mode.',
    ):
      pydantic_model.ExtendedLengyelConfig.from_dict({
          'computation_mode': 'forward',
          'T_e_target': 10.0,
      })

  def test_computation_mode_forward_raises_on_seed_impurities(self):
    with self.assertRaisesRegex(
        ValueError,
        'seed_impurity_weights must not be provided for forward computation'
        ' mode.',
    ):
      pydantic_model.ExtendedLengyelConfig.from_dict({
          'use_enrichment_model': False,
          'computation_mode': 'forward',
          'seed_impurity_weights': {'N': 1.0},
          'enrichment_factor': {'N': 1.0},
      })

  def test_computation_mode_inverse_valid_config(self):
    # This should not raise an error.
    pydantic_model.ExtendedLengyelConfig.from_dict({
        'use_enrichment_model': False,
        'computation_mode': 'inverse',
        'T_e_target': 10.0,
        'seed_impurity_weights': {'N': 1.0},
        'enrichment_factor': {'N': 1.0},
    })

  def test_computation_mode_inverse_raises_on_missing_target_temp(self):
    with self.assertRaisesRegex(
        ValueError,
        'T_e_target must be provided for inverse computation mode.',
    ):
      pydantic_model.ExtendedLengyelConfig.from_dict({
          'use_enrichment_model': False,
          'computation_mode': 'inverse',
          'T_e_target': None,
          'seed_impurity_weights': {'N': 1.0},
          'enrichment_factor': {'N': 1.0},
      })

  def test_computation_mode_inverse_raises_on_empty_seed_impurities(self):
    with self.assertRaisesRegex(
        ValueError,
        'seed_impurity_weights must be provided for inverse computation mode.',
    ):
      pydantic_model.ExtendedLengyelConfig.from_dict({
          'use_enrichment_model': False,
          'computation_mode': 'inverse',
          'T_e_target': 10.0,
          'seed_impurity_weights': {},
          'enrichment_factor': {},
      })

  def test_computation_mode_inverse_raises_on_zero_total_seed_weight(self):
    cases = {
        'constant_zero': {'N': 0.0, 'Ar': 0.0},
        'near_zero': {'N': 1e-9, 'Ar': 0.0},
        'zero_at_one_time': {'N': {0.0: 1.0, 1.0: 0.0}, 'Ar': 0.0},
        'zero_over_interval_between_species': {
            'N': {0.0: 1.0, 1.0: 0.0},
            'Ar': {0.0: 0.0, 2.0: 0.0, 3.0: 1.0},
        },
    }
    for name, seed_impurity_weights in cases.items():
      with self.subTest(name):
        with self.assertRaisesRegex(
            ValueError,
            'The sum of seed_impurity_weights must be positive',
        ):
          pydantic_model.ExtendedLengyelConfig.from_dict({
              'use_enrichment_model': False,
              'computation_mode': 'inverse',
              'T_e_target': 10.0,
              'seed_impurity_weights': seed_impurity_weights,
              'enrichment_factor': {'N': 1.0, 'Ar': 1.0},
          })

  def test_computation_mode_inverse_raises_on_unsupported_seed_impurity(self):
    # W is a real species that is reachable from a full ToraxConfig, but has
    # no Mavrin fit. 'Xx' is a fake symbol that can never be supported.
    cases = {
        'only_unsupported': ({'W': 1.0}, ['W']),
        'mixed': ({'N': 1.0, 'W': 0.1}, ['W']),
        'fake_symbol': ({'Xx': 1.0}, ['Xx']),
    }
    for name, (seed_impurity_weights, unsupported) in cases.items():
      with self.subTest(name):
        with self.assertRaisesRegex(
            ValueError,
            'without a non-coronal Mavrin cooling rate fit.*'
            + re.escape(str(unsupported)),
        ):
          pydantic_model.ExtendedLengyelConfig.from_dict({
              'use_enrichment_model': False,
              'computation_mode': 'inverse',
              'T_e_target': 10.0,
              'seed_impurity_weights': seed_impurity_weights,
              'enrichment_factor': {key: 1.0 for key in seed_impurity_weights},
          })

  def test_computation_mode_inverse_allows_seed_species_handover(self):
    # The total seeded weight stays positive while species are swapped.
    pydantic_model.ExtendedLengyelConfig.from_dict({
        'use_enrichment_model': False,
        'computation_mode': 'inverse',
        'T_e_target': 10.0,
        'seed_impurity_weights': {
            'N': {0.0: 1.0, 1.0: 0.0},
            'Ar': {0.0: 0.0, 1.0: 1.0},
        },
        'enrichment_factor': {'N': 1.0, 'Ar': 1.0},
    })

  def test_computation_mode_inverse_raises_on_none_seed_impurities(self):
    with self.assertRaisesRegex(
        ValueError,
        'seed_impurity_weights must be provided for inverse computation mode.',
    ):
      pydantic_model.ExtendedLengyelConfig.from_dict({
          'use_enrichment_model': False,
          'computation_mode': 'inverse',
          'T_e_target': 10.0,
          'seed_impurity_weights': None,
          'enrichment_factor': {},
      })

  def test_warning_for_unused_enrichment_factor(self):
    with self.assertLogs(level='WARNING') as cm:
      pydantic_model.ExtendedLengyelConfig.from_dict({
          'use_enrichment_model': True,
          'enrichment_factor': {'N': 1.0},
          # Other required fields for validation to pass
          'computation_mode': 'inverse',
          'T_e_target': 10.0,
          'seed_impurity_weights': {'N': 1.0},
      })
    self.assertIn(
        'enrichment_factor is provided but use_enrichment_model is True',
        cm.output[0],
    )

  def test_raises_error_if_enrichment_factor_is_none_when_not_using_model(self):
    with self.assertRaisesRegex(
        ValueError,
        'enrichment_factor must be provided when use_enrichment_model is'
        ' False.',
    ):
      pydantic_model.ExtendedLengyelConfig.from_dict({
          'use_enrichment_model': False,
          'enrichment_factor': None,
          # Other required fields for validation to pass
          'computation_mode': 'inverse',
          'T_e_target': 10.0,
          'seed_impurity_weights': {'N': 1.0},
      })

  def test_build_runtime_params_with_enrichment_model(self):
    config = pydantic_model.ExtendedLengyelConfig.from_dict({
        'use_enrichment_model': True,
        'enrichment_factor': None,
        'enrichment_model_multiplier': 2.0,
        # Other required fields
        'computation_mode': 'inverse',
        'T_e_target': 10.0,
        'seed_impurity_weights': {'N': 1.0},
        'fixed_impurity_concentrations': {'He4': 0.01},
    })
    runtime_params = config.build_runtime_params(t=0.0)
    self.assertIsNone(runtime_params.enrichment_factor)

  def test_build_runtime_params_without_enrichment_model(self):
    config = pydantic_model.ExtendedLengyelConfig.from_dict({
        'use_enrichment_model': False,
        'enrichment_factor': {'N': 2.5, 'He4': 3.5},
        # Other required fields
        'computation_mode': 'inverse',
        'T_e_target': 10.0,
        'seed_impurity_weights': {'N': 1.0},
        'fixed_impurity_concentrations': {'He4': 0.01},
    })
    runtime_params = config.build_runtime_params(t=0.0)
    self.assertIsNotNone(runtime_params.enrichment_factor)
    assert runtime_params.enrichment_factor is not None
    self.assertIn('N', runtime_params.enrichment_factor)
    self.assertIn('He4', runtime_params.enrichment_factor)
    np.testing.assert_allclose(runtime_params.enrichment_factor['N'], 2.5)
    np.testing.assert_allclose(runtime_params.enrichment_factor['He4'], 3.5)

  def test_optional_params_are_none_in_runtime_params_if_not_provided(self):
    config = pydantic_model.ExtendedLengyelConfig.from_dict({})
    runtime_params = config.build_runtime_params(t=0.0)
    self.assertIsNone(runtime_params.connection_length_target)
    self.assertIsNone(runtime_params.connection_length_divertor)
    self.assertIsNone(runtime_params.T_e_target)

  def test_species_consistency_validation_failure(self):
    """Tests failure when species sets do not match between core and edge."""
    config_dict = default_configs.get_default_config_dict()
    config_dict['geometry'] = {
        'geometry_type': 'chease',
        'geometry_file': 'iterhybrid.mat2cols',
    }
    config_dict['edge'] = {
        'model_name': 'extended_lengyel',
        'fixed_impurity_concentrations': {'Ne': 0.01},
        'use_enrichment_model': False,
        'enrichment_factor': {'Ne': 1.0},
        'diverted': True,
    }
    config_dict['plasma_composition']['impurity'] = {
        'impurity_mode': 'n_e_ratios',
        'species': {'Ar': 0.01},  # Ar vs Ne mismatch
    }
    with self.assertRaisesRegex(
        ValueError,
        'Mismatch between core plasma composition impurities and edge'
        ' impurities',
    ):
      model_config.ToraxConfig.from_dict(config_dict)

  def test_species_overlap_validation_failure(self):
    """Tests failure when seeded and fixed impurities overlap."""
    config_dict = default_configs.get_default_config_dict()
    config_dict['geometry'] = {
        'geometry_type': 'chease',
        'geometry_file': 'iterhybrid.mat2cols',
    }
    config_dict['edge'] = {
        'model_name': 'extended_lengyel',
        'computation_mode': 'inverse',
        'T_e_target': 10.0,
        'fixed_impurity_concentrations': {'Ne': 0.01},
        'seed_impurity_weights': {'Ne': 1.0},  # Ne is in both
        'use_enrichment_model': False,
        'enrichment_factor': {'Ne': 1.0},
        'diverted': True,
    }
    config_dict['plasma_composition']['impurity'] = {
        'impurity_mode': 'n_e_ratios',
        'species': {'Ne': 0.01},
    }
    with self.assertRaisesRegex(
        ValueError, 'Edge fixed and seeded impurities must be disjoint'
    ):
      model_config.ToraxConfig.from_dict(config_dict)

  def test_run_standalone_from_pydantic_config(self):
    """Tests that standalone can be run from a pydantic config."""
    # This test is based on
    # extended_lengyel_standalone_test.test_run_extended_lengyel_model_inverse_mode_fixed_point
    # --- Expected output values ---
    # Reference values from running the reference case in:
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
    # Inputs that would be configured in the TORAX config.
    config_dict_inputs = {
        'model_name': 'extended_lengyel',
        'T_e_target': 2.34,
        'seed_impurity_weights': {'N': 1.0, 'Ar': 0.05},
        'fixed_impurity_concentrations': {'He': 0.01},
        'connection_length_target': 20.0,
        'connection_length_divertor': 5.0,
        'angle_of_incidence_target': 3.0,
        'toroidal_flux_expansion': 1.0,
        'ratio_bpol_omp_to_bpol_avg': 4.0 / 3.0,
        'computation_mode': 'inverse',
        'solver_mode': 'fixed_point',
        'use_enrichment_model': False,
        'enrichment_factor': {'N': 1.0, 'Ar': 1.0, 'He': 1.0},
        'diverted': True,
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
    runtime_params_dict.pop('update_electron_temperature')
    runtime_params_dict.pop('update_ion_temperature')
    runtime_params_dict.pop('update_electron_density')
    runtime_params_dict.pop('update_impurities')
    runtime_params_dict.pop('use_enrichment_model')
    runtime_params_dict.pop('impurity_sot')
    # Remove initial_guess as it's not in the standalone args map
    # directly in this test setup
    runtime_params_dict.pop('initial_guess')
    kwargs = {**dynamic_inputs, **runtime_params_dict}
    # Run the model
    # We pass None for initial_guess since the standalone runner logic
    # handles defaults if None, and we haven't constructed a valid JAX
    # InitialGuess object in this test context easily without mocking more.
    # The pydantic test focuses on config parsing.
    outputs = extended_lengyel_standalone.run_extended_lengyel_standalone(
        initial_guess=None, **kwargs
    )
    # --- Assertions ---
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
        np.testing.assert_allclose(getattr(outputs, key), value, rtol=_RTOL)  # pyrefly: ignore[no-matching-overload]

  def test_multistart_num_guesses_validation(self):
    with self.assertRaisesRegex(
        ValueError,
        'greater than or equal to 2',
    ):
      pydantic_model.ExtendedLengyelConfig.from_dict({
          'multistart_num_guesses': 1,
      })

  def test_update_electron_density_default_and_explicit_false_valid(self):
    with self.subTest('default'):
      config_default = pydantic_model.ExtendedLengyelConfig.from_dict({})
      self.assertFalse(config_default.update_electron_density.value.any())

    with self.subTest('explicit_false'):
      config_explicit_false = pydantic_model.ExtendedLengyelConfig.from_dict({
          'update_electron_density': False,
      })
      self.assertFalse(
          config_explicit_false.update_electron_density.value.any()
      )

  def test_update_electron_density_time_varying_false_valid(self):
    config_time_varying_false = pydantic_model.ExtendedLengyelConfig.from_dict({
        'update_electron_density': {0.0: False, 1.0: False},
    })
    self.assertFalse(
        config_time_varying_false.update_electron_density.value.any()
    )

  def test_update_electron_density_scalar_and_time_varying_true_raises(self):
    with self.subTest('scalar_true'):
      with self.assertRaisesRegex(
          pydantic.ValidationError,
          "'update_electron_density' cannot be True because"
          " 'extended_lengyel' does not support the 'electron_density'"
          ' boundary condition.',
      ):
        pydantic_model.ExtendedLengyelConfig.from_dict({
            'update_electron_density': True,
        })

    with self.subTest('time_varying_true'):
      with self.assertRaisesRegex(
          pydantic.ValidationError,
          "'update_electron_density' cannot be True because"
          " 'extended_lengyel' does not support the 'electron_density'"
          ' boundary condition.',
      ):
        pydantic_model.ExtendedLengyelConfig.from_dict({
            'update_electron_density': {0.0: False, 1.0: True},
        })


if __name__ == '__main__':
  absltest.main()

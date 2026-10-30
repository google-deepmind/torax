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

import copy
import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from torax._src import constants
from torax._src import state
from torax._src.core_profiles import (
    solver_profiles as solver_profiles_lib,
)
from torax._src.core_profiles.plasma_composition import electron_density_ratios
from torax._src.core_profiles.plasma_composition import impurity_fractions
from torax._src.core_profiles.plasma_composition import plasma_composition
from torax._src.fvm import cell_variable
from torax._src.geometry import circular_geometry
from torax._src.orchestration import run_simulation
from torax._src.output_tools import output_grid_context
from torax._src.output_tools import output_keys
from torax._src.physics import fast_ion as fast_ion_lib
from torax._src.test_utils import core_profile_helpers
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import config
from torax._src.transport_model import transport_coeffs as transport_coeffs_lib

# pylint: disable=invalid-name


class InitialStatesTest(parameterized.TestCase):

  def test_initial_boundary_condition_from_time_dependent_params(self):
    """Tests that the initial boundary conditions are set from the config."""
    config_dict = default_configs.get_default_config_dict()
    # Boundary conditions can be time-dependent, but when creating the initial
    # core profiles, we want to grab the boundary condition params at time 0.
    config_dict = default_configs.get_default_config_dict()
    config_dict['profile_conditions'] = {
        'I_p': 15e6,
        'I_p_source': 'profile_conditions',
        'initial_psi_source': 'j',
        'T_i_right_bc': 27.7,
        'T_e_right_bc': {0.0: 42.0, 1.0: 0.001},
        'n_e_right_bc': ({0.0: 0.1e20, 1.0: 2.0e20}, 'step'),
        'normalize_n_e_to_nbar': False,
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    core_profiles, _, _ = core_profile_helpers.make_initial_inputs(torax_config)
    self.assertIsNotNone(core_profiles.kinetic.T_i.right_face_constraint)
    np.testing.assert_allclose(
        core_profiles.kinetic.T_i.right_face_constraint, 27.7
    )
    self.assertIsNotNone(core_profiles.kinetic.T_e.right_face_constraint)
    np.testing.assert_allclose(
        core_profiles.kinetic.T_e.right_face_constraint, 42.0
    )
    self.assertIsNotNone(core_profiles.kinetic.n_e.right_face_constraint)
    np.testing.assert_allclose(
        core_profiles.kinetic.n_e.right_face_constraint, 0.1e20
    )

  def test_core_profiles_quasineutrality_check(self):
    torax_config = config.ToraxConfig.from_dict(
        default_configs.get_default_config_dict()
    )
    core_profiles, _, _ = core_profile_helpers.make_initial_inputs(torax_config)
    assert core_profiles.quasineutrality_satisfied()
    core_profiles = dataclasses.replace(
        core_profiles,
        ions=dataclasses.replace(
            core_profiles.ions,
            Z_i=core_profiles.ions.Z_i * 2.0,
        ),
    )
    assert not core_profiles.quasineutrality_satisfied()

  def test_core_profiles_negative_values_check(self):
    geo = circular_geometry.CircularConfig().build_geometry()
    core_profiles = core_profile_helpers.make_zero_core_profiles(geo)
    with self.subTest('no negative values'):
      self.assertFalse(core_profiles.negative_temperature_or_density())
    with self.subTest('negative T_i triggers'):
      new_core_profiles = dataclasses.replace(
          core_profiles,
          kinetic=dataclasses.replace(
              core_profiles.kinetic,
              T_i=dataclasses.replace(
                  core_profiles.kinetic.T_i,
                  value=jnp.full_like(core_profiles.kinetic.T_i.value, -1.0),
              ),
          ),
      )
      self.assertTrue(new_core_profiles.negative_temperature_or_density())
    with self.subTest('negative psi does not trigger'):
      new_core_profiles = dataclasses.replace(
          core_profiles,
          flux=dataclasses.replace(
              core_profiles.flux,
              psi=dataclasses.replace(
                  core_profiles.flux.psi,
                  value=jnp.full_like(core_profiles.flux.psi.value, -1.0),
              ),
          ),
      )
      self.assertFalse(new_core_profiles.negative_temperature_or_density())

  def test_get_species_concentration(self):
    config_dict = default_configs.get_default_config_dict()
    config_dict['plasma_composition'] = {
        'main_ion': {'D': 0.48, 'T': 0.48, 'He3': 0.04},
        'impurity': {
            'impurity_mode': 'n_e_ratios',
            'species': {'Ne': 0.01},
        },
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    core_profiles, _, geo = core_profile_helpers.make_initial_inputs(
        torax_config
    )

    with self.subTest('main ion species'):
      he3_conc = core_profiles.get_species_concentration('He3')
      self.assertEqual(he3_conc.shape, geo.rho_norm.shape)
      np.testing.assert_allclose(
          he3_conc,
          core_profiles.kinetic.n_i.value
          * 0.04
          / core_profiles.kinetic.n_e.value,
      )

    with self.subTest('impurity species'):
      ne_conc = core_profiles.get_species_concentration('Ne')
      self.assertEqual(ne_conc.shape, geo.rho_norm.shape)
      np.testing.assert_allclose(ne_conc, 0.01)

    with self.subTest('unknown species raises ValueError'):
      with self.assertRaisesRegex(ValueError, 'Species W not found'):
        core_profiles.get_species_concentration('W')


class CoreProfilesTemperatureCheckTest(parameterized.TestCase):
  """Tests for the below_minimum_temperature method in CoreProfiles."""

  def setUp(self):
    super().setUp()
    self.geo = circular_geometry.CircularConfig(n_rho=5).build_geometry()
    self.base_profiles = core_profile_helpers.make_zero_core_profiles(self.geo)

  @parameterized.named_parameters(
      ('all_above', 0.2, 0.2, 100.0, False),  # 0.2 keV = 200 eV > 100 eV
      ('te_below', 0.05, 0.2, 100.0, True),  # 0.05 keV = 50 eV < 100 eV
      ('ti_below', 0.2, 0.05, 100.0, True),
      ('both_below', 0.05, 0.05, 100.0, True),
      # 0.1 keV = 100 eV. Logic is strictly <, so this should pass (False).
      ('exact_boundary', 0.1, 0.1, 100.0, False),
  )
  def test_below_minimum_temperature(
      self, te_val, ti_val, threshold_ev, expected
  ):
    """Verifies below_minimum_temperature returns correct boolean flag."""
    # Create profiles with constant values across the radius
    core_profiles = dataclasses.replace(
        self.base_profiles,
        kinetic=dataclasses.replace(
            self.base_profiles.kinetic,
            T_e=core_profile_helpers.make_constant_core_profile(
                self.geo, te_val
            ),
            T_i=core_profile_helpers.make_constant_core_profile(
                self.geo, ti_val
            ),
        ),
    )

    result = core_profiles.below_minimum_temperature(threshold_ev)

    self.assertIsInstance(result, jax.Array)
    self.assertEqual(result.dtype, jnp.bool_)
    self.assertEqual(result, expected)

  def test_below_minimum_temperature_mixed_profile(self):
    """Tests detection when only part of the profile is below threshold."""
    threshold_ev = 100.0  # 0.1 keV

    te_values = jnp.array([0.2, 0.2, 0.05, 0.2, 0.2])  # 0.05 is below threshold
    ti_values = jnp.array([0.2, 0.2, 0.2, 0.2, 0.2])

    core_profiles = dataclasses.replace(
        self.base_profiles,
        kinetic=dataclasses.replace(
            self.base_profiles.kinetic,
            T_e=dataclasses.replace(
                self.base_profiles.kinetic.T_e, value=te_values
            ),
            T_i=dataclasses.replace(
                self.base_profiles.kinetic.T_i, value=ti_values
            ),
        ),
    )

    self.assertTrue(core_profiles.below_minimum_temperature(threshold_ev))


class ImpurityFractionsTest(parameterized.TestCase):
  """Tests for the impurity_fractions attribute in CoreProfiles."""

  def setUp(self):
    super().setUp()
    self.base_config_dict = {
        'profile_conditions': {
            'I_p': 15.0e6,
            'I_p_source': 'profile_conditions',
            'initial_psi_source': 'j',
            'T_i': {0: {0: 15.0, 1: 1.0}},
            'T_e': {0: {0: 15.0, 1: 1.0}},
            'n_e': {0: {0: 1.2e20, 1: 0.8e20}},
        },
        'geometry': {'geometry_type': 'circular'},
        'sources': {},
        'transport': {
            'core_transport_models': {
                'prescribed': {'model_name': 'prescribed'}
            },
        },
        'solver': {'solver_type': 'linear'},
        'numerics': {},
        'time_step_calculator': {
            'calculator_type': 'fixed',
            'dt': 1.0,
            't_final': 5.0,
        },
        'pedestal': {},
    }

  def test_impurity_fractions_output_fractions_mode(self):
    config_dict = copy.deepcopy(self.base_config_dict)
    config_dict['plasma_composition'] = {
        'main_ion': 'D',
        'impurity': {
            'impurity_mode': 'fractions',
            'species': {
                'Ar': {0.0: {0.0: 0.1}, 5.0: {0.0: 0.3}},
                'Ne': {0.0: {0.0: 0.8}, 5.0: {0.0: 0.2}},
                'C': {0.0: {0.0: 0.1}, 5.0: {0.0: 0.5}},
            },
        },
        'Z_eff': 2.0,
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    state_history = run_simulation.run_simulation(torax_config)

    self.assertEqual(state_history.sim_error, state.SimError.NO_ERROR)

    for i, t in enumerate(state_history.times):
      fractions = state_history.core_profiles[i].ions.impurity_fractions
      impurity_config = torax_config.plasma_composition.impurity
      geo = torax_config.geometry.build_provider(t)
      assert isinstance(impurity_config, impurity_fractions.ImpurityFractions)
      expected_fractions = {
          'Ar': jnp.full_like(
              geo.rho_norm, impurity_config.species['Ar'].get_value(t)
          ),
          'C': jnp.full_like(
              geo.rho_norm, impurity_config.species['C'].get_value(t)
          ),
          'Ne': jnp.full_like(
              geo.rho_norm, impurity_config.species['Ne'].get_value(t)
          ),
      }
      self.assertEqual(fractions.keys(), expected_fractions.keys())
      for key in fractions:
        np.testing.assert_allclose(
            fractions[key],
            expected_fractions[key],
            rtol=1e-5,
            err_msg=(
                f'Mismatch in impurity fraction for {key} at time t={t}.\n'
                f'Got: {fractions[key]}, Expected: {expected_fractions[key]}'
            ),
        )

  def test_impurity_fractions_output_ne_ratios_mode(self):
    config_dict = copy.deepcopy(self.base_config_dict)
    config_dict['plasma_composition'] = {
        'main_ion': 'D',
        'impurity': {
            'impurity_mode': 'n_e_ratios',
            'species': {
                'Ne': {0.0: {0: 0.01}, 5.0: {0: 0.005}},
                'Ar': {0.0: {0: 0.005}, 5.0: {0: 0.01}},
                'W': {0.0: {0: 1e-4}, 5.0: {0: 1e-5}},
            },
        },
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    state_history = run_simulation.run_simulation(torax_config)

    self.assertEqual(state_history.sim_error, state.SimError.NO_ERROR)

    for i, t in enumerate(state_history.times):
      fractions = state_history.core_profiles[i].ions.impurity_fractions
      impurity_config = torax_config.plasma_composition.impurity
      geo = torax_config.geometry.build_provider(t)
      assert isinstance(
          impurity_config, electron_density_ratios.ElectronDensityRatios
      )
      ar_ratio = impurity_config.species['Ar'].get_value(t)
      ne_ratio = impurity_config.species['Ne'].get_value(t)
      w_ratio = impurity_config.species['W'].get_value(t)
      total_ratio = ne_ratio + ar_ratio + w_ratio
      expected_fractions = {
          'Ar': jnp.full_like(geo.rho_norm, ar_ratio / total_ratio),
          'Ne': jnp.full_like(geo.rho_norm, ne_ratio / total_ratio),
          'W': jnp.full_like(geo.rho_norm, w_ratio / total_ratio),
      }
      self.assertEqual(fractions.keys(), expected_fractions.keys())
      for key in fractions:
        np.testing.assert_allclose(
            fractions[key],
            expected_fractions[key],
            rtol=1e-5,
            err_msg=(
                f'Mismatch in impurity fraction for {key} at time t={t}.\n'
                f'Got: {fractions[key]}, Expected: {expected_fractions[key]}'
            ),
        )

  def test_negative_impurity_triggers_error(self):
    """Tests that an unphysical config leading to negative impurity fraction is caught."""
    config_dict = copy.deepcopy(self.base_config_dict)
    config_dict['plasma_composition'] = {
        'main_ion': 'D',
        'impurity': {
            'impurity_mode': plasma_composition._IMPURITY_MODE_NE_RATIOS_ZEFF,
            'species': {
                'C': 0.02,  # Carbon ratio is fixed and too high for Z_eff
                'W': None,  # Tungsten is constrained by Z_eff
            },
        },
        'Z_eff': 1.5,
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)

    state_history = run_simulation.run_simulation(
        torax_config, progress_bar=False
    )

    # The simulation should have stopped early with an error.
    self.assertEqual(
        state_history.sim_error, state.SimError.NEGATIVE_CORE_PROFILES
    )
    # Verify that the simulation terminated before reaching t_final.
    self.assertLess(
        state_history.times[-1], torax_config.time_step_calculator.t_final
    )


class CoreProfilesCachedPropertiesTest(parameterized.TestCase):
  """Tests for the cached properties in CoreProfiles."""

  def setUp(self):
    super().setUp()
    self.geo = circular_geometry.CircularConfig(n_rho=10).build_geometry()
    solver_profiles = core_profile_helpers.make_zero_core_profiles(self.geo)
    self.core_profiles = dataclasses.replace(
        solver_profiles,
        kinetic=dataclasses.replace(
            solver_profiles.kinetic,
            T_i=core_profile_helpers.make_constant_core_profile(self.geo, 1.0),
            T_e=core_profile_helpers.make_constant_core_profile(self.geo, 2.0),
            n_e=core_profile_helpers.make_constant_core_profile(
                self.geo, 3.0e20
            ),
            n_i=core_profile_helpers.make_constant_core_profile(
                self.geo, 2.5e20
            ),
            n_impurity=core_profile_helpers.make_constant_core_profile(
                self.geo, 0.25e20
            ),
        ),
    )

  def test_pressure_thermal_e(self):
    """Test that thermal pressures are computed correctly."""
    np.testing.assert_allclose(
        self.core_profiles.pressure_thermal_e.value,
        6e20 * constants.CONSTANTS.keV_to_J,
    )

  def test_pressure_thermal_i(self):
    np.testing.assert_allclose(
        self.core_profiles.pressure_thermal_i.value,
        2.75e20 * constants.CONSTANTS.keV_to_J,
    )

  def test_pressure_thermal_total(self):
    np.testing.assert_allclose(
        self.core_profiles.pressure_thermal_total.value,
        8.75e20 * constants.CONSTANTS.keV_to_J,
    )

  def test_psi_norm_face_spans_zero_to_one(self):
    psi = cell_variable.CellVariable(
        value=1.0 + self.geo.rho_norm**2,
        face_centers=self.geo.rho_face_norm,
        right_face_grad_constraint=jnp.array(2.0),
    )
    core_profiles = dataclasses.replace(
        self.core_profiles,
        flux=dataclasses.replace(self.core_profiles.flux, psi=psi),
    )
    np.testing.assert_allclose(
        [core_profiles.psi_norm_face[0], core_profiles.psi_norm_face[-1]],
        [0.0, 1.0],
    )

  def test_n_i_total_thermal_face_rescales_impurity_density(self):
    geo = circular_geometry.CircularConfig(n_rho=10).build_geometry()
    solver_profiles = core_profile_helpers.make_zero_core_profiles(geo)
    # Bundled impurity charge <Z^2>/<Z> = 3 and <Z> = 2, so the true impurity
    # particle density is 1.5x the bundled one.
    core_profiles = dataclasses.replace(
        solver_profiles,
        kinetic=dataclasses.replace(
            solver_profiles.kinetic,
            n_i=core_profile_helpers.make_constant_core_profile(geo, 2.5e20),
            n_impurity=core_profile_helpers.make_constant_core_profile(
                geo, 0.25e20
            ),
        ),
        ions=dataclasses.replace(
            solver_profiles.ions,
            Z_impurity_face=jnp.full_like(geo.rho_face, 3.0),
            charge_state_info_face=dataclasses.replace(
                solver_profiles.ions.charge_state_info_face,
                Z_avg=jnp.full_like(geo.rho_face, 2.0),
            ),
        ),
    )

    np.testing.assert_allclose(
        core_profiles.n_i_total_thermal_face, 2.5e20 + 1.5 * 0.25e20
    )

  def test_thermal_densities_with_fast_ions(self):
    geo = circular_geometry.CircularConfig(n_rho=10).build_geometry()
    solver_profiles = core_profile_helpers.make_zero_core_profiles(geo)

    fast_ion_he3 = fast_ion_lib.FastIon(
        species='He3',
        source='ICRH',
        n=core_profile_helpers.make_constant_core_profile(geo, 0.05e20),
        T=core_profile_helpers.make_constant_core_profile(geo, 100.0),
    )

    core_profiles = dataclasses.replace(
        solver_profiles,
        kinetic=dataclasses.replace(
            solver_profiles.kinetic,
            T_i=core_profile_helpers.make_constant_core_profile(geo, 1.0),
            T_e=core_profile_helpers.make_constant_core_profile(geo, 2.0),
            n_e=core_profile_helpers.make_constant_core_profile(geo, 3.0e20),
            n_i=core_profile_helpers.make_constant_core_profile(geo, 2.5e20),
            n_impurity=core_profile_helpers.make_constant_core_profile(
                geo, 0.25e20
            ),
        ),
        ions=dataclasses.replace(
            solver_profiles.ions,
            main_ion_fractions={'D': 1.0},
            impurity_fractions={'He3': jnp.ones(10)},
        ),
        fast_ions=(fast_ion_he3,),
    )

    np.testing.assert_allclose(
        core_profiles.n_impurity_thermal.value,
        0.20e20,
    )
    np.testing.assert_allclose(
        core_profiles.pressure_thermal_i.value,
        2.7e20 * constants.CONSTANTS.keV_to_J,
    )

  def test_cached_properties_with_jit(self):
    """Tests cached properties behave correctly within a jitted function."""

    @jax.jit
    def get_pressure_thermal_e(profiles):
      return profiles.pressure_thermal_e.value

    # Create two different CoreProfiles
    geo = circular_geometry.CircularConfig(n_rho=10).build_geometry()
    profiles1 = core_profile_helpers.make_zero_core_profiles(geo)
    profiles1 = dataclasses.replace(
        profiles1,
        kinetic=dataclasses.replace(
            profiles1.kinetic,
            T_e=core_profile_helpers.make_constant_core_profile(geo, 2.0),
            n_e=core_profile_helpers.make_constant_core_profile(geo, 3.0e20),
        ),
    )
    profiles2 = core_profile_helpers.make_zero_core_profiles(geo)
    profiles2 = dataclasses.replace(
        profiles2,
        kinetic=dataclasses.replace(
            profiles2.kinetic,
            T_e=core_profile_helpers.make_constant_core_profile(geo, 4.0),
            n_e=core_profile_helpers.make_constant_core_profile(geo, 1.5e20),
        ),
    )

    # Get expected values without jit
    expected1 = profiles1.pressure_thermal_e.value
    expected2 = profiles2.pressure_thermal_e.value

    # Get values from jitted function called sequentially
    actual1 = get_pressure_thermal_e(profiles1)
    actual2 = get_pressure_thermal_e(profiles2)
    actual3 = get_pressure_thermal_e(profiles1)

    np.testing.assert_allclose(actual1, expected1)
    np.testing.assert_allclose(actual2, expected2)
    np.testing.assert_allclose(actual3, expected1)

  def test_core_profiles_to_output_dict_keys_and_shapes(self):
    """Tests CoreProfiles.to_output_dict produces all keys with valid shapes."""
    geo = circular_geometry.CircularConfig(n_rho=10).build_geometry()
    profiles_t0 = core_profile_helpers.make_zero_core_profiles(geo)
    profiles_t1 = dataclasses.replace(
        profiles_t0,
        kinetic=dataclasses.replace(
            profiles_t0.kinetic,
            T_i=core_profile_helpers.make_constant_core_profile(geo, 5.0),
            T_e=core_profile_helpers.make_constant_core_profile(geo, 4.0),
        ),
    )
    # Stack two time steps.
    stacked_profiles = jax.tree.map(
        lambda *xs: np.stack(xs), profiles_t0, profiles_t1
    )
    rho_cell_plus_boundaries = np.concatenate(([0.0], geo.rho_norm, [1.0]))
    context = output_grid_context.OutputGridContext(
        times=np.array([0.0, 1.0]),
        rho_face_norm=geo.rho_face_norm,
        rho_cell_norm=geo.rho_norm,
        rho_cell_plus_boundaries_norm=rho_cell_plus_boundaries,
    )
    out_dict = stacked_profiles.to_output_dict(context)

    cell_plus_boundaries_keys = (
        output_keys.T_I,
        output_keys.T_E,
        output_keys.PSI,
        output_keys.V_LOOP,
        output_keys.N_E,
        output_keys.N_I,
        output_keys.N_IMPURITY,
        output_keys.TOROIDAL_ANGULAR_VELOCITY,
        output_keys.PRESSURE_THERMAL_E,
        output_keys.PRESSURE_THERMAL_I,
        output_keys.PRESSURE_THERMAL_TOTAL,
        output_keys.PRESSURE_FAST_I,
        output_keys.PRESSURE_TOTAL_I,
        output_keys.PRESSURE_TOTAL,
        output_keys.Z_I,
        output_keys.Z_IMPURITY,
        output_keys.A_IMPURITY,
        output_keys.Z_EFF,
        output_keys.SIGMA_PARALLEL,
        output_keys.J_TOROIDAL_TOTAL,
    )
    for key in cell_plus_boundaries_keys:
      self.assertIn(key, out_dict)
      dims, data, attrs = out_dict[key]
      self.assertEqual(dims, (output_keys.TIME, output_keys.RHO_NORM))
      self.assertEqual(data.shape, (2, len(rho_cell_plus_boundaries)))
      self.assertIn('units', attrs)

    face_keys = (
        output_keys.Q,
        output_keys.MAGNETIC_SHEAR,
        output_keys.I_P_PROFILE,
    )
    for key in face_keys:
      self.assertIn(key, out_dict)
      dims, data, _ = out_dict[key]
      self.assertEqual(dims, (output_keys.TIME, output_keys.RHO_FACE_NORM))
      self.assertEqual(data.shape, (2, len(geo.rho_face_norm)))

    scalar_keys = (
        output_keys.V_LOOP_LCFS,
        output_keys.A_I,
        output_keys.I_P,
    )
    for key in scalar_keys:
      self.assertIn(key, out_dict)
      dims, data, _ = out_dict[key]
      self.assertEqual(dims, (output_keys.TIME,))
      self.assertEqual(data.shape, (2,))

    self.assertIn(output_keys.MAIN_ION_FRACTIONS, out_dict)
    dims, data, attrs = out_dict[output_keys.MAIN_ION_FRACTIONS]
    self.assertEqual(dims, (output_keys.MAIN_ION, output_keys.TIME))
    self.assertEqual(data.shape, (1, 2))
    self.assertIn('units', attrs)

  def test_core_profiles_to_output_dict_fast_ions(self):
    """Tests that fast ion profiles are correctly packed in output dict."""
    geo = circular_geometry.CircularConfig(n_rho=10).build_geometry()
    fast_ion = fast_ion_lib.FastIon(
        source='nbi',
        species='D',
        n=core_profile_helpers.make_constant_core_profile(geo, 1.0e19),
        T=core_profile_helpers.make_constant_core_profile(geo, 50.0),
    )
    profiles = core_profile_helpers.make_zero_core_profiles(geo)
    profiles = dataclasses.replace(profiles, fast_ions=(fast_ion,))
    stacked_profiles = jax.tree.map(
        lambda *xs: np.stack(xs), profiles, profiles
    )
    rho_cell_plus_boundaries = np.concatenate(([0.0], geo.rho_norm, [1.0]))
    context = output_grid_context.OutputGridContext(
        times=np.array([0.0, 1.0]),
        rho_face_norm=geo.rho_face_norm,
        rho_cell_norm=geo.rho_norm,
        rho_cell_plus_boundaries_norm=rho_cell_plus_boundaries,
    )
    out_dict = stacked_profiles.to_output_dict(context)

    n_key = output_keys.n_fast_ion_key('nbi_D')
    t_key = output_keys.T_fast_ion_key('nbi_D')
    self.assertIn(n_key, out_dict)
    self.assertIn(t_key, out_dict)

    n_dims, n_data, n_attrs = out_dict[n_key]
    self.assertEqual(n_dims, (output_keys.TIME, output_keys.RHO_NORM))
    self.assertEqual(n_attrs['units'], 'm^-3')
    self.assertEqual(n_data.shape, (2, len(rho_cell_plus_boundaries)))

    t_dims, t_data, t_attrs = out_dict[t_key]
    self.assertEqual(t_dims, (output_keys.TIME, output_keys.RHO_NORM))
    self.assertEqual(t_attrs['units'], 'keV')
    self.assertEqual(t_data.shape, (2, len(rho_cell_plus_boundaries)))

  def test_core_profiles_to_output_dict_radial_a_impurity(self):
    """Tests non-constant A_impurity is packed as a spatial profile."""
    geo = circular_geometry.CircularConfig(n_rho=10).build_geometry()
    profiles = core_profile_helpers.make_zero_core_profiles(geo)
    # Set non-constant A_impurity across radius.
    profiles = dataclasses.replace(
        profiles,
        ions=dataclasses.replace(
            profiles.ions,
            A_impurity=np.arange(10, dtype=np.float32) + 1.0,
            A_impurity_face=np.arange(11, dtype=np.float32) + 1.0,
        ),
    )
    stacked_profiles = jax.tree.map(
        lambda *xs: np.stack(xs), profiles, profiles
    )
    rho_cell_plus_boundaries = np.concatenate(([0.0], geo.rho_norm, [1.0]))
    context = output_grid_context.OutputGridContext(
        times=np.array([0.0, 1.0]),
        rho_face_norm=geo.rho_face_norm,
        rho_cell_norm=geo.rho_norm,
        rho_cell_plus_boundaries_norm=rho_cell_plus_boundaries,
    )
    out_dict = stacked_profiles.to_output_dict(context)

    self.assertIn(output_keys.A_IMPURITY, out_dict)
    dims, data, attrs = out_dict[output_keys.A_IMPURITY]
    self.assertEqual(dims, (output_keys.TIME, output_keys.RHO_NORM))
    self.assertEqual(data.shape, (2, len(rho_cell_plus_boundaries)))
    self.assertEqual(attrs['units'], 'amu')
    expected_extended = output_grid_context.extend_cell_grid_to_boundaries(
        profiles.ions.A_impurity, profiles.ions.A_impurity_face
    )
    np.testing.assert_allclose(
        data, np.stack([expected_extended, expected_extended])
    )


class CoreTransportTest(parameterized.TestCase):

  def test_core_transport_to_output_dict(self):
    geo = circular_geometry.CircularConfig(n_rho=10).build_geometry()
    times = np.array([0.0, 1.0])
    context = output_grid_context.OutputGridContext(
        times=times,
        rho_face_norm=geo.rho_face_norm,
        rho_cell_norm=geo.rho_norm,
        rho_cell_plus_boundaries_norm=np.concatenate(
            [[0.0], geo.rho_norm, [1.0]]
        ),
    )
    n_face = geo.rho_face_norm.size
    turbulent_core = transport_coeffs_lib.TransportCoeffs(
        chi_face_ion=jnp.ones((2, n_face)) * 1.5,
        chi_face_el=jnp.ones((2, n_face)) * 2.0,
        d_face_el=jnp.ones((2, n_face)) * 0.5,
        v_face_el=jnp.ones((2, n_face)) * -0.2,
    )
    turbulent_pedestal = transport_coeffs_lib.TransportCoeffs(
        chi_face_ion=jnp.zeros((2, n_face)),
        chi_face_el=jnp.zeros((2, n_face)),
        d_face_el=jnp.zeros((2, n_face)),
        v_face_el=jnp.zeros((2, n_face)),
    )
    turbulent = transport_coeffs_lib.TurbulentTransport(
        core=turbulent_core,
        pedestal=turbulent_pedestal,
        core_components={},
        pedestal_components={},
    )
    neoclassical = transport_coeffs_lib.NeoclassicalTransport(
        chi_face_ion=jnp.ones((2, n_face)) * 0.1,
        chi_face_el=jnp.ones((2, n_face)) * 0.05,
        d_face_el=jnp.ones((2, n_face)) * 0.01,
        v_face_el=jnp.ones((2, n_face)) * -0.03,
        v_face_el_ware=jnp.ones((2, n_face)) * -0.01,
    )
    pereverzev = transport_coeffs_lib.PereverzevTransport(
        chi_face_ion=jnp.ones((2, n_face)) * 0.05,
        chi_face_el=jnp.ones((2, n_face)) * 0.04,
        full_v_heat_face_ion=jnp.ones((2, n_face)) * 0.02,
        full_v_heat_face_el=jnp.ones((2, n_face)) * 0.01,
        d_face_el=jnp.ones((2, n_face)) * 0.005,
        v_face_el=jnp.ones((2, n_face)) * -0.001,
    )
    total = transport_coeffs_lib.sum_transport_coeffs(
        turbulent.total, neoclassical, pereverzev
    )
    core_transport = state.CoreTransport(
        total=total,
        turbulent=turbulent,
        neoclassical=neoclassical,
        pereverzev=pereverzev,
    )
    out_dict = core_transport.to_output_dict(context)

    # Verify turbulent keys.
    self.assertIn(output_keys.CHI_TURB_I, out_dict)
    self.assertIn(output_keys.CHI_TURB_E, out_dict)
    self.assertIn(output_keys.D_TURB_E, out_dict)
    self.assertIn(output_keys.V_TURB_E, out_dict)

    # Verify neoclassical keys.
    self.assertIn(output_keys.CHI_NEO_I, out_dict)
    self.assertIn(output_keys.CHI_NEO_E, out_dict)
    self.assertIn(output_keys.D_NEO_E, out_dict)
    self.assertIn(output_keys.V_NEO_E, out_dict)
    self.assertIn(output_keys.V_NEO_WARE_E, out_dict)

    # Check dimensions and data for sample channels.
    dims, data, attrs = out_dict[output_keys.CHI_TURB_I]
    self.assertEqual(dims, (output_keys.TIME, output_keys.RHO_FACE_NORM))
    np.testing.assert_allclose(data, np.ones((2, n_face)) * 1.5)
    self.assertEqual(
        attrs, {'units': output_keys.Units.SQUARE_METER_PER_SECOND}
    )

  def test_core_transport_total_chi_max(self):
    geo = circular_geometry.CircularConfig(n_rho=10).build_geometry()
    core_transport = state.CoreTransport.zeros(geo)
    chi_max = core_transport.total.chi_max(geo)
    self.assertEqual(chi_max.shape, ())


class SolverNumericOutputsTest(parameterized.TestCase):

  def test_solver_numeric_outputs_to_output_dict(self):
    geo = circular_geometry.CircularConfig(n_rho=10).build_geometry()
    times = np.array([0.0, 1.0])
    context = output_grid_context.OutputGridContext(
        times=times,
        rho_face_norm=geo.rho_face_norm,
        rho_cell_norm=geo.rho_norm,
        rho_cell_plus_boundaries_norm=np.concatenate(
            [[0.0], geo.rho_norm, [1.0]]
        ),
    )
    solver_numeric_outputs = state.SolverNumericOutputs(
        outer_solver_iterations=np.array([3, 4]),
        inner_solver_iterations=np.array([10, 12]),
        solver_error_state=np.array([0, 0]),
        sawtooth_crash=np.array([False, True]),
    )
    out_dict = solver_numeric_outputs.to_output_dict(context)

    expected_keys = {
        output_keys.OUTER_SOLVER_ITERATIONS,
        output_keys.INNER_SOLVER_ITERATIONS,
        output_keys.SOLVER_ERROR_STATE,
        output_keys.SAWTOOTH_CRASH,
    }
    self.assertEqual(out_dict.keys(), expected_keys)

    for key in expected_keys:
      dims, _, attrs = out_dict[key]
      self.assertEqual(dims, (output_keys.TIME,))
      self.assertEqual(attrs, {})

    np.testing.assert_array_equal(
        out_dict[output_keys.OUTER_SOLVER_ITERATIONS][1], np.array([3, 4])
    )
    np.testing.assert_array_equal(
        out_dict[output_keys.INNER_SOLVER_ITERATIONS][1], np.array([10, 12])
    )
    np.testing.assert_array_equal(
        out_dict[output_keys.SOLVER_ERROR_STATE][1], np.array([0, 0])
    )
    np.testing.assert_array_equal(
        out_dict[output_keys.SAWTOOTH_CRASH][1], np.array([False, True])
    )


class SolverProfilesTest(parameterized.TestCase):
  """Unit tests for SolverProfiles and CoreProfiles."""

  def setUp(self):
    super().setUp()
    self.geo = circular_geometry.CircularConfig(n_rho=10).build_geometry()
    torax_config = config.ToraxConfig.from_dict(
        default_configs.get_default_config_dict()
    )
    core_profiles, self.runtime_params, _ = (
        core_profile_helpers.make_initial_inputs(torax_config)
    )
    self.base_profiles = solver_profiles_lib.SolverProfiles(
        kinetic=core_profiles.kinetic,
        ions=core_profiles.ions,
        flux=core_profiles.flux,
        toroidal_angular_velocity=core_profiles.toroidal_angular_velocity,
    )
    self.core_profiles = core_profiles

  def test_solver_profiles_properties(self):
    self.assertTrue(self.base_profiles.quasineutrality_satisfied())
    self.assertFalse(self.base_profiles.negative_temperature_or_density())
    self.assertFalse(self.base_profiles.below_minimum_temperature(10.0))

    np.testing.assert_allclose(
        self.base_profiles.impurity_density_scaling,
        self.core_profiles.impurity_density_scaling,
    )
    np.testing.assert_allclose(
        self.base_profiles.impurity_density_scaling_face,
        self.core_profiles.impurity_density_scaling_face,
    )
    np.testing.assert_allclose(
        self.base_profiles.psi_norm,
        self.core_profiles.psi_norm,
    )
    np.testing.assert_allclose(
        self.base_profiles.psi_norm_face,
        self.core_profiles.psi_norm_face,
    )
    np.testing.assert_allclose(
        self.base_profiles.pressure_thermal_e.value,
        self.core_profiles.pressure_thermal_e.value,
    )

  def test_core_profiles_pressures(self):
    result = state.CoreProfiles(
        kinetic=self.base_profiles.kinetic,
        ions=self.base_profiles.ions,
        flux=self.base_profiles.flux,
        toroidal_angular_velocity=self.base_profiles.toroidal_angular_velocity,
        fast_ions=self.core_profiles.fast_ions,
        conductivity=self.core_profiles.conductivity,
        poloidal_velocity=self.core_profiles.poloidal_velocity,
        psidot=self.core_profiles.psidot,
    )

    self.assertIsInstance(result, state.CoreProfiles)
    np.testing.assert_allclose(
        result.conductivity.sigma,
        self.core_profiles.conductivity.sigma,
    )
    np.testing.assert_allclose(
        result.conductivity.sigma_face,
        self.core_profiles.conductivity.sigma_face,
    )

    pressure_attrs = (
        'pressure_thermal_e',
        'pressure_thermal_i',
        'pressure_thermal_total',
        'pressure_fast_i',
        'pressure_total_i',
        'pressure_total',
    )
    for attr in pressure_attrs:
      p1 = getattr(result, attr)
      p2 = getattr(self.core_profiles, attr)
      np.testing.assert_allclose(p1.value, p2.value)
      np.testing.assert_allclose(p1.face_value(), p2.face_value())
      np.testing.assert_allclose(
          p1.cell_plus_boundaries(), p2.cell_plus_boundaries()
      )


if __name__ == '__main__':
  absltest.main()

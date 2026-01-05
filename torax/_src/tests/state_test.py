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
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.core_profiles.plasma_composition import electron_density_ratios
from torax._src.core_profiles.plasma_composition import impurity_fractions
from torax._src.core_profiles.plasma_composition import plasma_composition
from torax._src.geometry import circular_geometry
from torax._src.orchestration import run_simulation
from torax._src.test_utils import core_profile_helpers
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config

# pylint: disable=invalid-name


class InitialStatesTest(parameterized.TestCase):

  def test_initial_boundary_condition_from_time_dependent_params(self):
    """Tests that the initial boundary conditions are set from the config."""
    config = default_configs.get_default_config_dict()
    # Boundary conditions can be time-dependent, but when creating the initial
    # core profiles, we want to grab the boundary condition params at time 0.
    config['profile_conditions'] = {
        'T_i_right_bc': 27.7,
        'T_e_right_bc': {0.0: 42.0, 1.0: 0.001},
        'n_e_right_bc': ({0.0: 0.1e20, 1.0: 2.0e20}, 'step'),
        'normalize_n_e_to_nbar': False,
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    dynamic_provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    dynamic_runtime_params_slice, geo = (
        build_runtime_params.get_consistent_runtime_params_and_geometry(
            t=torax_config.numerics.t_initial,
            runtime_params_provider=dynamic_provider,
            geometry_provider=torax_config.geometry.build_provider,
        )
    )
    core_profiles = initialization.initial_core_profiles(
        runtime_params=dynamic_runtime_params_slice,
        geo=geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )
    np.testing.assert_allclose(core_profiles.T_i.right_face_constraint, 27.7)
    np.testing.assert_allclose(core_profiles.T_e.right_face_constraint, 42.0)
    np.testing.assert_allclose(core_profiles.n_e.right_face_constraint, 0.1e20)

  def test_core_profiles_quasineutrality_check(self):
    """Tests core_profiles quasineutrality check on initial state."""
    torax_config = model_config.ToraxConfig.from_dict(
        default_configs.get_default_config_dict()
    )
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    dynamic_runtime_params_slice_provider = (
        build_runtime_params.RuntimeParamsProvider.from_config(torax_config)
    )
    dynamic_runtime_params_slice, geo = (
        build_runtime_params.get_consistent_runtime_params_and_geometry(
            t=torax_config.numerics.t_initial,
            runtime_params_provider=dynamic_runtime_params_slice_provider,
            geometry_provider=torax_config.geometry.build_provider,
        )
    )
    core_profiles = initialization.initial_core_profiles(
        runtime_params=dynamic_runtime_params_slice,
        geo=geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )
    assert core_profiles.quasineutrality_satisfied()
    core_profiles = dataclasses.replace(
        core_profiles,
        Z_i=core_profiles.Z_i * 2.0,
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
          T_i=dataclasses.replace(
              core_profiles.T_i,
              value=jnp.array(-1.0),
          ),
      )
      self.assertTrue(new_core_profiles.negative_temperature_or_density())
    with self.subTest('negative psi does not trigger'):
      new_core_profiles = dataclasses.replace(
          core_profiles,
          psi=dataclasses.replace(
              core_profiles.psi,
              value=jnp.array(-1.0),
          ),
      )
      self.assertFalse(new_core_profiles.negative_temperature_or_density())


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
        T_e=core_profile_helpers.make_constant_core_profile(self.geo, te_val),
        T_i=core_profile_helpers.make_constant_core_profile(self.geo, ti_val),
    )

    result = core_profiles.below_minimum_temperature(threshold_ev)

    self.assertIsInstance(result, bool)
    self.assertEqual(result, expected)

  def test_below_minimum_temperature_mixed_profile(self):
    """Tests detection when only part of the profile is below threshold."""
    threshold_ev = 100.0  # 0.1 keV

    te_values = jnp.array([0.2, 0.2, 0.05, 0.2, 0.2])  # 0.05 is below threshold
    ti_values = jnp.array([0.2, 0.2, 0.2, 0.2, 0.2])

    core_profiles = dataclasses.replace(
        self.base_profiles,
        T_e=dataclasses.replace(self.base_profiles.T_e, value=te_values),
        T_i=dataclasses.replace(self.base_profiles.T_i, value=ti_values),
    )

    self.assertTrue(core_profiles.below_minimum_temperature(threshold_ev))


class ImpurityFractionsTest(parameterized.TestCase):
  """Tests for the impurity_fractions attribute in CoreProfiles."""

  def setUp(self):
    super().setUp()
    self.base_config_dict = {
        'profile_conditions': {
            'Ip': 15.0e6,
            'T_i': {0: {0: 15.0, 1: 1.0}},
            'T_e': {0: {0: 15.0, 1: 1.0}},
            'n_e': {0: {0: 1.2e20, 1: 0.8e20}},
        },
        'numerics': {'t_final': 5.0, 'fixed_dt': 1.0},
        'geometry': {'geometry_type': 'circular'},
        'sources': {},
        'transport': {'model_name': 'constant'},
        'solver': {'solver_type': 'linear'},
        'time_step_calculator': {'calculator_type': 'fixed'},
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
    torax_config = model_config.ToraxConfig.from_dict(config_dict)
    _, state_history = run_simulation.run_simulation(torax_config)

    self.assertEqual(state_history.sim_error, state.SimError.NO_ERROR)

    for i, t in enumerate(state_history.times):
      fractions = state_history.core_profiles[i].impurity_fractions
      impurity_config = torax_config.plasma_composition.impurity
      geo = torax_config.geometry.build_provider(t)
      assert isinstance(
          impurity_config, impurity_fractions.ImpurityFractions
      )
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
    torax_config = model_config.ToraxConfig.from_dict(config_dict)
    _, state_history = run_simulation.run_simulation(torax_config)

    self.assertEqual(state_history.sim_error, state.SimError.NO_ERROR)

    for i, t in enumerate(state_history.times):
      fractions = state_history.core_profiles[i].impurity_fractions
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
    torax_config = model_config.ToraxConfig.from_dict(config_dict)

    _, state_history = run_simulation.run_simulation(
        torax_config, progress_bar=False
    )

    # The simulation should have stopped early with an error.
    self.assertEqual(
        state_history.sim_error, state.SimError.NEGATIVE_CORE_PROFILES
    )
    # Verify that the simulation terminated before reaching t_final.
    self.assertLess(state_history.times[-1], torax_config.numerics.t_final)


class CoreProfilesCachedPropertiesTest(parameterized.TestCase):
  """Tests for the cached properties in CoreProfiles."""

  def setUp(self):
    super().setUp()
    self.geo = circular_geometry.CircularConfig(n_rho=10).build_geometry()
    base_core_profiles = core_profile_helpers.make_zero_core_profiles(self.geo)
    self.core_profiles = dataclasses.replace(
        base_core_profiles,
        T_i=core_profile_helpers.make_constant_core_profile(self.geo, 1.0),
        T_e=core_profile_helpers.make_constant_core_profile(self.geo, 2.0),
        n_e=core_profile_helpers.make_constant_core_profile(self.geo, 3.0e20),
        n_i=core_profile_helpers.make_constant_core_profile(self.geo, 2.5e20),
        n_impurity=core_profile_helpers.make_constant_core_profile(
            self.geo, 0.25e20
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
        T_e=core_profile_helpers.make_constant_core_profile(geo, 2.0),
        n_e=core_profile_helpers.make_constant_core_profile(geo, 3.0e20),
    )
    profiles2 = core_profile_helpers.make_zero_core_profiles(geo)
    profiles2 = dataclasses.replace(
        profiles2,
        T_e=core_profile_helpers.make_constant_core_profile(geo, 4.0),
        n_e=core_profile_helpers.make_constant_core_profile(geo, 1.5e20),
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


if __name__ == '__main__':
  absltest.main()

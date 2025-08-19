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
from jax import numpy as jnp
import numpy as np
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.config import plasma_composition
from torax._src.core_profiles import initialization
from torax._src.geometry import pydantic_model as geometry_pydantic_model
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
    dynamic_provider = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            torax_config
        )
    )
    dynamic_runtime_params_slice, geo = (
        build_runtime_params.get_consistent_dynamic_runtime_params_slice_and_geometry(
            t=torax_config.numerics.t_initial,
            dynamic_runtime_params_slice_provider=dynamic_provider,
            geometry_provider=torax_config.geometry.build_provider,
        )
    )
    static_slice = build_runtime_params.build_static_params_from_config(
        torax_config
    )
    core_profiles = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
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
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            torax_config
        )
    )
    dynamic_runtime_params_slice, geo = (
        build_runtime_params.get_consistent_dynamic_runtime_params_slice_and_geometry(
            t=torax_config.numerics.t_initial,
            dynamic_runtime_params_slice_provider=dynamic_runtime_params_slice_provider,
            geometry_provider=torax_config.geometry.build_provider,
        )
    )
    static_slice = build_runtime_params.build_static_params_from_config(
        torax_config
    )
    core_profiles = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
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
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
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
                'Ar': {0.0: 0.1, 5.0: 0.3},
                'C': {0.0: 0.1, 5.0: 0.5},
                'Ne': {0.0: 0.8, 5.0: 0.2},
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
      assert isinstance(
          impurity_config, plasma_composition.ImpurityFractionsModel
      )
      expected_ar = impurity_config.species['Ar'].get_value(t)
      expected_c = impurity_config.species['C'].get_value(t)
      expected_ne = impurity_config.species['Ne'].get_value(t)
      expected_fractions = np.array([expected_ar, expected_c, expected_ne])
      np.testing.assert_allclose(
          fractions,
          expected_fractions,
          rtol=1e-5,
          err_msg=(
              f'Mismatch in impurity fractions at time t={t}.\n'
              f'Got: {fractions}, Expected: {expected_fractions}'
          ),
      )

  def test_impurity_fractions_output_ne_ratios_mode(self):
    config_dict = copy.deepcopy(self.base_config_dict)
    config_dict['plasma_composition'] = {
        'main_ion': 'D',
        'impurity': {
            'impurity_mode': 'n_e_ratios',
            'species': {
                'Ar': {0.0: 0.005, 5.0: 0.01},
                'Ne': {0.0: 0.01, 5.0: 0.005},
                'W': {0.0: 1e-4, 5.0: 1e-5},
            },
        },
    }
    torax_config = model_config.ToraxConfig.from_dict(config_dict)
    _, state_history = run_simulation.run_simulation(torax_config)

    self.assertEqual(state_history.sim_error, state.SimError.NO_ERROR)

    self.assertEqual(
        torax_config.plasma_composition.get_impurity_names(),
        ('Ar', 'Ne', 'W'),
    )

    for i, t in enumerate(state_history.times):
      fractions = state_history.core_profiles[i].impurity_fractions
      impurity_config = torax_config.plasma_composition.impurity
      assert isinstance(impurity_config, plasma_composition.NeRatiosModel)
      ar_ratio = impurity_config.species['Ar'].get_value(t)
      ne_ratio = impurity_config.species['Ne'].get_value(t)
      w_ratio = impurity_config.species['W'].get_value(t)
      total_ratio = ne_ratio + ar_ratio + w_ratio
      expected_fractions = np.array([ar_ratio, ne_ratio, w_ratio]) / total_ratio
      np.testing.assert_allclose(
          fractions,
          expected_fractions,
          rtol=1e-5,
          err_msg=(
              f'Mismatch in impurity fractions at time t={t}.\n'
              f'Got: {fractions}, Expected: {expected_fractions}'
          ),
      )


if __name__ == '__main__':
  absltest.main()

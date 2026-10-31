# In file: torax/_src/core_profiles/tests/updaters_test.py

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

import dataclasses
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.core_profiles import updaters
from torax._src.fvm import cell_variable
from torax._src.geometry import circular_geometry
from torax._src.sources import source_profile_builders
from torax._src.test_utils import core_profile_helpers
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import config


# pylint: disable=invalid-name
class UpdatersTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Build a default geo object for convenience.
    self.geo = circular_geometry.CircularConfig(n_rho=4).build_geometry()

    T_e = cell_variable.CellVariable(
        value=jnp.ones_like(self.geo.rho_norm),
        face_centers=self.geo.rho_face_norm,
        right_face_constraint=1.0,
        right_face_grad_constraint=None,
    )
    n_e = cell_variable.CellVariable(
        value=jnp.ones_like(self.geo.rho_norm),
        face_centers=self.geo.rho_face_norm,
        right_face_constraint=1.0,
        right_face_grad_constraint=None,
    )

    self.core_profiles_t = mock.create_autospec(
        state.CoreProfiles,
        instance=True,
        T_e=T_e,
        n_e=n_e,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='psi_prescribed_constant',
          psi={0.0: {0.0: 10.0, 1.0: 10.0}},
          expected_psi_val_t1=10.0,
      ),
      dict(
          testcase_name='psi_prescribed_time_varying',
          psi={0.0: {0.0: 10.0, 1.0: 10.0}, 2.0: {0.0: 20.0, 1.0: 20.0}},
          expected_psi_val_t1=15.0,  # Linearly interpolated at t=1.0
      ),
  )
  def test_prescribed_psi_update(self, psi, expected_psi_val_t1):
    """Tests that psi is updated when evolve_current is False and psi is prescribed."""
    config_dict = default_configs.get_default_config_dict()
    config_dict['profile_conditions'] = {
        'psi': psi,
        'I_p': 15e6,  # I_p is required but not used for psi value here
        'I_p_source': 'profile_conditions',
    }
    config_dict['numerics'] = {
        'evolve_current': False,
    }
    config_dict['profile_conditions'][
        'initial_psi_source'
    ] = 'profile_conditions'

    torax_config = config.ToraxConfig.from_dict(config_dict)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )

    # Initialize at t=0
    core_profiles_t0, runtime_params_t0, geo = (
        core_profile_helpers.make_initial_inputs(torax_config)
    )

    dt = 1.0
    runtime_params_t1 = provider(t=1.0)

    # Simulate a step update where prescribed values are applied
    core_profiles_t1 = updaters.provide_core_profiles_t_plus_dt(
        dt=jnp.array(dt),
        runtime_params_t=runtime_params_t0,
        runtime_params_t_plus_dt=runtime_params_t1,
        geo_t_plus_dt=geo,
        core_profiles_t=core_profiles_t0,
    )

    # Check psi value
    expected_psi_array = np.full_like(geo.rho_norm, expected_psi_val_t1)
    np.testing.assert_allclose(
        core_profiles_t1.psi.value, expected_psi_array, rtol=1e-5
    )

  def test_psi_not_updated_if_not_prescribed(self):
    """Tests that psi retains previous value if not prescribed (None) and not evolving."""
    config_dict = default_configs.get_default_config_dict()
    # Don't provide psi in profile_conditions, it defaults to None
    config_dict['numerics'] = {
        'evolve_current': False,
    }
    config_dict['profile_conditions']['psi'] = {
        0.0: 10.0
    }  # Set initial only via config for init

    torax_config = config.ToraxConfig.from_dict(config_dict)

    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    core_profiles_t0, runtime_params_t0, geo = (
        core_profile_helpers.make_initial_inputs(torax_config)
    )

    # Create runtime_params at t=1.0, but force psi to None
    runtime_params_t1 = provider(t=1.0)
    # dataclasses.replace on RuntimeParams frozen dataclass
    # We need to reach into profile_conditions
    prof_cond = runtime_params_t1.profile_conditions
    object.__setattr__(
        prof_cond, 'psi', None
    )  # Hack to set frozen field for test

    core_profiles_t1 = updaters.provide_core_profiles_t_plus_dt(
        dt=jnp.array(1.0),
        runtime_params_t=runtime_params_t0,
        runtime_params_t_plus_dt=runtime_params_t1,
        geo_t_plus_dt=geo,
        core_profiles_t=core_profiles_t0,
    )

    # Should be same as t0 since not evolved and not prescribed
    np.testing.assert_allclose(
        core_profiles_t1.psi.value, core_profiles_t0.psi.value
    )

  def test_psi_not_updated_if_evolve_current_true(self):
    """Tests that psi from profile_conditions is IGNORED if evolve_current is True."""
    config_dict = default_configs.get_default_config_dict()
    config_dict['numerics'] = {'evolve_current': True}
    config_dict['profile_conditions'] = {
        'psi': {
            0.0: {0.0: 10.0, 1.0: 10.0},
            1.0: {0.0: 20.0, 1.0: 20.0},
        },  # Prescribed varying
        'I_p': 15e6,
        'I_p_source': 'profile_conditions',
        'initial_psi_source': 'profile_conditions',
    }

    torax_config = config.ToraxConfig.from_dict(config_dict)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    core_profiles_t0, runtime_params_t0, geo = (
        core_profile_helpers.make_initial_inputs(torax_config)
    )

    # Initial should be 10.0
    np.testing.assert_allclose(core_profiles_t0.psi.value, 10.0)

    runtime_params_t1 = provider(t=1.0)
    # At t=1, prescribed is 20.0. But evolve_current=True.

    core_profiles_t1 = updaters.provide_core_profiles_t_plus_dt(
        dt=jnp.array(1.0),
        runtime_params_t=runtime_params_t0,
        runtime_params_t_plus_dt=runtime_params_t1,
        geo_t_plus_dt=geo,
        core_profiles_t=core_profiles_t0,
    )

    # Expect psi value to NOT be updated to 20.0 (it should stay as t0 value
    # because provide_core_profiles_t_plus_dt only updates prescribed/BCs,
    # and evolving vars are handled by the solver).
    # Since it wasn't updated in provide_..., it should remain 10.0 here.
    np.testing.assert_allclose(core_profiles_t1.psi.value, 10.0)

  def test_provide_solver_state_prescribed_psi(self):
    config_dict = default_configs.get_default_config_dict()
    config_dict['profile_conditions'] = {
        'psi': {0.0: {0.0: 10.0, 1.0: 10.0}, 2.0: {0.0: 20.0, 1.0: 20.0}},
        'I_p': 15e6,
        'initial_psi_source': 'profile_conditions',
    }
    config_dict['numerics'] = {
        'evolve_current': False,
    }

    torax_config = config.ToraxConfig.from_dict(config_dict)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    core_profiles_t0, runtime_params_t0, geo = (
        core_profile_helpers.make_initial_inputs(torax_config)
    )

    runtime_params_t1 = provider(t=1.0)
    result = updaters.provide_solver_state_t_plus_dt(
        dt=jnp.array(1.0),
        runtime_params_t=runtime_params_t0,
        runtime_params_t_plus_dt=runtime_params_t1,
        geo_t_plus_dt=geo,
        core_profiles_t=core_profiles_t0,
    )

    self.assertIsInstance(result, state.SolverState)
    expected_psi_array = np.full_like(geo.rho_norm, 15.0)
    np.testing.assert_allclose(result.psi.value, expected_psi_array, rtol=1e-5)

  def test_provide_solver_state_preserves_evolving_and_updates_bcs(self):
    config_dict = default_configs.get_default_config_dict()
    config_dict['numerics'] = {
        'evolve_current': True,
        'evolve_ion_heat': True,
        'evolve_electron_heat': True,
        'evolve_density': True,
    }
    config_dict['profile_conditions']['T_i_right_bc'] = {0.0: 1.0, 1.0: 2.5}
    config_dict['profile_conditions']['T_e_right_bc'] = {0.0: 1.0, 1.0: 3.5}
    config_dict['profile_conditions']['n_e_right_bc'] = {
        0.0: 1.0e19,
        1.0: 5.5e19,
    }

    torax_config = config.ToraxConfig.from_dict(config_dict)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    core_profiles_t0, runtime_params_t0, geo = (
        core_profile_helpers.make_initial_inputs(torax_config)
    )

    runtime_params_t1 = provider(t=1.0)
    result = updaters.provide_solver_state_t_plus_dt(
        dt=jnp.array(1.0),
        runtime_params_t=runtime_params_t0,
        runtime_params_t_plus_dt=runtime_params_t1,
        geo_t_plus_dt=geo,
        core_profiles_t=core_profiles_t0,
    )

    self.assertIsInstance(result, state.SolverState)
    with self.subTest('evolving_values_preserved'):
      # Evolving channels retain values from t0.
      np.testing.assert_allclose(result.psi.value, core_profiles_t0.psi.value)
      np.testing.assert_allclose(result.T_i.value, core_profiles_t0.T_i.value)
      np.testing.assert_allclose(result.T_e.value, core_profiles_t0.T_e.value)
      np.testing.assert_allclose(result.n_e.value, core_profiles_t0.n_e.value)
    with self.subTest('boundary_conditions_updated'):
      # Boundary conditions are updated to values at t1.
      assert result.T_i.right_face_constraint is not None
      np.testing.assert_allclose(result.T_i.right_face_constraint, 2.5)

      assert result.T_e.right_face_constraint is not None
      np.testing.assert_allclose(result.T_e.right_face_constraint, 3.5)

      assert result.n_e.right_face_constraint is not None
      np.testing.assert_allclose(result.n_e.right_face_constraint, 5.5e19)
      self.assertIsNotNone(result.toroidal_angular_velocity)

  def test_update_core_and_source_profiles_updates_fast_ions_before_neoclassical(
      self,
  ):
    config_dict = default_configs.get_default_config_dict()
    config_dict['geometry'] = {'geometry_type': 'circular', 'n_rho': 4}
    config_dict['numerics'] = {
        'evolve_current': True,
        'enable_fast_ions': True,
    }
    config_dict['plasma_composition'] = {
        'main_ion': {'D': 0.5, 'T': 0.5},
        'impurity': {
            'impurity_mode': 'n_e_ratios',
            'species': {'He3': 0.03},
        },
    }
    config_dict['sources'] = {
        'icrh': {
            'model_name': 'scaled_profile',
            'minority_species': 'He3',
            'is_explicit': False,
        },
    }
    config_dict['profile_conditions']['fast_ions'] = [{
        'source': 'icrh',
        'species': 'He3',
        'n': {0.0: {0.0: 1e19, 1.0: 1e19}},
        'T': {0.0: {0.0: 100.0, 1.0: 100.0}},
        'n_right_bc': 0.0,
        'T_right_bc': 0.0,
    }]
    torax_config = config.ToraxConfig.from_dict(config_dict)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    models = torax_config.build_models()
    core_profiles_t0, runtime_params_t0, geo = (
        core_profile_helpers.make_initial_inputs(torax_config)
    )
    runtime_params_t1 = provider(t=1.0)
    source_states = models.source_models.initial_state(runtime_params_t0)
    explicit_source_profiles, _ = source_profile_builders.build_source_profiles(
        source_models=models.source_models,
        source_states=source_states,
        runtime_params=runtime_params_t0,
        geo=geo,
        core_profiles=core_profiles_t0,
        explicit=True,
    )

    # Core profiles at t+dt initially has stale / empty fast ions.
    stale_core_profiles = dataclasses.replace(core_profiles_t0, fast_ions=())

    observed_core_profiles = []
    original_calc = (
        models.neoclassical_model.calculate_conductivity_current_and_rotation
    )

    def spy_calc(runtime_params, geo, core_profiles, **kwargs):
      observed_core_profiles.append(core_profiles)
      return original_calc(runtime_params, geo, core_profiles, **kwargs)

    with jax.disable_jit():
      with mock.patch.object(
          type(models.neoclassical_model),
          'calculate_conductivity_current_and_rotation',
          side_effect=spy_calc,
      ):
        (
            result,
            _,
            _,
            _,
        ) = updaters.update_core_and_source_profiles_after_step(
            dt=jnp.array(1.0),
            x_new=(core_profiles_t0.psi,),
            runtime_params_t_plus_dt=runtime_params_t1,
            geo=geo,
            core_profiles_t=core_profiles_t0,
            core_profiles_t_plus_dt=stale_core_profiles,
            explicit_source_profiles=explicit_source_profiles,
            source_models=models.source_models,
            neoclassical_model=models.neoclassical_model,
            evolving_names=('psi',),
            source_states=source_states,
        )

    self.assertLen(observed_core_profiles, 1)
    passed_core_profiles = observed_core_profiles[0]
    self.assertNotEmpty(passed_core_profiles.fast_ions)
    passed_he3 = [
        fi for fi in passed_core_profiles.fast_ions if fi.species == 'He3'
    ]
    self.assertLen(passed_he3, 1)
    np.testing.assert_allclose(passed_he3[0].n.value, 1e19)
    self.assertNotEmpty(result.fast_ions)
    final_he3 = [fi for fi in result.fast_ions if fi.species == 'He3']
    self.assertLen(final_he3, 1)
    np.testing.assert_allclose(final_he3[0].n.value, 1e19)
    self.assertIn('icrh', [fi.source for fi in result.fast_ions])


if __name__ == '__main__':
  absltest.main()

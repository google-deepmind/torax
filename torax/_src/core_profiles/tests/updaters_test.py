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

from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.core_profiles import solver_state as solver_state_lib
from torax._src.core_profiles import updaters
from torax._src.geometry import circular_geometry
from torax._src.test_utils import core_profile_helpers
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import config


# pylint: disable=invalid-name
class UpdatersTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Build a default geo object for convenience.
    self.geo = circular_geometry.CircularConfig(n_rho=4).build_geometry()

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
    solver_state = updaters.provide_solver_state_t_plus_dt(
        dt=jnp.array(dt),
        runtime_params_t=runtime_params_t0,
        runtime_params_t_plus_dt=runtime_params_t1,
        geo_t_plus_dt=geo,
        core_profiles_t=core_profiles_t0,
    )

    # Check psi value
    expected_psi_array = np.full_like(geo.rho_norm, expected_psi_val_t1)
    np.testing.assert_allclose(
        solver_state.psi.value, expected_psi_array, rtol=1e-5
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

    solver_state = updaters.provide_solver_state_t_plus_dt(
        dt=jnp.array(1.0),
        runtime_params_t=runtime_params_t0,
        runtime_params_t_plus_dt=runtime_params_t1,
        geo_t_plus_dt=geo,
        core_profiles_t=core_profiles_t0,
    )

    # Should be same as t0 since not evolved and not prescribed
    np.testing.assert_allclose(
        solver_state.psi.value, core_profiles_t0.flux.psi.value
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
    np.testing.assert_allclose(core_profiles_t0.flux.psi.value, 10.0)

    runtime_params_t1 = provider(t=1.0)
    # At t=1, prescribed is 20.0. But evolve_current=True.

    solver_state = updaters.provide_solver_state_t_plus_dt(
        dt=jnp.array(1.0),
        runtime_params_t=runtime_params_t0,
        runtime_params_t_plus_dt=runtime_params_t1,
        geo_t_plus_dt=geo,
        core_profiles_t=core_profiles_t0,
    )

    # Expect psi value to NOT be updated to 20.0 (it should stay as t0 value
    # because provide_solver_state_t_plus_dt only updates prescribed/BCs, and
    # evolving vars are handled by the solver).
    # Since it wasn't updated in provide_..., it should remain 10.0 here.
    np.testing.assert_allclose(solver_state.psi.value, 10.0)

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

    self.assertIsInstance(result, solver_state_lib.SolverState)
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

    self.assertIsInstance(result, solver_state_lib.SolverState)
    with self.subTest('evolving_values_preserved'):
      # Evolving channels retain values from t0.
      np.testing.assert_allclose(
          result.psi.value, core_profiles_t0.flux.psi.value
      )
      np.testing.assert_allclose(
          result.T_i.value, core_profiles_t0.kinetic.T_i.value
      )
      np.testing.assert_allclose(
          result.T_e.value, core_profiles_t0.kinetic.T_e.value
      )
      np.testing.assert_allclose(
          result.n_e.value, core_profiles_t0.kinetic.n_e.value
      )
    with self.subTest('boundary_conditions_updated'):
      # Boundary conditions are updated to values at t1.
      assert result.T_i.right_face_constraint is not None
      np.testing.assert_allclose(result.T_i.right_face_constraint, 2.5)

      assert result.T_e.right_face_constraint is not None
      np.testing.assert_allclose(result.T_e.right_face_constraint, 3.5)

      assert result.n_e.right_face_constraint is not None
      np.testing.assert_allclose(result.n_e.right_face_constraint, 5.5e19)
      self.assertIsNotNone(result.toroidal_angular_velocity)


if __name__ == '__main__':
  absltest.main()

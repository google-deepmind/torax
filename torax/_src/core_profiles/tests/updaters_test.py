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

import copy
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
import numpy as np
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.core_profiles import updaters
from torax._src.fvm import cell_variable
from torax._src.geometry import circular_geometry
from torax._src.physics import formulas
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config


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
    pressure_thermal_e = cell_variable.CellVariable(
        value=jnp.ones_like(self.geo.rho_norm),
        face_centers=self.geo.rho_face_norm,
        right_face_constraint=1.0,
        right_face_grad_constraint=None,
    )
    pressure_thermal_i = cell_variable.CellVariable(
        value=jnp.ones_like(self.geo.rho_norm),
        face_centers=self.geo.rho_face_norm,
        right_face_constraint=1.0,
        right_face_grad_constraint=None,
    )
    pressure_thermal_total = cell_variable.CellVariable(
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
        pressure_thermal_e=pressure_thermal_e,
        pressure_thermal_i=pressure_thermal_i,
        pressure_thermal_total=pressure_thermal_total,
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
    config = default_configs.get_default_config_dict()
    config['profile_conditions'] = {
        'psi': psi,
        'Ip': 15e6,  # Ip is required but not used for psi value here
    }
    config['numerics'] = {
        'evolve_current': False,
    }
    config['profile_conditions']['initial_psi_mode'] = 'profile_conditions'

    torax_config = model_config.ToraxConfig.from_dict(config)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )

    # Initialize at t=0
    runtime_params_t0 = provider(t=0.0)
    geo = torax_config.geometry.build_provider(t=0.0)
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()

    core_profiles_t0 = initialization.initial_core_profiles(
        runtime_params_t0,
        geo,
        source_models,
        neoclassical_models,
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
    config = default_configs.get_default_config_dict()
    # Don't provide psi in profile_conditions, it defaults to None
    config['numerics'] = {
        'evolve_current': False,
    }
    config['profile_conditions']['psi'] = {
        0.0: 10.0
    }  # Set initial only via config for init

    torax_config = model_config.ToraxConfig.from_dict(config)

    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params_t0 = provider(t=0.0)
    geo = torax_config.geometry.build_provider(t=0.0)
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()

    core_profiles_t0 = initialization.initial_core_profiles(
        runtime_params_t0,
        geo,
        source_models,
        neoclassical_models,
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
    config = default_configs.get_default_config_dict()
    config['numerics'] = {'evolve_current': True}
    config['profile_conditions'] = {
        'psi': {
            0.0: {0.0: 10.0, 1.0: 10.0},
            1.0: {0.0: 20.0, 1.0: 20.0},
        },  # Prescribed varying
        'initial_psi_mode': 'profile_conditions',
    }

    torax_config = model_config.ToraxConfig.from_dict(config)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params_t0 = provider(t=0.0)
    geo = torax_config.geometry.build_provider(t=0.0)
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()

    core_profiles_t0 = initialization.initial_core_profiles(
        runtime_params_t0,
        geo,
        source_models,
        neoclassical_models,
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

  def test_update_energy_state(self):
    """Tests that energy state is updated correctly."""
    config = default_configs.get_default_config_dict()
    torax_config = model_config.ToraxConfig.from_dict(config)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params = provider(t=0.0)

    energy_state_t = initialization._initialise_internal_energy(
        runtime_params, self.core_profiles_t, self.geo
    )

    # Pretend we take a step of half the window size and increase the electron
    # and ion pressure.
    mock_dt = runtime_params.numerics.dW_dt_window / 2.0
    core_profiles_t_plus_dt = copy.deepcopy(self.core_profiles_t)
    core_profiles_t_plus_dt.pressure_thermal_e = cell_variable.CellVariable(
        value=jnp.full_like(self.geo.rho_norm, 2.0),
        face_centers=self.geo.rho_face_norm,
        right_face_constraint=1.0,
        right_face_grad_constraint=None,
    )
    core_profiles_t_plus_dt.pressure_thermal_i = cell_variable.CellVariable(
        value=jnp.full_like(self.geo.rho_norm, 3.0),
        face_centers=self.geo.rho_face_norm,
        right_face_constraint=1.0,
        right_face_grad_constraint=None,
    )
    W_thermal_e_t_plus_dt, W_thermal_i_t_plus_dt, _ = (
        formulas.calculate_stored_thermal_energy(
            core_profiles_t_plus_dt.pressure_thermal_e,
            core_profiles_t_plus_dt.pressure_thermal_i,
            core_profiles_t_plus_dt.pressure_thermal_total,
            self.geo,
        )
    )

    # Get the new energy state from the update function.
    energy_state_t_plus_dt = updaters._update_energy_state(
        runtime_params,
        self.geo,
        core_profiles_t_plus_dt,
        energy_state_t,
        mock_dt,
    )

    # Check that the time history is updated.
    expected_t_history = jnp.concatenate([
        energy_state_t.t_history[1:],
        jnp.atleast_1d(energy_state_t.t_history[-1]) + mock_dt,
    ])
    np.testing.assert_allclose(
        energy_state_t_plus_dt.t_history, expected_t_history
    )

    # Check that the W_thermal history is updated.
    expected_W_thermal_i_history = jnp.concatenate([
        energy_state_t.W_thermal_i_history[1:],
        jnp.atleast_1d(W_thermal_i_t_plus_dt),
    ])
    np.testing.assert_allclose(
        energy_state_t_plus_dt.W_thermal_i_history, expected_W_thermal_i_history
    )
    expected_W_thermal_e_history = jnp.concatenate([
        energy_state_t.W_thermal_e_history[1:],
        jnp.atleast_1d(W_thermal_e_t_plus_dt),
    ])
    np.testing.assert_allclose(
        energy_state_t_plus_dt.W_thermal_e_history, expected_W_thermal_e_history
    )

    # Check that the dW_dt is calculated correctly.
    np.testing.assert_allclose(
        energy_state_t_plus_dt.dW_thermal_i_dt,
        (W_thermal_i_t_plus_dt - energy_state_t.W_thermal_i) / mock_dt,
    )
    np.testing.assert_allclose(
        energy_state_t_plus_dt.dW_thermal_e_dt,
        (W_thermal_e_t_plus_dt - energy_state_t.W_thermal_e) / mock_dt,
    )

    # As we took a step of half the window size, the smoothed dW_dt should be
    # the same as the un-smoothed dW_dt.
    np.testing.assert_allclose(
        energy_state_t_plus_dt.dW_thermal_i_dt_smoothed,
        energy_state_t_plus_dt.dW_thermal_i_dt,
    )
    np.testing.assert_allclose(
        energy_state_t_plus_dt.dW_thermal_e_dt_smoothed,
        energy_state_t_plus_dt.dW_thermal_e_dt,
    )

    # Take another step and check the dW_dt.
    energy_state_t_plus_2dt = updaters._update_energy_state(
        runtime_params,
        self.geo,
        core_profiles_t_plus_dt,
        energy_state_t_plus_dt,
        mock_dt,
    )
    # Raw dW_dt values should be zero as we haven't changed the pressures.
    np.testing.assert_allclose(
        energy_state_t_plus_2dt.dW_thermal_i_dt,
        0.0,
    )
    np.testing.assert_allclose(
        energy_state_t_plus_2dt.dW_thermal_e_dt,
        0.0,
    )
    # Smoothed dW_dt values should be computed vs the 0th state
    np.testing.assert_allclose(
        energy_state_t_plus_2dt.dW_thermal_i_dt_smoothed,
        (energy_state_t_plus_2dt.W_thermal_i - energy_state_t.W_thermal_i)
        / (runtime_params.numerics.dW_dt_window),
    )
    np.testing.assert_allclose(
        energy_state_t_plus_2dt.dW_thermal_e_dt_smoothed,
        (energy_state_t_plus_2dt.W_thermal_e - energy_state_t.W_thermal_e)
        / (runtime_params.numerics.dW_dt_window),
    )


if __name__ == '__main__':
  absltest.main()

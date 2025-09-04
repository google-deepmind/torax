### FILEPATH: torax/mhd/sawtooth/tests/sawtooth_model_test.py
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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.orchestration import initial_state as initial_state_lib
from torax._src.orchestration import step_function
from torax._src.torax_pydantic import model_config

_NRHO = 10
_CRASH_STEP_DURATION = 1e-3
_FIXED_DT = 0.1

# Needed since we do not call torax.__init__ in this test, which normally sets
# this.
jax.config.update('jax_enable_x64', True)


class SawtoothModelTest(parameterized.TestCase):
  """Sawtooth model integration tests by running the SimulationStepFn."""

  def setUp(self):
    super().setUp()
    test_config_dict = {
        'numerics': {
            'evolve_current': True,
            'evolve_density': True,
            'evolve_ion_heat': True,
            'evolve_electron_heat': True,
            'fixed_dt': _FIXED_DT,
        },
        # Default initial current will lead to a sawtooth being triggered.
        'profile_conditions': {
            'Ip': 13e6,
            'initial_j_is_total_current': True,
            'initial_psi_from_j': True,
            'current_profile_nu': 3,
            'n_e_nbar_is_fGW': True,
            'normalize_n_e_to_nbar': True,
            'nbar': 0.85,
            'n_e': {0: {0.0: 1.5, 1.0: 1.0}},
        },
        'plasma_composition': {},
        'geometry': {'geometry_type': 'circular', 'n_rho': _NRHO},
        'pedestal': {},
        'sources': {'ohmic': {}},
        'solver': {
            'solver_type': 'linear',
            'use_pereverzev': False,
        },
        'time_step_calculator': {'calculator_type': 'fixed'},
        'transport': {'model_name': 'constant'},
        'mhd': {
            'sawtooth': {
                'trigger_model': {
                    'model_name': 'simple',
                    'minimum_radius': 0.2,
                    's_critical': 0.2,
                },
                'redistribution_model': {
                    'model_name': 'simple',
                    'flattening_factor': 1.01,
                    'mixing_radius_multiplier': 1.5,
                },
                'crash_step_duration': _CRASH_STEP_DURATION,
            }
        },
    }
    torax_config = model_config.ToraxConfig.from_dict(test_config_dict)
    self._torax_config = torax_config

    solver = torax_config.solver.build_solver(
        physics_models=torax_config.build_physics_models(),
    )

    geometry_provider = torax_config.geometry.build_provider
    self.runtime_params_provider = (
        build_runtime_params.RuntimeParamsProvider.from_config(torax_config)
    )

    self.step_fn = step_function.SimulationStepFn(
        solver=solver,
        time_step_calculator=torax_config.time_step_calculator.time_step_calculator,
        geometry_provider=geometry_provider,
        runtime_params_provider=self.runtime_params_provider,
    )

    self.initial_state, self.initial_post_processed_outputs = (
        initial_state_lib.get_initial_state_and_post_processed_outputs(
            t=torax_config.numerics.t_initial,
            runtime_params_provider=self.runtime_params_provider,
            geometry_provider=geometry_provider,
            step_fn=self.step_fn,
        )
    )

  def test_sawtooth_crash(self):
    """Tests that default values lead to crash and compares post-crash to ref."""
    output_state, _ = self.step_fn(
        input_state=self.initial_state,
        previous_post_processed_outputs=self.initial_post_processed_outputs,
    )
    sim_error = step_function.check_for_errors(
        self.runtime_params_provider.numerics,
        output_state,
        self.initial_post_processed_outputs,
    )

    np.testing.assert_equal(sim_error, state.SimError.NO_ERROR)
    np.testing.assert_equal(
        output_state.solver_numeric_outputs.sawtooth_crash, np.array(True)
    )
    np.testing.assert_equal(output_state.dt, np.array(_CRASH_STEP_DURATION))
    np.testing.assert_array_equal(
        output_state.t, self.initial_state.t + np.array(_CRASH_STEP_DURATION)
    )

    np.testing.assert_allclose(
        output_state.core_profiles.T_e.value,
        _POST_CRASH_TEMPERATURE,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        output_state.core_profiles.n_e.value, _POST_CRASH_N, rtol=1e-6
    )
    np.testing.assert_allclose(
        output_state.core_profiles.psi.value, _POST_CRASH_PSI, rtol=1e-6
    )

  def test_no_sawtooth_crash(self):
    """Tests that if q>1, no crash occurs."""
    raised_q_face = self.initial_state.core_profiles.q_face * 2
    initial_state = dataclasses.replace(
        self.initial_state,
        core_profiles=dataclasses.replace(
            self.initial_state.core_profiles, q_face=raised_q_face
        ),
    )
    output_state, _ = self.step_fn(
        input_state=initial_state,
        previous_post_processed_outputs=self.initial_post_processed_outputs,
    )
    sim_error = step_function.check_for_errors(
        self.runtime_params_provider.numerics,
        output_state,
        self.initial_post_processed_outputs,
    )
    np.testing.assert_equal(sim_error, state.SimError.NO_ERROR)
    np.testing.assert_equal(
        output_state.solver_numeric_outputs.sawtooth_crash, np.array(False)
    )
    np.testing.assert_equal(output_state.dt, np.array(_FIXED_DT))
    np.testing.assert_equal(
        output_state.t, np.array(self.initial_state.t + _FIXED_DT)
    )

  def test_no_subsequent_sawtooth_crashes(self):
    """Tests for no subsequent sawtooth crashes even if q in trigger condition."""
    # This crashes
    output_state0, post_processed_outputs0 = self.step_fn(
        input_state=self.initial_state,
        previous_post_processed_outputs=self.initial_post_processed_outputs,
    )

    # q is in trigger condition, but sawtooth_crash is True so no crash.
    new_input_state_should_not_crash = dataclasses.replace(
        output_state0,
        core_profiles=dataclasses.replace(
            self.initial_state.core_profiles,
            q_face=self.initial_state.core_profiles.q_face,
        ),
    )

    # Check that the sawtooth is indeed triggered if sawtooth_crash is
    # set to False.
    new_input_state_should_crash = dataclasses.replace(
        output_state0,
        core_profiles=dataclasses.replace(
            self.initial_state.core_profiles,
            q_face=self.initial_state.core_profiles.q_face,
        ),
        solver_numeric_outputs=state.SolverNumericOutputs(sawtooth_crash=False),
    )

    with self.subTest('no_subsequent_sawtooth_crashes'):
      output_state_should_not_crash, _ = self.step_fn(
          input_state=new_input_state_should_not_crash,
          previous_post_processed_outputs=post_processed_outputs0,
      )
      sim_error = step_function.check_for_errors(
          self.runtime_params_provider.numerics,
          output_state_should_not_crash,
          post_processed_outputs0,
      )
      np.testing.assert_equal(sim_error, state.SimError.NO_ERROR)
      np.testing.assert_equal(
          output_state_should_not_crash.solver_numeric_outputs.sawtooth_crash,
          np.array(False),
      )
      np.testing.assert_equal(
          output_state_should_not_crash.dt, np.array(_FIXED_DT)
      )
      np.testing.assert_array_equal(
          output_state_should_not_crash.t,
          self.initial_state.t + np.array(_CRASH_STEP_DURATION + _FIXED_DT),
      )

    with self.subTest('crashes_if_sawtooth_crash_is_false'):
      output_state_should_crash, _ = self.step_fn(
          input_state=new_input_state_should_crash,
          previous_post_processed_outputs=post_processed_outputs0,
      )
      np.testing.assert_equal(sim_error, state.SimError.NO_ERROR)
      np.testing.assert_equal(
          output_state_should_crash.solver_numeric_outputs.sawtooth_crash,
          np.array(True),
      )
      np.testing.assert_equal(
          output_state_should_crash.dt, np.array(_CRASH_STEP_DURATION)
      )
      np.testing.assert_array_equal(
          output_state_should_crash.t,
          self.initial_state.t + np.array(2 * _CRASH_STEP_DURATION),
      )


_POST_CRASH_TEMPERATURE = np.array([
    9.80214764,
    9.77449557,
    9.74682154,
    9.71912539,
    9.69140691,
    8.17937075,
    6.2258966,
    4.5,
    3.1,
    1.7,
])

_POST_CRASH_N = np.array([
    0.92905438e20,
    0.92652621e20,
    0.92399804e20,
    0.92146987e20,
    0.91894169e20,
    0.88178024e20,
    0.8345057e20,
    0.79219014e20,
    0.75698169e20,
    0.72177324e20,
])

_POST_CRASH_PSI = np.array([
    9.778742,
    11.342102,
    14.360384,
    18.737049,
    24.378128,
    31.058185,
    38.126174,
    44.844899,
    50.742815,
    55.729866,
])


if __name__ == '__main__':
  absltest.main()

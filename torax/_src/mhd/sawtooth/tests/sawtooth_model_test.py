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

"""Sawtooth model integration tests."""

import dataclasses
import json
import pathlib

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.mhd.sawtooth.tests import regenerate_sawtooth_refs
from torax._src.orchestration import initial_state as initial_state_lib
from torax._src.orchestration import step_function
from torax._src.torax_pydantic import model_config


# Import shared constants from the regeneration script
_NRHO = regenerate_sawtooth_refs.NRHO
_CRASH_STEP_DURATION = regenerate_sawtooth_refs.CRASH_STEP_DURATION
_FIXED_DT = regenerate_sawtooth_refs.FIXED_DT

# Needed since we do not call torax.__init__ in this test, which normally sets
# this.
jax.config.update('jax_enable_x64', True)


def _load_sawtooth_references() -> dict:
  """Loads sawtooth reference values from the local JSON file."""
  json_path = pathlib.Path(__file__).parent / regenerate_sawtooth_refs.REFERENCES_FILE
  with open(json_path, 'r') as f:
    return json.load(f)


class SawtoothModelTest(parameterized.TestCase):
  """Sawtooth model integration tests by running the SimulationStepFn."""

  def setUp(self):
    super().setUp()
    # Use the shared test configuration from the regeneration script
    test_config_dict = regenerate_sawtooth_refs.get_sawtooth_test_config()
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
            step_fn=self.step_fn,
        )
    )

    # Load sawtooth crash reference values from local JSON
    sawtooth_refs = _load_sawtooth_references()
    self._post_crash_temperature = np.array(sawtooth_refs['post_crash_temperature'])
    self._post_crash_n = np.array(sawtooth_refs['post_crash_n'])
    self._post_crash_psi = np.array(sawtooth_refs['post_crash_psi'])

  def test_sawtooth_crash(self):
    """Tests that default values lead to crash and compares post-crash to ref."""
    output_state, _ = self.step_fn(
        input_state=self.initial_state,
        previous_post_processed_outputs=self.initial_post_processed_outputs,
    )
    sim_error = self.step_fn.check_for_errors(
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
        self._post_crash_temperature,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        output_state.core_profiles.n_e.value,
        self._post_crash_n,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        output_state.core_profiles.psi.value,
        self._post_crash_psi,
        rtol=1e-6,
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
    sim_error = self.step_fn.check_for_errors(
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
        solver_numeric_outputs=state.SolverNumericOutputs(
            inner_solver_iterations=np.array(0, np.int64),
            outer_solver_iterations=np.array(0, np.int64),
            solver_error_state=np.array(0, np.int64),
            sawtooth_crash=False,
        ),
    )

    with self.subTest('no_subsequent_sawtooth_crashes'):
      output_state_should_not_crash, _ = self.step_fn(
          input_state=new_input_state_should_not_crash,
          previous_post_processed_outputs=post_processed_outputs0,
      )
      sim_error = self.step_fn.check_for_errors(
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


if __name__ == '__main__':
  absltest.main()

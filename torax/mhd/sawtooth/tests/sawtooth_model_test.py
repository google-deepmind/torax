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
from torax import state
from torax.config import build_runtime_params
from torax.orchestration import initial_state as initial_state_lib
from torax.orchestration import step_function
from torax.sources import source_models as source_models_lib
from torax.torax_pydantic import model_config

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
            'current_eq': True,
            'dens_eq': True,
            'ion_heat_eq': True,
            'el_heat_eq': True,
            'fixed_dt': _FIXED_DT,
        },
        # Default initial current will lead to a sawtooth being triggered.
        'profile_conditions': {
            'Ip_tot': 13,
            'initial_j_is_total_current': True,
            'initial_psi_from_j': True,
            'nu': 3,
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
        'transport': {'transport_model': 'constant'},
        'mhd': {
            'sawtooth': {
                'trigger_model': {
                    'trigger_model_type': 'simple',
                    'minimum_radius': 0.2,
                    's_critical': 0.2,
                },
                'redistribution_model': {
                    'redistribution_model_type': 'simple',
                    'flattening_factor': 1.01,
                    'mixing_radius_multiplier': 1.5,
                },
                'crash_step_duration': _CRASH_STEP_DURATION,
            }
        },
    }
    torax_config = model_config.ToraxConfig.from_dict(test_config_dict)

    transport_model = torax_config.transport.build_transport_model()
    pedestal_model = torax_config.pedestal.build_pedestal_model()

    source_models = source_models_lib.SourceModels(
        torax_config.sources.source_model_config
    )

    solver = torax_config.solver.build_solver(
        transport_model=transport_model,
        source_models=source_models,
        pedestal_model=pedestal_model,
    )

    mhd_models = torax_config.mhd.build_mhd_models()

    self.geometry_provider = torax_config.geometry.build_provider

    self.static_runtime_params_slice = (
        build_runtime_params.build_static_params_from_config(torax_config)
    )

    self.dynamic_runtime_params_slice_provider = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            torax_config
        )
    )

    self.step_fn = step_function.SimulationStepFn(
        solver=solver,
        time_step_calculator=torax_config.time_step_calculator.time_step_calculator,
        transport_model=transport_model,
        pedestal_model=pedestal_model,
        mhd_models=mhd_models,
    )

    self.initial_state, self.initial_post_processed_outputs = (
        initial_state_lib.get_initial_state_and_post_processed_outputs(
            t=torax_config.numerics.t_initial,
            static_runtime_params_slice=self.static_runtime_params_slice,
            dynamic_runtime_params_slice_provider=self.dynamic_runtime_params_slice_provider,
            geometry_provider=self.geometry_provider,
            step_fn=self.step_fn,
        )
    )

  def test_sawtooth_crash(self):
    """Tests that default values lead to crash and compares post-crash to ref."""
    output_state, _, sim_error = self.step_fn(
        static_runtime_params_slice=self.static_runtime_params_slice,
        dynamic_runtime_params_slice_provider=self.dynamic_runtime_params_slice_provider,
        geometry_provider=self.geometry_provider,
        input_state=self.initial_state,
        previous_post_processed_outputs=self.initial_post_processed_outputs,
    )

    np.testing.assert_equal(sim_error, state.SimError.NO_ERROR)
    np.testing.assert_equal(output_state.sawtooth_crash, np.array(True))
    np.testing.assert_equal(output_state.dt, np.array(_CRASH_STEP_DURATION))
    np.testing.assert_equal(
        output_state.t, self.initial_state.t + np.array(_CRASH_STEP_DURATION)
    )

    np.testing.assert_allclose(
        output_state.core_profiles.temp_el.value,
        _POST_CRASH_TEMPERATURE,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        output_state.core_profiles.ne.value, _POST_CRASH_N, rtol=1e-6
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
    output_state, _, sim_error = self.step_fn(
        static_runtime_params_slice=self.static_runtime_params_slice,
        dynamic_runtime_params_slice_provider=self.dynamic_runtime_params_slice_provider,
        geometry_provider=self.geometry_provider,
        input_state=initial_state,
        previous_post_processed_outputs=self.initial_post_processed_outputs,
    )
    np.testing.assert_equal(sim_error, state.SimError.NO_ERROR)
    np.testing.assert_equal(output_state.sawtooth_crash, np.array(False))
    np.testing.assert_equal(output_state.dt, np.array(_FIXED_DT))
    np.testing.assert_equal(
        output_state.t, self.initial_state.t + np.array(_FIXED_DT)
    )

  def test_no_subsequent_sawtooth_crashes(self):
    """Tests for no subsequent sawtooth crashes even if q in trigger condition."""
    # This crashes
    output_state0, post_processed_outputs0, _ = self.step_fn(
        static_runtime_params_slice=self.static_runtime_params_slice,
        dynamic_runtime_params_slice_provider=self.dynamic_runtime_params_slice_provider,
        geometry_provider=self.geometry_provider,
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
        sawtooth_crash=False,
    )

    with self.subTest('no_subsequent_sawtooth_crashes'):
      output_state_should_not_crash, _, sim_error = self.step_fn(
          static_runtime_params_slice=self.static_runtime_params_slice,
          dynamic_runtime_params_slice_provider=self.dynamic_runtime_params_slice_provider,
          geometry_provider=self.geometry_provider,
          input_state=new_input_state_should_not_crash,
          previous_post_processed_outputs=post_processed_outputs0,
      )
      np.testing.assert_equal(sim_error, state.SimError.NO_ERROR)
      np.testing.assert_equal(
          output_state_should_not_crash.sawtooth_crash, np.array(False)
      )
      np.testing.assert_equal(
          output_state_should_not_crash.dt, np.array(_FIXED_DT)
      )
      np.testing.assert_equal(
          output_state_should_not_crash.t,
          self.initial_state.t + np.array(_CRASH_STEP_DURATION + _FIXED_DT),
      )

    with self.subTest('crashes_if_sawtooth_crash_is_false'):
      output_state_should_crash, _, sim_error = self.step_fn(
          static_runtime_params_slice=self.static_runtime_params_slice,
          dynamic_runtime_params_slice_provider=self.dynamic_runtime_params_slice_provider,
          geometry_provider=self.geometry_provider,
          input_state=new_input_state_should_crash,
          previous_post_processed_outputs=post_processed_outputs0,
      )
      np.testing.assert_equal(sim_error, state.SimError.NO_ERROR)
      np.testing.assert_equal(
          output_state_should_crash.sawtooth_crash, np.array(True)
      )
      np.testing.assert_equal(
          output_state_should_crash.dt, np.array(_CRASH_STEP_DURATION)
      )
      np.testing.assert_equal(
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
    0.92905438,
    0.92652621,
    0.92399804,
    0.92146987,
    0.91894169,
    0.88178024,
    0.8345057,
    0.79219014,
    0.75698169,
    0.72177324,
])

_POST_CRASH_PSI = np.array([
    9.7786644,
    11.34198514,
    14.36022924,
    18.73686254,
    24.37792036,
    31.05797517,
    38.12598722,
    44.84476207,
    50.74274513,
    55.72986591,
])


if __name__ == '__main__':
  absltest.main()

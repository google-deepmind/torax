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
import os
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax.numpy as jnp
import numpy as np
from torax._src import state
from torax._src.config import config_loader
from torax._src.orchestration import explicit_model_outputs as explicit_model_outputs_lib
from torax._src.orchestration import run_simulation
from torax._src.orchestration import sim_state as sim_state_lib
from torax._src.orchestration import step_function
from torax._src.output_tools import post_processing
from torax._src.test_utils import default_configs
from torax._src.test_utils import paths
from torax._src.torax_pydantic import config
from torax._src.torax_pydantic import interpolated_param_1d


def get_step_fn_sim_state_and_post_processed_outputs(
    config_dict: dict[str, Any] | None = None,
) -> tuple[
    step_function.SimulationStepFn,
    sim_state_lib.SimState,
    post_processing.PostProcessedOutputs,
]:
  if config_dict is None:
    config_dict = default_configs.get_default_config_dict()
  torax_config = config.ToraxConfig.from_dict(config_dict)
  (
      sim_state,
      post_processed_outputs,
      step_fn,
  ) = run_simulation.prepare_simulation(torax_config)
  return step_fn, sim_state, post_processed_outputs


class StepFunctionTest(parameterized.TestCase):

  def test_no_error(self):
    step_fn, sim_state, post_processed_outputs = (
        get_step_fn_sim_state_and_post_processed_outputs()
    )
    error = step_fn.check_for_errors(
        sim_state,
        post_processed_outputs,
    )
    self.assertEqual(error, state.SimError.NO_ERROR)

  def test_nan_in_bc(self):
    step_fn, sim_state, post_processed_outputs = (
        get_step_fn_sim_state_and_post_processed_outputs()
    )
    core_profiles = dataclasses.replace(
        sim_state.core_profiles,
        kinetic=dataclasses.replace(
            sim_state.core_profiles.kinetic,
            T_i=dataclasses.replace(
                sim_state.core_profiles.kinetic.T_i,
                right_face_constraint=jnp.array(jnp.nan),
            ),
        ),
    )
    new_sim_state_core_profiles = dataclasses.replace(
        sim_state, core_profiles=core_profiles
    )
    error = step_fn.check_for_errors(
        new_sim_state_core_profiles,
        post_processed_outputs,
    )
    self.assertEqual(error, state.SimError.NAN_DETECTED)

  def test_nan_in_post_processed_outputs(self):
    step_fn, sim_state, post_processed_outputs = (
        get_step_fn_sim_state_and_post_processed_outputs()
    )
    new_post_processed_outputs = dataclasses.replace(
        post_processed_outputs,
        P_aux_total=jnp.array(jnp.nan),
    )
    error = step_fn.check_for_errors(
        sim_state,
        new_post_processed_outputs,
    )
    self.assertEqual(error, state.SimError.NAN_DETECTED)

  def test_nan_in_source_array(self):
    step_fn, sim_state, post_processed_outputs = (
        get_step_fn_sim_state_and_post_processed_outputs()
    )
    nan_array = np.zeros_like(sim_state.geometry.rho)
    nan_array[-1] = np.nan
    bootstrap_current = dataclasses.replace(
        sim_state.core_sources.bootstrap_current,
        j_parallel_bootstrap=nan_array,
    )
    new_core_sources = dataclasses.replace(
        sim_state.core_sources, bootstrap_current=bootstrap_current
    )
    new_sim_state_sources = dataclasses.replace(
        sim_state, core_sources=new_core_sources
    )
    error = step_fn.check_for_errors(
        new_sim_state_sources,
        post_processed_outputs,
    )
    self.assertEqual(error, state.SimError.NAN_DETECTED)

  def test_below_min_dt(self):
    config_dict = default_configs.get_default_config_dict()
    config_dict['time_step_calculator'] = {
        'min_dt': 2.0,
        'dt_backtracking_factor': 2.0,
    }
    step_fn, sim_state, post_processed_outputs = (
        get_step_fn_sim_state_and_post_processed_outputs(config_dict)
    )

    new_sim_state = dataclasses.replace(
        sim_state,
        dt=jnp.array(3.0),
        solver_numeric_outputs=state.SolverNumericOutputs(
            solver_error_state=1,
            outer_solver_iterations=0,
            inner_solver_iterations=0,
            sawtooth_crash=False,
        ),
    )
    error = step_fn.check_for_errors(
        new_sim_state,
        post_processed_outputs,
    )
    self.assertEqual(error, state.SimError.REACHED_MIN_DT)

  def test_no_error_when_below_min_dt_but_solver_converged(self):
    config_dict = default_configs.get_default_config_dict()
    config_dict['time_step_calculator'] = {
        't_final': 5.0,
        'min_dt': 2.0,
        'dt_backtracking_factor': 2.0,
        'exact_t_final': True,
    }
    step_fn, sim_state, post_processed_outputs = (
        get_step_fn_sim_state_and_post_processed_outputs(config_dict)
    )
    new_sim_state = dataclasses.replace(
        sim_state,
        dt=jnp.array(1.0),
        t=jnp.array(5.0),
        solver_numeric_outputs=state.SolverNumericOutputs(
            solver_error_state=0,
            outer_solver_iterations=0,
            inner_solver_iterations=0,
            sawtooth_crash=False,
        ),
    )
    error = step_fn.check_for_errors(
        new_sim_state,
        post_processed_outputs,
    )
    self.assertEqual(error, state.SimError.NO_ERROR)

  def test_fixed_step_with_smaller_passed_max_dt(self):
    config_dt = 0.1
    passed_max_dt = 0.01

    config_dict = default_configs.get_default_config_dict()
    config_dict['time_step_calculator'] = {
        'calculator_type': 'fixed',
        'dt': config_dt,
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    (
        sim_state,
        post_processed_outputs,
        step_fn,
    ) = run_simulation.prepare_simulation(torax_config)
    runtime_params = step_fn.runtime_params_provider(
        torax_config.time_step_calculator.t_initial
    )
    explicit_model_outputs = explicit_model_outputs_lib.ExplicitModelOutputs(
        sources={},
        pedestal=None,
    )
    output_state, _ = step_fn._fixed_step(
        max_dt=jnp.array(passed_max_dt),
        runtime_params_t=runtime_params,
        geo_t=sim_state.geometry,
        explicit_model_outputs=explicit_model_outputs,
        edge_outputs=None,
        input_state=sim_state,
        previous_post_processed_outputs=post_processed_outputs,
        runtime_params_provider=step_fn.runtime_params_provider,
        geometry_provider=step_fn.geometry_provider,
        model_state_t=sim_state.model_state,
    )
    self.assertGreater(config_dt, passed_max_dt)
    np.testing.assert_allclose(
        output_state.t, sim_state.t + passed_max_dt, atol=1e-7
    )

  def test_fixed_step_with_larger_passed_max_dt(self):
    # TODO(b/456188184): Improve test coverage for the fixed step function.
    config_dt = 0.1
    passed_max_dt = 1.0

    config_dict = default_configs.get_default_config_dict()
    config_dict['time_step_calculator'] = {
        'calculator_type': 'fixed',
        'dt': config_dt,
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    (
        sim_state,
        post_processed_outputs,
        step_fn,
    ) = run_simulation.prepare_simulation(torax_config)
    runtime_params = step_fn.runtime_params_provider(
        torax_config.time_step_calculator.t_initial
    )
    explicit_model_outputs = explicit_model_outputs_lib.ExplicitModelOutputs(
        sources={},
        pedestal=None,
    )
    output_state, _ = step_fn._fixed_step(
        max_dt=jnp.array(passed_max_dt),
        runtime_params_t=runtime_params,
        geo_t=sim_state.geometry,
        explicit_model_outputs=explicit_model_outputs,
        edge_outputs=None,
        input_state=sim_state,
        previous_post_processed_outputs=post_processed_outputs,
        runtime_params_provider=step_fn.runtime_params_provider,
        geometry_provider=step_fn.geometry_provider,
        model_state_t=sim_state.model_state,
    )
    self.assertGreater(passed_max_dt, config_dt)
    np.testing.assert_allclose(
        output_state.t, sim_state.t + config_dt, atol=1e-7
    )

  def test_adaptive_step_with_smaller_passed_max_dt(self):
    config_dt = 0.1
    passed_max_dt = 0.01

    config_dict = default_configs.get_default_config_dict()
    config_dict['time_step_calculator'] = {
        'calculator_type': 'fixed',
        'dt': config_dt,
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    (
        sim_state,
        post_processed_outputs,
        step_fn,
    ) = run_simulation.prepare_simulation(torax_config)
    runtime_params = step_fn.runtime_params_provider(
        torax_config.time_step_calculator.t_initial
    )
    explicit_model_outputs = explicit_model_outputs_lib.ExplicitModelOutputs(
        sources={},
        pedestal=None,
    )
    output_state, _ = step_fn._adaptive_step(
        max_dt=jnp.array(passed_max_dt),
        runtime_params_t=runtime_params,
        geo_t=sim_state.geometry,
        explicit_model_outputs=explicit_model_outputs,
        edge_outputs=None,
        input_state=sim_state,
        previous_post_processed_outputs=post_processed_outputs,
        runtime_params_provider=step_fn.runtime_params_provider,
        geometry_provider=step_fn.geometry_provider,
        model_state_t=sim_state.model_state,
    )
    self.assertTrue(np.less_equal(output_state.dt, passed_max_dt))

  def test_call_with_sawtooth_solver_smoke_test(self):
    """Smoke test for the boolean logic around the sawtooth solver.

    This cannot be testing using mock.patch because all the python code is
    called when compiling and then none is called when running the jax.lax.cond.
    """
    crash_step_duration = 1e-3
    passed_max_dt = 1e-2

    config_dict = default_configs.get_default_config_dict()
    config_dict['mhd'] = {
        'sawtooth': {
            'trigger_model': {
                'model_name': 'simple',
            },
            'redistribution_model': {
                'model_name': 'simple',
            },
            'crash_step_duration': crash_step_duration,
        }
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    (
        sim_state,
        post_processed_outputs,
        step_fn,
    ) = run_simulation.prepare_simulation(torax_config)

    _, _ = step_fn(
        sim_state,
        post_processed_outputs,
        max_dt=jnp.array(passed_max_dt),
    )

  def test_fixed_time_step_correct_time(self):
    config_dict = default_configs.get_default_config_dict()
    control_dt = 0.01
    torax_config = config.ToraxConfig.from_dict(config_dict)
    (
        sim_state,
        post_processed_outputs,
        step_fn,
    ) = run_simulation.prepare_simulation(torax_config)

    t_initial = sim_state.t
    output_state, _ = step_fn.fixed_time_step(
        jnp.array(control_dt),
        sim_state,
        post_processed_outputs,
    )
    np.testing.assert_allclose(
        output_state.t, t_initial + control_dt, atol=1e-7
    )
    np.testing.assert_allclose(output_state.dt, control_dt, atol=1e-7)

  def test_fixed_time_step_t_less_than_min_dt(self):
    config_dict = default_configs.get_default_config_dict()
    config_dict['time_step_calculator'] = {
        'min_dt': 0.1,
        'allow_backtracking': True,
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    (
        sim_state,
        post_processed_outputs,
        step_fn,
    ) = run_simulation.prepare_simulation(torax_config)
    output_state, _ = step_fn.fixed_time_step(
        jnp.array(0.01),
        sim_state,
        post_processed_outputs,
    )
    np.testing.assert_allclose(output_state.dt, 0.01, atol=1e-7)

  def test_fixed_time_step_exits_if_min_dt_reached(self):
    torax_config = config_loader.build_torax_config_from_file(
        os.path.join(
            paths.test_data_dir(), 'test_iterhybrid_predictor_corrector.py'
        )
    )
    torax_config.update_fields({
        'time_step_calculator.min_dt': 0.1,
        'time_step_calculator.allow_backtracking': True,
    })

    requested_t = 1.0
    sim_state, post_processed_outputs, step_fn = (
        run_simulation.prepare_simulation(torax_config)
    )
    sim_state, post_processed_outputs = step_fn.fixed_time_step(
        np.array(requested_t), sim_state, post_processed_outputs
    )

    # Check we exited early
    self.assertLess(sim_state.t, requested_t)

    # Check that the MIN_DT error is reported.
    sim_error = step_fn.check_for_errors(sim_state, post_processed_outputs)
    self.assertEqual(sim_error, state.SimError.REACHED_MIN_DT)

  def test_step_function_overrides(self):
    original_ip = 15e6
    config_dict = default_configs.get_default_config_dict()
    config_dict['profile_conditions']['I_p'] = original_ip
    torax_config = config.ToraxConfig.from_dict(config_dict)
    (
        sim_state,
        post_processed_outputs,
        step_fn,
    ) = run_simulation.prepare_simulation(torax_config)
    params_provider = step_fn.runtime_params_provider

    # Run a step with overridden I_p.
    assert params_provider.profile_conditions.I_p is not None
    ip_update = interpolated_param_1d.TimeVaryingScalarUpdate(
        value=params_provider.profile_conditions.I_p.value * 2.0  # pyrefly: ignore[bad-argument-type]
    )
    runtime_params_overrides = params_provider.update_provider(
        lambda x: (x.profile_conditions.I_p,),  # pyrefly: ignore[bad-argument-type]
        (ip_update,),
    )
    override_state, override_post_processed_outputs = step_fn(
        sim_state,
        post_processed_outputs,
        runtime_params_overrides=runtime_params_overrides,
    )

    # Update the config itself and re-run the step.
    torax_config.update_fields({'profile_conditions.I_p': original_ip * 2.0})
    step_fn = run_simulation.make_step_fn(torax_config)
    ref_state, ref_post_processed_outputs = step_fn(
        # Use original state and post-processed outputs as the initial value.
        sim_state,
        post_processed_outputs,
    )

    chex.assert_trees_all_close(override_state, ref_state)
    chex.assert_trees_all_close(
        override_post_processed_outputs, ref_post_processed_outputs
    )

  def test_step_function_geometry_overrides(self):
    config_dict = default_configs.get_default_config_dict()
    torax_config = config.ToraxConfig.from_dict(config_dict)
    (
        sim_state,
        post_processed_outputs,
        step_fn,
    ) = run_simulation.prepare_simulation(torax_config)

    # Construct a new step function with a different geometry.
    torax_config.update_fields({'geometry.calc_Phi_b_dot': False})
    new_step_fn = run_simulation.make_step_fn(torax_config)
    # Use original step function with new step function's geometry as overrides.
    override_state, override_post_processed_outputs = step_fn(
        sim_state,
        post_processed_outputs,
        geometry_overrides=new_step_fn.geometry_provider,
    )

    # Use the new step function.
    ref_state, ref_post_processed_outputs = new_step_fn(
        # Use original state and post-processed outputs as the initial value.
        sim_state,
        post_processed_outputs,
    )

    chex.assert_trees_all_close(override_state, ref_state)
    chex.assert_trees_all_close(
        override_post_processed_outputs, ref_post_processed_outputs
    )


if __name__ == '__main__':
  absltest.main()

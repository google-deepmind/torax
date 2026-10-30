# Copyright 2026 DeepMind Technologies Limited
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

import concurrent.futures
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax import numpy as jnp
import torax
from torax._src import jax_utils
from torax._src.orchestration import jit_run_loop
from torax._src.test_utils import default_configs
import torax.experimental as torax_experimental
import tqdm

# pylint: disable=invalid-name,protected-access

jax.config.update('jax_enable_x64', True)


class JitRunLoopTest(parameterized.TestCase):

  @parameterized.parameters('chi', 'from_previous_dt')
  def test_get_max_steps_non_fixed_raises_not_implemented_error(
      self, calculator_type
  ):
    config_dict = default_configs.get_default_config_dict()
    config_dict['time_step_calculator'] = {'calculator_type': calculator_type}
    torax_config = torax.ToraxConfig.from_dict(config_dict)
    step_fn = torax_experimental.make_step_fn(torax_config)

    with self.assertRaisesRegex(
        NotImplementedError, 'max_steps must be specified when using'
    ):
      jit_run_loop._get_max_steps(step_fn)

    with self.assertRaisesRegex(
        NotImplementedError, 'max_steps must be specified when using'
    ):
      jit_run_loop.run_loop(step_fn)

  @parameterized.named_parameters(
      dict(
          testcase_name='fixed_no_adaptive_exact_division',
          t_initial=0.0,
          t_final=5.0,
          dt=0.5,
          adaptive_dt=False,
          expected_max_steps=10,
      ),
      dict(
          testcase_name='fixed_no_adaptive_ceil_division',
          t_initial=0.0,
          t_final=5.0,
          dt=2.0,
          adaptive_dt=False,
          expected_max_steps=3,
      ),
      dict(
          testcase_name='fixed_no_adaptive_time_varying_dt',
          t_initial=0.0,
          t_final=5.0,
          dt={0.0: 2.0, 2.0: 0.5},
          adaptive_dt=False,
          expected_max_steps=10,
      ),
      dict(
          testcase_name='fixed_with_adaptive_exact_division',
          t_initial=0.0,
          t_final=5.0,
          dt=0.5,
          adaptive_dt=True,
          expected_max_steps=100,
      ),
      dict(
          testcase_name='fixed_with_adaptive_ceil_division',
          t_initial=0.0,
          t_final=5.0,
          dt=2.0,
          adaptive_dt=True,
          expected_max_steps=30,
      ),
  )
  def test_get_max_steps_fixed(
      self,
      t_initial,
      t_final,
      dt,
      adaptive_dt,
      expected_max_steps,
  ):
    config_dict = default_configs.get_default_config_dict()
    config_dict['time_step_calculator'] = {
        'calculator_type': 'fixed',
        't_initial': t_initial,
        't_final': t_final,
        'dt': dt,
        'allow_backtracking': adaptive_dt,
    }
    torax_config = torax.ToraxConfig.from_dict(config_dict)
    step_fn = torax_experimental.make_step_fn(torax_config)

    self.assertEqual(jit_run_loop._get_max_steps(step_fn), expected_max_steps)

  def test_run_loop_fixed_default_and_explicit_max_steps(self):
    config_dict = default_configs.get_default_config_dict()
    config_dict['time_step_calculator'] = {
        'calculator_type': 'fixed',
        't_initial': 0.0,
        't_final': 0.2,
        'dt': 0.1,
        'allow_backtracking': False,
    }
    torax_config = torax.ToraxConfig.from_dict(config_dict)
    step_fn = torax_experimental.make_step_fn(torax_config)

    states_default, outputs_default, sim_error_default = jit_run_loop.run_loop(
        step_fn
    )
    self.assertEqual(sim_error_default, torax.SimError.NO_ERROR)
    self.assertLen(states_default.t, 3)
    for leaf in jax.tree.leaves((states_default, outputs_default)):
      self.assertIsInstance(leaf, jax.Array)
      self.assertEqual(leaf.shape[0], 3)

    states_explicit, outputs_explicit, sim_error_explicit = (
        jit_run_loop.run_loop(step_fn, max_steps=5)
    )
    self.assertEqual(sim_error_explicit, torax.SimError.NO_ERROR)
    self.assertLen(states_explicit.t, 3)
    for leaf in jax.tree.leaves((states_explicit, outputs_explicit)):
      self.assertIsInstance(leaf, jax.Array)
      self.assertEqual(leaf.shape[0], 3)
      if jnp.issubdtype(leaf.dtype, jnp.floating):
        self.assertFalse(bool(jnp.any(jnp.isnan(leaf))))

  @parameterized.named_parameters(
      dict(
          testcase_name='after_valid_step',
          T_e={
              0.0: {0.0: 2.0, 1.0: 2.0},
              0.1: {0.0: 2.0, 1.0: 2.0},
              0.2: {0.0: 0.1, 1.0: 0.1},
          },
          expected_times=[0.0, 0.1],
      ),
      dict(
          testcase_name='at_t_initial',
          T_e={0.0: {0.0: 0.1, 1.0: 0.1}},
          expected_times=[0.0],
      ),
  )
  def test_run_loop_early_stop_preserves_sim_error_and_drops_invalid_step(
      self,
      T_e,
      expected_times,
  ):
    config_dict = default_configs.get_default_config_dict()
    config_dict['numerics'] = {
        'evolve_ion_heat': False,
        'evolve_electron_heat': False,
        'T_min_eV': 500.0,
    }
    config_dict['profile_conditions']['T_e'] = T_e
    config_dict['profile_conditions']['T_i'] = {0.0: {0.0: 2.0, 1.0: 2.0}}
    config_dict['time_step_calculator'] = {
        'calculator_type': 'fixed',
        't_initial': 0.0,
        't_final': 0.5,
        'dt': 0.1,
        'allow_backtracking': False,
    }
    torax_config = torax.ToraxConfig.from_dict(config_dict)
    step_fn = torax_experimental.make_step_fn(torax_config)

    states, outputs, sim_error = jit_run_loop.run_loop(step_fn, max_steps=5)

    self.assertEqual(sim_error, torax.SimError.LOW_TEMPERATURE_COLLAPSE)
    chex.assert_trees_all_close(states.t, jnp.asarray(expected_times))
    expected_num_states = len(expected_times)
    for leaf in jax.tree.leaves((states, outputs)):
      self.assertIsInstance(leaf, jax.Array)
      self.assertEqual(leaf.shape[0], expected_num_states)
      if jnp.issubdtype(leaf.dtype, jnp.floating):
        self.assertFalse(bool(jnp.any(jnp.isnan(leaf))))

  def test_max_steps_precedence(self):
    config_dict = default_configs.get_default_config_dict()
    config_dict['time_step_calculator'] = {
        'calculator_type': 'fixed',
        't_initial': 0.0,
        't_final': 1.0,
        'dt': 0.1,
        'allow_backtracking': False,
        'max_steps': 5,
    }
    torax_config = torax.ToraxConfig.from_dict(config_dict)
    step_fn = torax_experimental.make_step_fn(torax_config)

    # Config attribute (5) takes precedence over default_max_steps (10).
    self.assertEqual(jit_run_loop._get_max_steps(step_fn), 5)
    states_cfg, _, sim_error_cfg = jit_run_loop.run_loop(step_fn)
    self.assertEqual(sim_error_cfg, torax.SimError.DID_NOT_REACH_T_FINAL)
    self.assertLen(states_cfg.t, 6)

    # Explicit run_loop argument (2) takes precedence over config attribute (5).
    states_arg, _, sim_error_arg = jit_run_loop.run_loop(step_fn, max_steps=2)
    self.assertEqual(sim_error_arg, torax.SimError.DID_NOT_REACH_T_FINAL)
    self.assertLen(states_arg.t, 3)

  def test_gradient(self):

    torax_config = torax.build_torax_config_from_file(
        'examples/iterhybrid_rampup.py'
    )
    # TODO(b/575786554): Implement better handling of coarse-converged steps
    # during gradient passes. For now, disable the coarse-tolerance fallback so
    # that every step backtracks until it converges to full tolerance (required
    # for implicit differentiation).
    torax_config.update_fields({'solver.residual_coarse_tol': 1e-5})
    step_fn = torax_experimental.make_step_fn(torax_config)
    runtime_params_provider = step_fn.runtime_params_provider
    assert runtime_params_provider.profile_conditions.I_p is not None

    original_times = runtime_params_provider.profile_conditions.I_p.time
    start_time = original_times[0]
    end_time = original_times[-1]
    mid_time = (start_time + end_time) / 2
    I_p_new_times = jnp.array([start_time, mid_time, end_time])

    @jax.jit
    def fun(I_p_override_values: jax.Array):
      I_p_overrides = torax_experimental.TimeVaryingScalarUpdate(
          time=I_p_new_times,
          value=I_p_override_values,
      )
      runtime_overrides = runtime_params_provider.update_provider_from_mapping(
          {'profile_conditions.I_p': I_p_overrides}
      )
      states, post_processed_outputs, final_i = jit_run_loop.run_loop_jit(
          step_fn=step_fn,
          max_steps=200,
          runtime_params_overrides=runtime_overrides,
      )
      error_states = states.solver_numeric_outputs.solver_error_state
      return post_processed_outputs.Q_fusion[final_i], (error_states, final_i)

    original_values = runtime_params_provider.profile_conditions.I_p.value
    start_I_p = original_values[0]
    end_I_p = original_values[-1]
    mid_I_p = (start_I_p + end_I_p) / 2
    I_p_new_values = jnp.array([start_I_p, mid_I_p, end_I_p])

    grad_vjp, aux = jax.jit(jax.grad(fun, has_aux=True))(I_p_new_values)

    # jax.test_util.check_grads could be used here, but its very slow.
    # Scale the step to the I_p magnitude (~1e7 A): an absolute step of O(1e-6)
    # A changes Q_fusion by ~1e-13 relative, which is dominated by roundoff.
    index = 1
    eps = 1e-6 * I_p_new_values[index]
    eps_vec = (
        jax.nn.one_hot(index, len(I_p_new_values), dtype=jnp.float64) * eps
    )
    Q_plus, aux_plus = fun(I_p_new_values + eps_vec)
    Q_minus, aux_minus = fun(I_p_new_values - eps_vec)
    grad_diff = (Q_plus - Q_minus) / (2 * eps)

    # Gradients through the Newton-Raphson solver use implicit differentiation
    # (jax.lax.custom_root), which is only valid at a converged root, so every
    # step must converge to full tolerance (solver_error_state == 0).
    for error_states, final_i in (aux, aux_plus, aux_minus):
      chex.assert_trees_all_equal(
          error_states[: final_i + 1],
          jnp.zeros_like(error_states[: final_i + 1]),
      )

    chex.assert_trees_all_close(grad_diff, grad_vjp[index], rtol=1e-3)

  def test_progress_bar_update(self):
    torax_config = torax.ToraxConfig.from_dict(
        default_configs.get_default_config_dict()
    )
    step_fn = torax_experimental.make_step_fn(torax_config)

    with mock.patch.object(tqdm, 'tqdm') as mock_tqdm_cls:
      mock_pbar = mock.MagicMock()
      mock_pbar.total = 100
      mock_pbar.n = 0
      mock_tqdm_cls.return_value.__enter__.return_value = mock_pbar

      jit_run_loop.run_loop(
          step_fn=step_fn,
          max_steps=10,
          progress_bar=True,
          log_n_steps=2,
      )

      self.assertGreater(mock_pbar.refresh.call_count, 0)
      self.assertGreater(mock_pbar.n, 0)
      self.assertTrue(mock_pbar.set_description.called)
      last_desc = mock_pbar.set_description.call_args[0][0]
      self.assertStartsWith(last_desc, 'Simulating (t=')
    with jit_run_loop._LOCK:
      self.assertEmpty(jit_run_loop._TQDM_REF)

  def test_timestep_logging(self):
    torax_config = torax.ToraxConfig.from_dict(
        default_configs.get_default_config_dict()
    )
    step_fn = torax_experimental.make_step_fn(torax_config)

    with mock.patch.object(tqdm.tqdm, 'write') as mock_write:
      jit_run_loop.run_loop(
          step_fn=step_fn,
          max_steps=10,
          log_timestep_info=True,
          log_n_steps=5,
      )

      self.assertGreaterEqual(mock_write.call_count, 2)
      for call in mock_write.call_args_list:
        log_line = call[0][0]
        self.assertIn('Simulation time:', log_line)
        self.assertIn('previous dt:', log_line)
        self.assertIn('previous solver iterations:', log_line)

  def test_thread_safety(self):
    torax_config = torax.ToraxConfig.from_dict(
        default_configs.get_default_config_dict()
    )
    step_fn = torax_experimental.make_step_fn(torax_config)

    pbars = []

    def _make_pbar(*args, **kwargs):
      del args, kwargs
      pbar = mock.MagicMock()
      pbar.total = 100
      pbar.__enter__.return_value = pbar
      pbars.append(pbar)
      return pbar

    num_threads = 4
    with (
        mock.patch.object(tqdm, 'tqdm', side_effect=_make_pbar),
        concurrent.futures.ThreadPoolExecutor(
            max_workers=num_threads
        ) as executor,
    ):
      futures = [
          executor.submit(
              jit_run_loop.run_loop,
              step_fn=step_fn,
              max_steps=10,
              progress_bar=True,
              log_n_steps=2,
          )
          for _ in range(num_threads)
      ]
      results = [f.result() for f in futures]

    self.assertLen(results, num_threads)
    for stacked_states, stacked_outputs, _ in results:
      self.assertNotEmpty(stacked_states.t)
      self.assertNotEmpty(stacked_outputs.Q_fusion)

    # Each run must have updated its own progress bar to the same progress.
    self.assertLen(pbars, num_threads)
    for pbar in pbars:
      self.assertGreater(pbar.refresh.call_count, 0)
      self.assertEqual(pbar.n, pbars[0].n)
    self.assertGreater(pbars[0].n, 0)

    with jit_run_loop._LOCK:
      self.assertEmpty(jit_run_loop._TQDM_REF)

  def test_changing_log_n_steps_does_not_recompile(self):
    torax_config = torax.ToraxConfig.from_dict(
        default_configs.get_default_config_dict()
    )
    step_fn = torax_experimental.make_step_fn(torax_config)

    with (
        mock.patch.object(tqdm, 'tqdm') as mock_tqdm_cls,
        mock.patch.object(tqdm.tqdm, 'write'),
    ):
      mock_tqdm_cls.return_value.__enter__.return_value = mock.MagicMock(
          total=100
      )
      jit_run_loop.run_loop(
          step_fn=step_fn,
          max_steps=10,
          progress_bar=True,
          log_timestep_info=True,
          log_n_steps=2,
      )
      num_compiles = jax_utils.get_number_of_compiles(jit_run_loop.run_loop_jit)

      jit_run_loop.run_loop(
          step_fn=step_fn,
          max_steps=10,
          progress_bar=True,
          log_timestep_info=True,
          log_n_steps=3,
      )

    self.assertEqual(
        jax_utils.get_number_of_compiles(jit_run_loop.run_loop_jit),
        num_compiles,
    )


if __name__ == '__main__':
  absltest.main()

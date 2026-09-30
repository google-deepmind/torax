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
import chex
import jax
from jax import numpy as jnp
import torax
from torax._src import jax_utils
from torax._src.orchestration import jit_run_loop
from torax._src.test_utils import default_configs
import torax.experimental as torax_experimental
import tqdm

# pylint: disable=invalid-name

jax.config.update('jax_enable_x64', True)


class JitRunLoopTest(absltest.TestCase):

  def test_gradient(self):

    torax_config = torax.build_torax_config_from_file(
        'examples/iterhybrid_rampup.py'
    )
    step_fn = torax_experimental.make_step_fn(torax_config)
    runtime_params_provider = step_fn.runtime_params_provider

    original_times = runtime_params_provider.profile_conditions.Ip.time
    start_time = original_times[0]
    end_time = original_times[-1]
    mid_time = (start_time + end_time) / 2
    Ip_new_times = jnp.array([start_time, mid_time, end_time])

    @jax.jit
    def fun(
        Ip_override_values: jax.Array,
    ):
      Ip_overrides = torax_experimental.TimeVaryingScalarUpdate(
          time=Ip_new_times,
          value=Ip_override_values,
      )
      runtime_overrides = runtime_params_provider.update_provider_from_mapping(
          {'profile_conditions.Ip': Ip_overrides}
      )
      _, post_processed_outputs, final_i = jit_run_loop.run_loop_jit(
          step_fn=step_fn,
          max_steps=200,
          runtime_params_overrides=runtime_overrides,
      )
      return post_processed_outputs.Q_fusion[final_i]

    original_values = runtime_params_provider.profile_conditions.Ip.value
    start_Ip = original_values[0]
    end_Ip = original_values[-1]
    mid_Ip = (start_Ip + end_Ip) / 2
    Ip_new_values = jnp.array([start_Ip, mid_Ip, end_Ip])

    # Use value-and-grad to avoid compiling twice.
    grad_fn = jax.jit(jax.value_and_grad(fun))
    _, grad_vjp = grad_fn(Ip_new_values)

    # jax.test_util.check_grads could be used here, but its very slow.
    eps = 1e-6
    index = 1
    eps_vec = jax.nn.one_hot(index, len(Ip_new_values), dtype=jnp.float64) * eps
    grad_diff = (
        grad_fn(Ip_new_values + eps_vec)[0]
        - grad_fn(Ip_new_values - eps_vec)[0]
    ) / (2 * eps)

    chex.assert_trees_all_close(grad_diff, grad_vjp[index], atol=5e-9)

  def test_progress_bar_update(self):
    torax_config = torax.ToraxConfig.from_dict(
        default_configs.get_default_config_dict()
    )
    step_fn = torax_experimental.make_step_fn(torax_config)

    mock_pbar = mock.MagicMock(spec=tqdm.tqdm)
    mock_pbar.total = 100
    mock_pbar.n = 0

    with jit_run_loop.LOCK:
      jit_run_loop.COUNTER += 1
      bar_ref = jit_run_loop.COUNTER
      jit_run_loop.TQDM_REF[bar_ref] = mock_pbar

    try:
      jit_run_loop.run_loop(
          step_fn=step_fn,
          max_steps=10,
          progress_bar=True,
          progress_bar_ref=bar_ref,
          log_n_steps=2,
      )
    finally:
      with jit_run_loop.LOCK:
        jit_run_loop.TQDM_REF.pop(bar_ref, None)

    self.assertGreater(mock_pbar.refresh.call_count, 0)
    self.assertGreater(mock_pbar.n, 0)
    self.assertTrue(mock_pbar.set_description.called)
    last_desc = mock_pbar.set_description.call_args[0][0]
    self.assertStartsWith(last_desc, 'Simulating (t=')

  def test_standalone_progress_bar(self):
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
          progress_bar_ref=0,
          log_n_steps=2,
      )

      self.assertGreater(mock_pbar.refresh.call_count, 0)
    with jit_run_loop.LOCK:
      self.assertEmpty(jit_run_loop.TQDM_REF)

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
    for unstacked_states, unstacked_outputs, _ in results:
      self.assertNotEmpty(unstacked_states)
      self.assertNotEmpty(unstacked_outputs)

    # Each run must have updated its own progress bar to the same progress.
    self.assertLen(pbars, num_threads)
    for pbar in pbars:
      self.assertGreater(pbar.refresh.call_count, 0)
      self.assertEqual(pbar.n, pbars[0].n)
    self.assertGreater(pbars[0].n, 0)

    with jit_run_loop.LOCK:
      self.assertEmpty(jit_run_loop.TQDM_REF)

  def test_toggling_logging_does_not_recompile(self):
    torax_config = torax.ToraxConfig.from_dict(
        default_configs.get_default_config_dict()
    )
    step_fn = torax_experimental.make_step_fn(torax_config)

    jit_run_loop.run_loop(step_fn=step_fn, max_steps=10)
    num_compiles = jax_utils.get_number_of_compiles(jit_run_loop.run_loop_jit)

    with mock.patch.object(tqdm, 'tqdm') as mock_tqdm_cls:
      mock_tqdm_cls.return_value.__enter__.return_value = mock.MagicMock(
          total=100
      )
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

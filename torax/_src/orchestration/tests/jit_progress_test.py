# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for jit_progress module and progress bar integration with jit_run_loop."""

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from torax._src import jax_utils
from torax._src import state
from torax._src.config import config_loader
from torax._src.orchestration import jit_progress
from torax._src.orchestration import run_simulation
from torax._src.test_utils import paths


class HostBarTest(parameterized.TestCase):

  def test_init_state(self):
    bar = jit_progress._HostBar(t_initial=0.0, t_final=10.0)
    self.assertEqual(bar._bar.total, 100.0)
    self.assertEqual(bar._bar.n, 0.0)
    self.assertEqual(bar._bar.desc, 'Compiling')
    bar.close(None)

  def test_report_time_advances_and_updates_desc(self):
    bar = jit_progress._HostBar(t_initial=0.0, t_final=10.0)
    bar.report_time(5.0)
    self.assertEqual(bar._bar.n, 50.0)
    self.assertTrue(bar._bar.desc.startswith('Simulating (t=5.00000)'))

    bar.report_time(7.5)
    self.assertEqual(bar._bar.n, 75.0)
    self.assertTrue(bar._bar.desc.startswith('Simulating (t=7.50000)'))
    bar.close(None)

  def test_monotonic_clamping_and_overshoot(self):
    bar = jit_progress._HostBar(t_initial=0.0, t_final=10.0)
    bar.report_time(5.0)
    self.assertEqual(bar._bar.n, 50.0)

    # Out-of-order callback with earlier time should not regress bar.n.
    bar.report_time(3.0)
    self.assertEqual(bar._bar.n, 50.0)

    # Overshooting t_final should clamp at 100%.
    bar.report_time(12.0)
    self.assertEqual(bar._bar.n, 100.0)
    bar.close(None)

  def test_close_with_final_time(self):
    bar = jit_progress._HostBar(t_initial=0.0, t_final=10.0)
    bar.report_time(3.0)
    self.assertEqual(bar._bar.n, 30.0)
    bar.close(8.0)
    self.assertEqual(bar._bar.n, 80.0)

  def test_idempotent_close(self):
    bar = jit_progress._HostBar(t_initial=0.0, t_final=10.0)
    bar.report_time(4.0)
    self.assertEqual(bar._bar.n, 40.0)
    bar.close(6.0)
    self.assertEqual(bar._bar.n, 60.0)
    # Subsequent calls to close or report_time should be safe no-ops
    bar.close(8.0)
    self.assertEqual(bar._bar.n, 60.0)
    bar.report_time(9.0)
    self.assertEqual(bar._bar.n, 60.0)

  def test_degenerate_interval(self):
    # If t_final <= t_initial, division by zero should be guarded against.
    bar = jit_progress._HostBar(t_initial=5.0, t_final=5.0)
    bar.report_time(5.0)
    self.assertEqual(bar._bar.n, 0.0)
    bar.close(None)


class EmitProgressTest(absltest.TestCase):

  def test_emit_progress_under_jit(self):
    with jit_progress.JitProgressBar(t_initial=0.0, t_final=10.0) as pbar:
      bar_id = pbar.id_array
      report_interval = jnp.asarray(pbar.report_interval)

      @jax.jit
      def step(t, previous_t):
        jit_progress.emit_progress(
            bar_id=bar_id,
            t=t,
            previous_t=previous_t,
            t_initial=jnp.asarray(0.0),
            report_interval=report_interval,
        )

      assert pbar._host_bar is not None
      # Report interval is 10.0 / 100.0 = 0.1
      # Step from 0.0 to 0.05: prev_bucket = 0, curr_bucket = 0 -> no emit
      step(jnp.asarray(0.05), jnp.asarray(0.0))
      jax.effects_barrier()
      self.assertEqual(pbar._host_bar._bar.n, 0.0)

      # Step from 0.05 to 0.15: prev_bucket = 0, curr_bucket = 1 -> emit fires!
      step(jnp.asarray(0.15), jnp.asarray(0.05))
      jax.effects_barrier()
      self.assertAlmostEqual(pbar._host_bar._bar.n, 1.5)


class JitProgressBarTest(absltest.TestCase):

  def test_context_manager_lifecycle(self):
    with jit_progress.JitProgressBar(t_initial=0.0, t_final=1.0) as pbar:
      bar_id = int(pbar._id)
      self.assertIn(bar_id, jit_progress._registry)
      self.assertIsNotNone(pbar._host_bar)
    self.assertNotIn(bar_id, jit_progress._registry)
    self.assertIsNone(pbar._host_bar)

  def test_exception_cleanup(self):
    bar_id = None
    try:
      with jit_progress.JitProgressBar(t_initial=0.0, t_final=1.0) as pbar:
        bar_id = int(pbar._id)
        self.assertIn(bar_id, jit_progress._registry)
        raise ValueError('Test error inside context')
    except ValueError:
      pass
    self.assertNotIn(bar_id, jit_progress._registry)

  def test_params_property(self):
    with jit_progress.JitProgressBar(t_initial=0.0, t_final=10.0) as pbar:
      params = pbar.params
      self.assertIsInstance(params, jit_progress.JitProgressParams)
      self.assertEqual(int(params.bar_id), pbar._id)
      self.assertAlmostEqual(float(params.report_interval), 0.1)

  def test_custom_report_fraction(self):
    pbar = jit_progress.JitProgressBar(
        t_initial=0.0, t_final=10.0, report_fraction=0.05
    )
    # 10.0 * 0.05 = 0.5
    self.assertAlmostEqual(pbar.report_interval, 0.5)

  def test_compilation_stability_across_runs(self):
    """Verifies that varying bar_id in params does not cause recompilation."""

    @jax.jit
    def dummy_loop(progress_params):
      t = jnp.asarray(0.5)
      prev_t = jnp.asarray(0.0)
      jit_progress.emit_progress(
          bar_id=progress_params.bar_id,
          t=t,
          previous_t=prev_t,
          t_initial=jnp.asarray(0.0),
          report_interval=progress_params.report_interval,
      )

    # First run
    with jit_progress.JitProgressBar(0.0, 1.0) as pbar1:
      dummy_loop(pbar1.params)
      jax.effects_barrier()

    # Second run with different bar_id
    with jit_progress.JitProgressBar(0.0, 1.0) as pbar2:
      self.assertNotEqual(pbar1._id, pbar2._id)
      dummy_loop(pbar2.params)
      jax.effects_barrier()

    # Check JAX cache size
    self.assertEqual(jax_utils.get_number_of_compiles(dummy_loop), 1)


class JitRunLoopProgressTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    test_data_dir = paths.test_data_dir()
    config_path = os.path.join(test_data_dir, 'test_implicit.py')
    self.torax_config = config_loader.build_torax_config_from_file(config_path)

  def test_run_simulation_jitted_with_progress_bar(self):
    xr_data_tree, state_history = run_simulation.run_simulation(
        self.torax_config,
        max_steps=5,
        progress_bar=True,
        _use_jitted_run_loop=True,
    )
    self.assertEqual(
        state_history.sim_error, state.SimError.DID_NOT_REACH_T_FINAL
    )
    self.assertIsNotNone(xr_data_tree)
    self.assertLen(state_history.times, 6)

  def test_run_simulation_jitted_without_progress_bar(self):
    xr_data_tree, state_history = run_simulation.run_simulation(
        self.torax_config,
        max_steps=5,
        progress_bar=False,
        _use_jitted_run_loop=True,
    )
    self.assertEqual(
        state_history.sim_error, state.SimError.DID_NOT_REACH_T_FINAL
    )
    self.assertIsNotNone(xr_data_tree)
    self.assertLen(state_history.times, 6)

  def test_early_termination_finalizes_honestly(self):
    with mock.patch.object(
        jit_progress.JitProgressBar, 'finalize', wraps=None
    ) as mock_finalize:
      _, state_history = run_simulation.run_simulation(
          self.torax_config,
          max_steps=1,
          progress_bar=True,
          _use_jitted_run_loop=True,
      )
      self.assertEqual(
          state_history.sim_error, state.SimError.DID_NOT_REACH_T_FINAL
      )
      # finalize should have been called with the actual reached time
      self.assertEqual(mock_finalize.call_count, 1)


if __name__ == '__main__':
  absltest.main()

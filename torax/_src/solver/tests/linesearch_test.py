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

"""Tests for linesearch module."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax.numpy as jnp
from torax._src.solver import linesearch


class BacktrackingLinesearchTest(parameterized.TestCase):

  def test_linesearch_success(self):
    def residual_fn(x):
      return x - 2.0

    x_init = jnp.array(0.0)
    direction = jnp.array(2.0)  # Newton step to root

    def accept_fn(step_size, trial_norm):
      del step_size
      return trial_norm <= 1.0  # Initial is 2.0, so decreasing is good

    def norm_fn(res):
      return jnp.abs(res)

    final = linesearch.backtracking_linesearch(
        residual_fn=residual_fn,
        x_init=x_init,
        direction=direction,
        accept_fn=accept_fn,
        norm_fn=norm_fn,
        initial_residual=x_init - 2.0,
        initial_residual_norm=jnp.array(2.0),
        delta_reduction_factor=0.5,
        max_steps=10,
    )

    self.assertTrue(bool(final.step_found))
    self.assertLessEqual(int(final.iteration), 10)
    chex.assert_trees_all_close(final.x, jnp.array(2.0))

  def test_linesearch_backtracking(self):
    # A function that increases residual if step is too large
    def residual_fn(x):
      # If x > 1.0, return large residual, else return x - 2.0
      return jnp.where(x > 1.0, 10.0, x - 2.0)

    x_init = jnp.array(0.0)
    direction = jnp.array(2.0)

    def accept_fn(step_size, trial_norm):
      del step_size
      return trial_norm <= 1.5  # Initial norm is 2.0. 1.0 is good.

    def norm_fn(res):
      return jnp.abs(res)

    final = linesearch.backtracking_linesearch(
        residual_fn=residual_fn,
        x_init=x_init,
        direction=direction,
        accept_fn=accept_fn,
        norm_fn=norm_fn,
        initial_residual=x_init - 2.0,
        initial_residual_norm=jnp.array(2.0),
        delta_reduction_factor=0.5,
        max_steps=10,
    )

    self.assertTrue(bool(final.step_found))
    self.assertGreater(int(final.iteration), 1)  # Must have backtracked
    chex.assert_trees_all_close(final.x, jnp.array(1.0))

  def test_linesearch_pytree(self):
    def residual_fn(x):
      return {'a': x['a'] - 2.0, 'b': x['b'] - 3.0}

    x_init = {'a': jnp.array(0.0), 'b': jnp.array(0.0)}
    direction = {'a': jnp.array(2.0), 'b': jnp.array(3.0)}

    def accept_fn(step_size, trial_norm):
      del step_size
      return trial_norm <= 1.0

    def norm_fn(res):
      return jnp.sqrt(res['a'] ** 2 + res['b'] ** 2)

    init_res = residual_fn(x_init)
    init_norm = norm_fn(init_res)

    final = linesearch.backtracking_linesearch(
        residual_fn=residual_fn,
        x_init=x_init,
        direction=direction,
        accept_fn=accept_fn,
        norm_fn=norm_fn,
        initial_residual=init_res,
        initial_residual_norm=init_norm,
        delta_reduction_factor=0.5,
        max_steps=10,
    )

    self.assertTrue(bool(final.step_found))
    chex.assert_trees_all_close(
        final.x, {'a': jnp.array(2.0), 'b': jnp.array(3.0)}
    )


if __name__ == '__main__':
  absltest.main()

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

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax import numpy as jnp
import jax.test_util as jtu
from torax._src import jax_utils


class WhileLoopBoundedTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self._terminating_step = 4
    self._max_steps = 10
    self._init_value = 0.5
    self.init_state = (0, self._init_value)

    self._cond_fun = lambda state: state[0] < self._terminating_step
    self._body_fun = lambda state: (state[0] + 1, jnp.sin(state[1]))

    def f_while(x, max_steps=self._max_steps, implementation='while_loop'):
      init_state = (0, x)
      return jax_utils.while_loop_bounded(
          self._cond_fun,
          self._body_fun,
          init_state,
          max_steps=max_steps,
          implementation=implementation,
      )[0][1]

    def f_explicit(x, n_times=self._terminating_step):
      result = x
      for _ in range(n_times):
        result = jnp.sin(result)
      return result

    self._f_while = f_while
    self._f_explicit = f_explicit

  @parameterized.parameters(['scan', 'while_loop'])
  def test_forward_agrees_with_while_loop(self, implementation):
    output_state, _, _ = jax_utils.while_loop_bounded(
        self._cond_fun,
        self._body_fun,
        self.init_state,
        self._max_steps,
        implementation=implementation,
    )
    chex.assert_trees_all_close(
        output_state,
        jax.lax.while_loop(self._cond_fun, self._body_fun, self.init_state),
    )

  @parameterized.parameters(['scan', 'while_loop'])
  def test_forward_agrees_with_explicit(self, implementation):
    chex.assert_trees_all_close(
        self._f_while(self._init_value, implementation=implementation),
        self._f_explicit(self._init_value),
    )

  @parameterized.parameters(['scan', 'while_loop'])
  def test_grad_agrees_with_explicit(self, implementation):
    chex.assert_trees_all_close(
        jax.grad(self._f_while)(
            self._init_value, implementation=implementation
        ),
        jax.grad(self._f_explicit)(self._init_value),
    )

  @parameterized.parameters(['scan', 'while_loop'])
  def test_max_steps_is_respected_if_loop_would_continue(self, implementation):
    final_state, num_steps, _ = jax_utils.while_loop_bounded(
        self._cond_fun,
        self._body_fun,
        self.init_state,
        max_steps=2,
        implementation=implementation,
    )
    final_i, final_value = final_state
    self.assertEqual(final_i, 2)
    self.assertEqual(num_steps, 2)
    chex.assert_trees_all_close(
        final_value, self._f_explicit(self._init_value, n_times=2)
    )

  @parameterized.parameters(['scan', 'while_loop'])
  def test_grad_max_steps_is_respected_if_loop_would_continue(
      self, implementation
  ):
    chex.assert_trees_all_close(
        jax.grad(self._f_while)(
            self._init_value, max_steps=2, implementation=implementation
        ),
        jax.grad(self._f_explicit)(self._init_value, n_times=2),
    )

  @parameterized.parameters(['scan', 'while_loop'])
  def test_output_history(self, implementation):
    _, num_steps, output_history = jax_utils.while_loop_bounded(
        self._cond_fun,
        self._body_fun,
        self.init_state,
        max_steps=self._max_steps,
        implementation=implementation,
    )
    history_i, history_values = output_history
    # output_history should be (max_steps, ...) shaped.
    self.assertEqual(history_i.shape, (self._max_steps,))
    self.assertEqual(history_values.shape, (self._max_steps,))
    self.assertEqual(num_steps, self._terminating_step)

    # Check each executed step.
    for step in range(self._terminating_step):
      chex.assert_trees_all_close(
          history_values[step],
          self._f_explicit(self._init_value, n_times=step + 1),
      )
    # Steps after termination should contain NaNs for floats and 0 for ints.
    for step in range(self._terminating_step, self._max_steps):
      self.assertTrue(jnp.isnan(history_values[step]))
      self.assertEqual(history_i[step], 0)

  @parameterized.product(implementation=['scan', 'while_loop'])
  def test_forward_mode_jvp(self, implementation):
    """Test that forward mode JVP matches that of jax.lax.while_loop."""

    terminating_step = 6
    max_steps = 10
    cond_fun = lambda state: state[0] < terminating_step
    body_fun = lambda state: (
        state[0] + 1,
        {
            'a': state[1]['a'][-1] * jnp.sin(state[1]['b']),
            'b': state[1]['b'][0] * jnp.cos(state[1]['a']),
        },
    )

    def f(x):
      init_state = (0, x)
      return jax_utils.while_loop_bounded(
          cond_fun,
          body_fun,
          init_state,
          max_steps=max_steps,
          implementation=implementation,
      )[0][1]

    def f_ref(x):
      init_state = (0, x)
      return jax.lax.while_loop(
          cond_fun,
          body_fun,
          init_state,
      )[1]

    x = {
        'a': jnp.array([0.2, 0.3, -0.3, 0.0]),
        'b': jnp.array([1.0, 2.0, 3.0, 4.0]),
    }

    primals, tangents = jax.jvp(f, (x,), (x,))
    primals_ref, tangents_ref = jax.jvp(f_ref, (x,), (x,))

    chex.assert_trees_all_close(primals_ref, primals)
    chex.assert_trees_all_close(tangents_ref, tangents)

  @parameterized.parameters(['scan', 'while_loop'])
  def test_closure_grad(self, implementation):
    """Test that gradients can be taken through a closure."""

    @jax.jit
    def f_loss(x):
      terminating_step = 6
      cond_fun = lambda state: state[0] < terminating_step
      body_fun = lambda state: (
          state[0] + 1,
          x['x'] * jnp.sin(x['x'] * state[1]),
      )
      init_state = (0, 0.5)
      out = jax_utils.while_loop_bounded(
          cond_fun,
          body_fun,
          init_state,
          max_steps=10,
          implementation=implementation,
      )[0][1]
      return jnp.sum(out)

    x = {'x': 0.2}
    jtu.check_grads(f_loss, (x,), modes=('rev', 'fwd'), order=1)

  @parameterized.parameters(['scan', 'while_loop'])
  def test_vmap(self, implementation):
    """Test that while_loop_bounded can be transformed with jax.vmap."""

    def f(x):
      cond_fun = lambda state: state[0] < 4
      body_fun = lambda state: (state[0] + 1, jnp.sin(state[1]))
      init_state = (0, x)
      return jax_utils.while_loop_bounded(
          cond_fun,
          body_fun,
          init_state,
          max_steps=10,
          implementation=implementation,
      )[0][1]

    xs = jnp.array([0.1, 0.5, 1.0, 2.0])
    vmapped_out = jax.vmap(f)(xs)
    expected_out = jnp.array([self._f_explicit(x) for x in xs])
    chex.assert_trees_all_close(vmapped_out, expected_out)

  @parameterized.parameters(['scan', 'while_loop'])
  def test_vmap_with_closure(self, implementation):
    """Test that while_loop_bounded with closures works under jax.vmap."""

    def f(a, x):
      cond_fun = lambda state: state[0] < 4
      body_fun = lambda state: (state[0] + 1, a * state[1])
      init_state = (0, x)
      return jax_utils.while_loop_bounded(
          cond_fun,
          body_fun,
          init_state,
          max_steps=10,
          implementation=implementation,
      )[0][1]

    a = jnp.array([2.0, 3.0])
    x = jnp.array([1.0, 2.0])
    vmapped_out = jax.vmap(f)(a, x)
    expected_out = jnp.array([1.0 * (2.0**4), 2.0 * (3.0**4)])
    chex.assert_trees_all_close(vmapped_out, expected_out)


if __name__ == '__main__':
  absltest.main()

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
import chex
import jax
from jax import numpy as jnp
import torax
from torax._src.orchestration import jit_run_loop
import torax.experimental as torax_experimental

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


if __name__ == '__main__':
  absltest.main()

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
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.test_util as jtu
from torax._src.config import config_loader
from torax._src.orchestration import run_simulation
from torax._src.torax_pydantic import interpolated_param_1d


class StepFunctionTest(parameterized.TestCase):

  @parameterized.parameters([
      'basic_config',
      'iterhybrid_predictor_corrector',
  ])
  def test_step_function_grad(self, config_name_no_py):
    example_config_paths = config_loader.example_config_paths()
    example_config_path = example_config_paths[config_name_no_py]
    cfg = config_loader.build_torax_config_from_file(example_config_path)
    (
        sim_state,
        post_processed_outputs,
        step_fn,
    ) = run_simulation.prepare_simulation(cfg)
    params_provider = step_fn.runtime_params_provider
    input_value = params_provider.profile_conditions.Ip.value

    @jax.jit
    def f(override_value):
      ip_update = interpolated_param_1d.TimeVaryingScalarUpdate(
          value=override_value
      )
      runtime_params_overrides = params_provider.update_provider(
          lambda x: (x.profile_conditions.Ip,),
          (ip_update,),
      )
      _, new_post_processed_outputs = step_fn(
          sim_state,
          post_processed_outputs,
          runtime_params_overrides=runtime_params_overrides,
      )
      return new_post_processed_outputs.Q_fusion

    jtu.check_grads(f, (input_value,), order=1, modes=('rev',))


if __name__ == '__main__':
  absltest.main()

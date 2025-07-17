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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from torax._src.config import runtime_params_slice
from torax._src.solver import linear_theta_method
from torax._src.solver import nonlinear_theta_method
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config


class PydanticModelTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='linear',
          solver_type='linear',
          expected_type=linear_theta_method.LinearThetaMethod,
      ),
      dict(
          testcase_name='newton_raphson',
          solver_type='newton_raphson',
          expected_type=nonlinear_theta_method.NewtonRaphsonThetaMethod,
      ),
      dict(
          testcase_name='optimizer',
          solver_type='optimizer',
          expected_type=nonlinear_theta_method.OptimizerThetaMethod,
      ),
  )
  def test_build_solver_from_config(self, solver_type, expected_type):
    """Builds a solver from the config."""
    config = default_configs.get_default_config_dict()
    config['solver'] = {
        'solver_type': solver_type,
        'theta_implicit': 0.5,
    }
    torax_config = model_config.ToraxConfig.from_dict(config)

    solver = torax_config.solver.build_solver(
        static_runtime_params_slice=mock.create_autospec(
            runtime_params_slice.StaticRuntimeParamsSlice,
            instance=True,
            evolve_ion_heat=True,
            evolve_electron_heat=True,
            evolve_current=True,
            evolve_density=True,
        ),
        physics_models=torax_config.build_physics_models(),
    )
    self.assertIsInstance(solver, expected_type)
    self.assertEqual(torax_config.solver.theta_implicit, 0.5)


if __name__ == '__main__':
  absltest.main()

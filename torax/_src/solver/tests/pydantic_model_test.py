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
from torax._src import jax_utils
from torax._src.solver import linear_theta_method
from torax._src.solver import nonlinear_theta_method
from torax._src.solver import stationary_theta_method
from torax._src.solver import pydantic_model as solver_pydantic_model
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
      dict(
          testcase_name='stationary',
          solver_type='stationary',
          expected_type=stationary_theta_method.StationaryThetaMethod,
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
        physics_models=torax_config.build_physics_models(),
    )
    self.assertIsInstance(solver, expected_type)
    self.assertEqual(torax_config.solver.theta_implicit, 0.5)

  @parameterized.parameters('linear', 'newton_raphson', 'optimizer','stationary')
  def test_solver_under_jit(self, solver_type):
    config = default_configs.get_default_config_dict()
    config['solver'] = {
        'solver_type': solver_type,
        'D_pereverzev': 0.5,
    }
    torax_config = model_config.ToraxConfig.from_dict(config)

    solver = torax_config.solver

    @jax.jit
    def f(solver: solver_pydantic_model.SolverConfig):
      return solver.build_runtime_params

    with self.subTest('first_jit_compiles_and_returns_expected_value'):
      output = f(solver)
      self.assertIsInstance(
          output, solver_pydantic_model.runtime_params.RuntimeParams
      )
      self.assertEqual(output.D_pereverzev, 0.5)
      self.assertEqual(jax_utils.get_number_of_compiles(f), 1)

    with self.subTest('second_jit_updates_value_without_recompile'):
      solver._update_fields({'D_pereverzev': 0.6})
      output = f(solver)
      self.assertEqual(output.D_pereverzev, 0.6)
      self.assertEqual(jax_utils.get_number_of_compiles(f), 1)


if __name__ == '__main__':
  absltest.main()

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
from torax.config import runtime_params_slice
from torax.pedestal_model import pydantic_model as pedestal_pydantic_model
from torax.sources import pydantic_model as source_pydantic_model
from torax.sources import source_models as source_models_lib
from torax.stepper import linear_theta_method
from torax.stepper import nonlinear_theta_method
from torax.stepper import pydantic_model as stepper_pydantic_model
from torax.transport_model import pydantic_model as transport_pydantic_model


class PydanticModelTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='linear',
          solver_model=stepper_pydantic_model.LinearThetaMethod,
          expected_type=linear_theta_method.LinearThetaMethod,
      ),
      dict(
          testcase_name='newton_raphson',
          solver_model=stepper_pydantic_model.NewtonRaphsonThetaMethod,
          expected_type=nonlinear_theta_method.NewtonRaphsonThetaMethod,
      ),
      dict(
          testcase_name='optimizer',
          solver_model=stepper_pydantic_model.OptimizerThetaMethod,
          expected_type=nonlinear_theta_method.OptimizerThetaMethod,
      ),
  )
  def test_build_solver_from_config(self, solver_model, expected_type):
    """Builds a solver from the config."""

    solver_pydantic = solver_model(theta_implicit=0.5)
    transport = transport_pydantic_model.ConstantTransportModel()
    transport_model = transport.build_transport_model()
    pedestal = pedestal_pydantic_model.NoPedestal()
    pedestal_model = pedestal.build_pedestal_model()
    sources = source_pydantic_model.Sources.from_dict({})
    source_models = source_models_lib.SourceModels(
        sources=sources
    )
    solver = solver_pydantic.build_solver(
        static_runtime_params_slice=mock.create_autospec(
            runtime_params_slice.StaticRuntimeParamsSlice,
            instance=True,
            evolve_ion_heat=True,
            evolve_electron_heat=True,
            evolve_current=True,
            evolve_density=True,
        ),
        transport_model=transport_model,
        source_models=source_models,
        pedestal_model=pedestal_model,
    )
    self.assertIsInstance(solver, expected_type)
    self.assertEqual(solver_pydantic.theta_implicit, 0.5)


if __name__ == '__main__':
  absltest.main()

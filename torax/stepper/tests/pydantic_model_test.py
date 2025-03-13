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
from torax.config import build_sim
from torax.pedestal_model import pydantic_model as pedestal_pydantic_model
from torax.sources import pydantic_model as source_pydantic_model
from torax.sources import source_models as source_models_lib
from torax.stepper import linear_theta_method
from torax.stepper import nonlinear_theta_method
from torax.stepper import pydantic_model as stepper_pydantic_model


class PydanticModelTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='linear',
          stepper_type='linear',
          expected_type=linear_theta_method.LinearThetaMethod,
      ),
      dict(
          testcase_name='newton_raphson',
          stepper_type='newton_raphson',
          expected_type=nonlinear_theta_method.NewtonRaphsonThetaMethod,
      ),
      dict(
          testcase_name='optimizer',
          stepper_type='optimizer',
          expected_type=nonlinear_theta_method.OptimizerThetaMethod,
      ),
  )
  def test_build_stepper_from_config(self, stepper_type, expected_type):
    """Builds a stepper from the config."""
    stepper = stepper_pydantic_model.Stepper.from_dict({
        'stepper_type': stepper_type,
        'theta_imp': 0.5,
    })
    transport_model_builder = (
        build_sim.build_transport_model_builder_from_config('constant')
    )
    transport_model = transport_model_builder()
    pedestal = pedestal_pydantic_model.Pedestal()
    pedestal_model = pedestal.build_pedestal_model()
    sources = source_pydantic_model.Sources.from_dict({})
    source_models = source_models_lib.SourceModels(
        sources=sources.source_model_config
    )
    stepper_model = stepper.build_stepper_model(
        transport_model=transport_model,
        source_models=source_models,
        pedestal_model=pedestal_model,
    )
    self.assertIsInstance(stepper_model, expected_type)
    self.assertEqual(stepper.stepper_config.theta_imp, 0.5)


if __name__ == '__main__':
  absltest.main()

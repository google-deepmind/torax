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

"""Pydantic config for Stepper."""

from typing import Any, Union
import pydantic
from torax.stepper import linear_theta_method
from torax.stepper import nonlinear_theta_method
from torax.stepper import runtime_params
from torax.torax_pydantic import torax_pydantic
from typing_extensions import Annotated


StepperConfig = Union[
    Annotated[linear_theta_method.Linear, pydantic.Tag('linear')],
    Annotated[
        nonlinear_theta_method.NewtonRaphson, pydantic.Tag('newton_raphson')
    ],
    Annotated[nonlinear_theta_method.Optimizer, pydantic.Tag('optimizer')],
]


def get_discriminator_value(model: dict[str, Any]) -> str:
  return model.get('stepper_type', 'linear')


class Stepper(torax_pydantic.BaseModelMutable):
  """Config for a stepper."""

  stepper_config: Annotated[
      StepperConfig, pydantic.Discriminator(get_discriminator_value)
  ]

  @pydantic.model_validator(mode='before')
  @classmethod
  def _conform_data(cls, data: dict[str, Any]) -> dict[str, Any]:
    # If we are running with the standard class constructor we don't need to do
    # any custom validation.
    if 'stepper_config' in data:
      return data

    return {'stepper_config': data}

  def build_dynamic_params(self) -> runtime_params.DynamicRuntimeParams:
    return self.stepper_config.build_dynamic_params()

  def build_static_params(self) -> runtime_params.StaticRuntimeParams:
    return runtime_params.StaticRuntimeParams(
        theta_imp=self.stepper_config.theta_imp,
        convection_dirichlet_mode=self.stepper_config.convection_dirichlet_mode,
        convection_neumann_mode=self.stepper_config.convection_neumann_mode,
        use_pereverzev=self.stepper_config.use_pereverzev,
        predictor_corrector=self.stepper_config.predictor_corrector,
    )

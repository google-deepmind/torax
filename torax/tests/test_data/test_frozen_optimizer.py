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

"""Config for test_implicit. Basic test of implicit solver."""

import functools

from torax import config as config_lib
from torax import geometry
from torax import sim as sim_lib
from torax.sources import default_sources
from torax.sources import runtime_params as source_runtime_params
from torax.sources import source_models as source_models_lib
from torax.stepper import nonlinear_theta_method
from torax.stepper import runtime_params as stepper_runtime_params
from torax.tests.test_lib import sim_test_case
from torax.transport_model import constant as constant_transport_model


def get_config() -> config_lib.Config:
  return config_lib.Config(
      profile_conditions=config_lib.ProfileConditions(
          set_pedestal=False,
      ),
      numerics=config_lib.Numerics(
          t_final=1,
      ),
  )


def get_geometry(config: config_lib.Config) -> geometry.Geometry:
  return geometry.build_circular_geometry(config)


def get_transport_model() -> constant_transport_model.ConstantTransportModel:
  return constant_transport_model.ConstantTransportModel()


def get_sources() -> source_models_lib.SourceModels:
  """Returns the source models used in the simulation."""
  source_models = default_sources.get_default_sources()
  # multiplier for ion-electron heat exchange term for sensitivity
  source_models.qei_source.runtime_params.Qei_mult = 0.0
  # remove bootstrap current
  source_models.j_bootstrap.runtime_params.bootstrap_mult = 0.0
  source_models.sources['fusion_heat_source'].runtime_params.mode = (
      source_runtime_params.Mode.ZERO
  )
  source_models.sources['ohmic_heat_source'].runtime_params.mode = (
      source_runtime_params.Mode.ZERO
  )
  return source_models


def get_stepper_builder() -> nonlinear_theta_method.OptimizerThetaMethodBuilder:
  """Returns a builder for the stepper that includes its runtime params."""
  builder = nonlinear_theta_method.OptimizerThetaMethodBuilder(
      runtime_params=stepper_runtime_params.RuntimeParams(
          predictor_corrector=False,
          theta_imp=1.0,
      )
  )
  return builder


def get_sim() -> sim_lib.Sim:
  """Returns a simulation object."""
  # This approach is currently lightweight because so many objects require
  # config for construction, but over time we expect to transition to most
  # config taking place via constructor args in this function.
  config = get_config()
  geo = get_geometry(config)
  transport_model = get_transport_model()
  stepper_builder = get_stepper_builder()
  stepper_builder.builder = functools.partial(
      sim_test_case.make_frozen_optimizer_stepper,
      config=config,
      transport_params=transport_model.runtime_params,
  )
  return sim_lib.build_sim_from_config(
      config=config,
      geo=geo,
      stepper_builder=stepper_builder,
      transport_model=transport_model,
      source_models=get_sources(),
  )

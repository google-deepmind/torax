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

"""Config for testing fixed timestep."""

from torax import config as config_lib
from torax import geometry
from torax import sim as sim_lib
from torax.sources import default_sources
from torax.sources import runtime_params as source_runtime_params
from torax.sources import source_models as source_models_lib
from torax.stepper import linear_theta_method
from torax.time_step_calculator import fixed_time_step_calculator
from torax.transport_model import qlknn_wrapper


def get_config() -> config_lib.Config:
  # This config based approach is deprecated.
  # Over time more will be built with pure Python constructors in `get_sim`.
  return config_lib.Config(
      numerics=config_lib.Numerics(
          t_final=2,
          use_fixed_dt=True,
          fixed_dt=2e-2,
      ),
      # Do not use the fusion heat source.
      solver=config_lib.SolverConfig(
          predictor_corrector=False,
          use_pereverzev=True,
      ),
  )


def get_geometry(config: config_lib.Config) -> geometry.Geometry:
  return geometry.build_circular_geometry(config)


def get_transport_model() -> qlknn_wrapper.QLKNNTransportModel:
  return qlknn_wrapper.QLKNNTransportModel()


def get_sources() -> source_models_lib.SourceModels:
  """Returns the source models used in the simulation."""
  source_models = default_sources.get_default_sources()
  # remove bootstrap current
  source_models.j_bootstrap.runtime_params.bootstrap_mult = 0.0
  source_models.sources['fusion_heat_source'].runtime_params.mode = (
      source_runtime_params.Mode.ZERO
  )
  source_models.sources['ohmic_heat_source'].runtime_params.mode = (
      source_runtime_params.Mode.ZERO
  )
  return source_models


def get_sim() -> sim_lib.Sim:
  # This approach is currently lightweight because so many objects require
  # config for construction, but over time we expect to transition to most
  # config taking place via constructor args in this function.
  sim_config = get_config()
  geo = get_geometry(sim_config)
  if sim_config.numerics.use_fixed_dt:
    time_step_calculator = fixed_time_step_calculator.FixedTimeStepCalculator()
  else:
    time_step_calculator = None
  return sim_lib.build_sim_from_config(
      config=sim_config,
      geo=geo,
      stepper_builder=linear_theta_method.LinearThetaMethod,
      source_models=get_sources(),
      transport_model=get_transport_model(),
      time_step_calculator=time_step_calculator,
  )

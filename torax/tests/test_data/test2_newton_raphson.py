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

"""Config for test2_newton_raphson."""

import functools

from torax import config as config_lib
from torax import config_slice
from torax import geometry
from torax import sim as sim_lib
from torax import stepper as stepper_lib
from torax import transport_model as transport_model_lib
from torax.sources import source_config
from torax.sources import source_profiles
from torax.stepper import nonlinear_theta_method


def make_frozen_newton_raphson_stepper(
    transport_model: transport_model_lib.TransportModel,
    sources: source_profiles.Sources,
    config: config_lib.Config,
) -> stepper_lib.Stepper:
  """Makes a Newton Raphson stepper with frozen coefficients.

  Under these conditions, we can test that the nonlinear stepper behaves the
  same as
  the linear solver.

  Args:
    transport_model: Transport model.
    sources: TORAX sources/sinks used to compute profile terms in the state
      evolution equations.
    config: General TORAX config.

  Returns:
    Stepper: the stepper.
  """
  # Get the dynamic config for the start of the simulation.
  dynamic_config_slice = config_slice.build_dynamic_config_slice(config)
  callback_builder = functools.partial(
      sim_lib.FrozenCoeffsCallback,
      dynamic_config_slice=dynamic_config_slice,
  )
  functools.partial(sim_lib.FrozenCoeffsCallback, config=config)
  return nonlinear_theta_method.NewtonRaphsonThetaMethod(
      transport_model,
      sources=sources,
      callback_class=callback_builder,  # pytype: disable=wrong-arg-types
  )


def get_config() -> config_lib.Config:
  return config_lib.Config(
      set_pedestal=False,
      Qei_mult=0,
      t_final=1,
      solver=config_lib.SolverConfig(
          predictor_corrector=False,
          theta_imp=1.0,
      ),
      bootstrap_mult=0,  # remove bootstrap current
      transport=config_lib.TransportConfig(
          transport_model="constant",
      ),
      sources=dict(
          fusion_heat_source=source_config.SourceConfig(
              source_type=source_config.SourceType.ZERO,
          ),
          ohmic_heat_source=source_config.SourceConfig(
              source_type=source_config.SourceType.ZERO,
          ),
      ),
  )


def get_geometry(config: config_lib.Config) -> geometry.Geometry:
  return geometry.build_circular_geometry(config)


def get_sim() -> sim_lib.Sim:
  # This approach is currently lightweight because so many objects require
  # config for construction, but over time we expect to transition to most
  # config taking place via constructor args in this function.
  config = get_config()
  geo = get_geometry(config)
  return sim_lib.build_sim_from_config(
      config,
      geo,
      functools.partial(
          make_frozen_newton_raphson_stepper,
          config=config,
      ),
  )

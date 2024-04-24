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

"""Tests bootstrap current with current, heat, and particle transport.

Constant transport coefficient model
"""

from torax import config as config_lib
from torax import geometry
from torax import sim as sim_lib
from torax.sources import source_config
from torax.stepper import linear_theta_method
from torax.transport_model import constant as constant_transport_model


def get_config() -> config_lib.Config:
  return config_lib.Config(
      profile_conditions=config_lib.ProfileConditions(
          set_pedestal=False,
          nbar=0.85,  # initial density (in Greenwald fraction units)
      ),
      numerics=config_lib.Numerics(
          Qei_mult=1,
          ion_heat_eq=True,
          el_heat_eq=True,
          dens_eq=True,
          current_eq=True,
          resistivity_mult=100,  # to shorten current diffusion time
          bootstrap_mult=1,  # remove bootstrap current
          t_final=1,
      ),
      # set flat Ohmic current to provide larger range of current evolution for
      # test
      nu=0,
      S_pellet_tot=0.0,
      S_puff_tot=0.0,
      S_nbi_tot=0.0,
      solver=config_lib.SolverConfig(
          predictor_corrector=False,
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


def get_transport_model() -> constant_transport_model.ConstantTransportModel:
  return constant_transport_model.ConstantTransportModel(
      runtime_params=constant_transport_model.RuntimeParams(
          # diffusion coefficient in electron density equation in m^2/s
          De_const=0.5,
          # convection coefficient in electron density equation in m^2/s
          Ve_const=-0.2,
      ),
  )


def get_sim() -> sim_lib.Sim:
  # This approach is currently lightweight because so many objects require
  # config for construction, but over time we expect to transition to most
  # config taking place via constructor args in this function.
  config = get_config()
  geo = get_geometry(config)
  return sim_lib.build_sim_from_config(
      config=config,
      geo=geo,
      stepper_builder=linear_theta_method.LinearThetaMethod,
      transport_model=get_transport_model(),
  )

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

"""Tests semi-implicit convection as carried out with FiPy.

Semi-implicit convection can lead to numerical instability at boundary
condition. No pedestal, implicit + pereverzev-corrigan, Ti+Te,
Pei standard dens, chi from CGM.
"""

from torax import config as config_lib
from torax import geometry
from torax import sim as sim_lib
from torax.sources import source_config
from torax.stepper import linear_theta_method


def get_config() -> config_lib.Config:
  return config_lib.Config(
      # This is test6 but modified to not use the pedestal feature, to exercise
      # the convection term at the boundary. This causes FiPy to explode.
      set_pedestal=False,
      t_final=1,
      transport=config_lib.TransportConfig(
          transport_model='CGM',
      ),
      solver=config_lib.SolverConfig(
          predictor_corrector=False,
          # Use FiPy's less stable boundary condition handling, to verify that
          # this makes us reproduce FiPy's behavior.
          convection_dirichlet_mode='semi-implicit',
          convection_neumann_mode='semi-implicit',
          coupling_use_explicit_source=True,
          use_pereverzev=True,
      ),
      bootstrap_mult=0,  # remove bootstrap current
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
      config, geo, linear_theta_method.LinearThetaMethod
  )

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

"""Tests CHEASE geometry with Ip from config and psi from prescribed total j.

Ip from parameters. implicit, psi (current diffusion) only
"""

from torax import config as config_lib
from torax import geometry
from torax import sim as sim_lib
from torax.sources import source_config
from torax.stepper import linear_theta_method


def get_config() -> config_lib.Config:
  return config_lib.Config(
      profile_conditions=config_lib.ProfileConditions(
          set_pedestal=False,
      ),
      numerics=config_lib.Numerics(
          Qei_mult=0,
          ion_heat_eq=False,
          el_heat_eq=False,
          current_eq=True,
          resistivity_mult=100,  # to shorten current diffusion time
          bootstrap_mult=0,  # remove bootstrap current
          t_final=3,
      ),
      initial_psi_from_j=True,
      initial_j_is_total_current=False,
      nu=2,
      w=0.18202270915319393,
      S_pellet_tot=0,
      S_puff_tot=0,
      S_nbi_tot=0,
      transport=config_lib.TransportConfig(
          transport_model="constant",
      ),
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
  return geometry.build_chease_geometry(
      config,
      geometry_file="ITER_hybrid_citrin_equil_cheasedata.mat2cols",
      Ip_from_parameters=True,
  )


def get_sim() -> sim_lib.Sim:
  # This approach is currently lightweight because so many objects require
  # config for construction, but over time we expect to transition to most
  # config taking place via constructor args in this function.
  config = get_config()
  geo = get_geometry(config)
  return sim_lib.build_sim_from_config(
      config, geo, linear_theta_method.LinearThetaMethod
  )

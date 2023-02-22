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

"""test20: CHEASE geometry.

Ip from CHEASE. explicit, Ti only, no Pei, no pedestal, constant chi
"""

from torax import config as config_lib
from torax import geometry
from torax import sim as sim_lib
from torax.sources import source_config
from torax.tests.test_lib import explicit_stepper


def get_config() -> config_lib.Config:
  return config_lib.Config(
      set_pedestal=False,
      ion_heat_eq=True,
      el_heat_eq=False,
      dtmult=0.9,
      Ptot=200.0e6,
      w=0.18202270915319393,
      Qei_mult=0,
      t_final=0.1,
      bootstrap_mult=0,  # remove bootstrap current
      S_pellet_tot=0,
      S_puff_tot=0,
      S_nbi_tot=0,
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
          generic_ion_el_heat_source=source_config.SourceConfig(
              source_type=source_config.SourceType.FORMULA_BASED,
              is_explicit=True,
          ),
      ),
  )


def get_geometry(config: config_lib.Config) -> geometry.Geometry:
  return geometry.build_chease_geometry(
      config,
      geometry_file="ITER_hybrid_citrin_equil_cheasedata.mat2cols",
      Ip_from_parameters=False,
  )


def get_sim() -> sim_lib.Sim:
  # This approach is currently lightweight because so many objects require
  # config for construction, but over time we expect to transition to most
  # config taking place via constructor args in this function.
  config = get_config()
  geo = get_geometry(config)
  return sim_lib.build_sim_from_config(
      config, geo, explicit_stepper.ExplicitStepper
  )

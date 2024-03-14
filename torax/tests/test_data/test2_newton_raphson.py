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
from torax import geometry
from torax import sim as sim_lib
from torax.sources import source_config
from torax.tests.test_lib import sim_test_case


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
          sim_test_case.make_frozen_newton_raphson_stepper,
          config=config,
      ),
  )

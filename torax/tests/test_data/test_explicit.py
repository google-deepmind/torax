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

"""Config for test_explicit. Basic test of explicit linear solver."""

from torax import config as config_lib
from torax import geometry
from torax import sim as sim_lib
from torax.sources import source_config
from torax.tests.test_lib import explicit_stepper


def get_config() -> config_lib.Config:
  # This config based approach is deprecated.
  # Over time more will be built with pure Python constructors in `get_sim`.
  return config_lib.Config(
      set_pedestal=False,
      dtmult=0.9,
      Ptot=200.0e6,
      Qei_mult=0,
      t_final=0.1,
      bootstrap_mult=0,  # remove bootstrap current
      ion_heat_eq=True,
      el_heat_eq=False,
      # Do not use the fusion heat source.
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
      transport=config_lib.TransportConfig(transport_model='constant'),
      solver=config_lib.SolverConfig(
          predictor_corrector=False,
          use_pereverzev=False,
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
      config, geo, explicit_stepper.ExplicitStepper
  )

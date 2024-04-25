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

"""Tests combined current diffusion and heat transport with use_absolute_jext.

Implicit solver + pereverzev-corrigan, Ti+Te+Psi, Pei standard dens,
pedestal, chi from qlknn.

Same as test_psi_and_heat but with fext=0 but use_absolute_jext and Iext=3.
Result should be the same as test_psi_and_heat since fext=0 is ignored.
"""

from torax import config as config_lib
from torax import geometry
from torax import sim as sim_lib
from torax.sources import default_sources
from torax.sources import runtime_params as source_runtime_params
from torax.sources import source_models as source_models_lib
from torax.stepper import linear_theta_method
from torax.transport_model import qlknn_wrapper


def get_config() -> config_lib.Config:
  return config_lib.Config(
      profile_conditions=config_lib.ProfileConditions(
          Ti_bound_left=8,
          Te_bound_left=8,
      ),
      numerics=config_lib.Numerics(
          current_eq=True,
          resistivity_mult=100,  # to shorten current diffusion time
          t_final=2,
      ),
      # set flat Ohmic current to provide larger range of current evolution for
      # test
      nu=0,
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
  source_models.jext.runtime_params.use_absolute_jext = True
  source_models.jext.runtime_params.fext = 0.0
  source_models.jext.runtime_params.Iext = 3.0
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
  config = get_config()
  geo = get_geometry(config)
  return sim_lib.build_sim_from_config(
      config=config,
      geo=geo,
      stepper_builder=linear_theta_method.LinearThetaMethod,
      source_models=get_sources(),
      transport_model=get_transport_model(),
  )

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

"""Tests particle transport when using CGM model.

CGM transport model. Pedestal. Particle sources including NBI
"""

from torax import config as config_lib
from torax import geometry
from torax import sim as sim_lib
from torax.sources import default_sources
from torax.sources import runtime_params as source_runtime_params
from torax.sources import source_models as source_models_lib
from torax.stepper import linear_theta_method
from torax.transport_model import critical_gradient as cgm_transport_model


def get_config() -> config_lib.Config:
  return config_lib.Config(
      profile_conditions=config_lib.ProfileConditions(
          set_pedestal=True,
          nbar=0.85,  # initial density (in Greenwald fraction units)
          ne_bound_right=0.2,
          neped=1.0,
          # set flat Ohmic current to provide larger range of current evolution
          # for test
          nu=0,
      ),
      numerics=config_lib.Numerics(
          ion_heat_eq=True,
          el_heat_eq=True,
          dens_eq=True,
          current_eq=True,
          resistivity_mult=100,  # to shorten current diffusion time
          t_final=2,
      ),
      solver=config_lib.SolverConfig(
          predictor_corrector=False,
          use_pereverzev=True,
          d_per=0.0,
      ),
  )


def get_geometry(config: config_lib.Config) -> geometry.Geometry:
  return geometry.build_circular_geometry(config)


def get_transport_model() -> cgm_transport_model.CriticalGradientModel:
  return cgm_transport_model.CriticalGradientModel(
      runtime_params=cgm_transport_model.RuntimeParams(
          # CGM model ratio of ion heat conductivity to particle diffusion
          CGM_D_ratio=8,
      )
  )


def get_sources() -> source_models_lib.SourceModels:
  """Returns the source models used in the simulation."""
  source_models = default_sources.get_default_sources()
  # multiplier for ion-electron heat exchange term for sensitivity
  source_models.qei_source.runtime_params.Qei_mult = 1.0
  source_models.j_bootstrap.runtime_params.bootstrap_mult = 1.0
  # total pellet particles/s (continuous pellet model)
  source_models.sources['pellet_source'].runtime_params.S_pellet_tot = 1.0e22
  # total pellet particles/s
  source_models.sources['gas_puff_source'].runtime_params.S_puff_tot = 0.5e22
  # NBI total particle source
  source_models.sources['nbi_particle_source'].runtime_params.S_nbi_tot = 0.3e22
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

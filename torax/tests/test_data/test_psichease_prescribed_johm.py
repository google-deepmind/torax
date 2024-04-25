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
from torax.sources import default_sources
from torax.sources import runtime_params as source_runtime_params
from torax.sources import source_models as source_models_lib
from torax.stepper import linear_theta_method
from torax.transport_model import constant as constant_transport_model


def get_config() -> config_lib.Config:
  return config_lib.Config(
      profile_conditions=config_lib.ProfileConditions(
          set_pedestal=False,
          initial_psi_from_j=True,
          initial_j_is_total_current=False,
          nu=2,
      ),
      numerics=config_lib.Numerics(
          ion_heat_eq=False,
          el_heat_eq=False,
          current_eq=True,
          resistivity_mult=100,  # to shorten current diffusion time
          t_final=3,
      ),
      solver=config_lib.SolverConfig(
          predictor_corrector=False,
      ),
  )


def get_geometry(config: config_lib.Config) -> geometry.Geometry:
  return geometry.build_chease_geometry(
      config,
      geometry_file="ITER_hybrid_citrin_equil_cheasedata.mat2cols",
      Ip_from_parameters=True,
  )


def get_transport_model() -> constant_transport_model.ConstantTransportModel:
  return constant_transport_model.ConstantTransportModel()


def get_sources() -> source_models_lib.SourceModels:
  """Returns the source models used in the simulation."""
  source_models = default_sources.get_default_sources()
  # multiplier for ion-electron heat exchange term for sensitivity
  source_models.qei_source.runtime_params.Qei_mult = 0.0
  # remove bootstrap current
  source_models.j_bootstrap.runtime_params.bootstrap_mult = 0.0
  # total pellet particles/s (continuous pellet model)
  source_models.sources["pellet_source"].runtime_params.S_pellet_tot = 0.0
  # Gaussian width in normalized radial coordinate r
  source_models.sources["generic_ion_el_heat_source"].runtime_params.w = (
      0.18202270915319393
  )
  # total pellet particles/s
  source_models.sources["gas_puff_source"].runtime_params.S_puff_tot = 0
  # NBI total particle source
  source_models.sources["nbi_particle_source"].runtime_params.S_nbi_tot = 0.0
  source_models.sources["fusion_heat_source"].runtime_params.mode = (
      source_runtime_params.Mode.ZERO
  )
  source_models.sources["ohmic_heat_source"].runtime_params.mode = (
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

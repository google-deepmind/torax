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

import dataclasses
from torax import config as config_lib
from torax import geometry
from torax import sim as sim_lib
from torax.sources import default_sources
from torax.sources import runtime_params as source_runtime_params
from torax.sources import source_models as source_models_lib
from torax.stepper import runtime_params as stepper_runtime_params
from torax.tests.test_lib import explicit_stepper
from torax.transport_model import constant as constant_transport_model


def get_config() -> config_lib.Config:
  # This config based approach is deprecated.
  # Over time more will be built with pure Python constructors in `get_sim`.
  return config_lib.Config(
      profile_conditions=config_lib.ProfileConditions(
          set_pedestal=False,
      ),
      numerics=config_lib.Numerics(
          dtmult=0.9,
          t_final=0.1,
          ion_heat_eq=True,
          el_heat_eq=False,
      ),
  )


def get_geometry(config: config_lib.Config) -> geometry.Geometry:
  return geometry.build_circular_geometry(config)


def get_transport_model() -> constant_transport_model.ConstantTransportModel:
  return constant_transport_model.ConstantTransportModel()


def get_sources() -> source_models_lib.SourceModels:
  """Returns the source models used in the simulation."""
  source_models = default_sources.get_default_sources()
  # multiplier for ion-electron heat exchange term for sensitivity
  source_models.qei_source.runtime_params.Qei_mult = 0.0
  # remove bootstrap current
  source_models.j_bootstrap.runtime_params.bootstrap_mult = 0.0
  source_models.sources['generic_ion_el_heat_source'].runtime_params = (
      dataclasses.replace(
          source_models.sources['generic_ion_el_heat_source'].runtime_params,
          # total heating (including accounting for radiation) r
          Ptot=200.0e6,  # pylint: disable=unexpected-keyword-arg
          is_explicit=True,
      )
  )
  source_models.sources['fusion_heat_source'].runtime_params.mode = (
      source_runtime_params.Mode.ZERO
  )
  source_models.sources['ohmic_heat_source'].runtime_params.mode = (
      source_runtime_params.Mode.ZERO
  )
  return source_models


def get_stepper_builder() -> explicit_stepper.ExplicitStepperBuilder:
  """Returns a builder for the stepper that includes its runtime params."""
  builder = explicit_stepper.ExplicitStepperBuilder(
      runtime_params=stepper_runtime_params.RuntimeParams(
          predictor_corrector=False,
          use_pereverzev=False,
      )
  )
  return builder


def get_sim() -> sim_lib.Sim:
  # This approach is currently lightweight because so many objects require
  # config for construction, but over time we expect to transition to most
  # config taking place via constructor args in this function.
  config = get_config()
  geo = get_geometry(config)
  return sim_lib.build_sim_from_config(
      config=config,
      geo=geo,
      source_models=get_sources(),
      transport_model=get_transport_model(),
      stepper_builder=get_stepper_builder(),
  )

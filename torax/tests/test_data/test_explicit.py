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
from torax import geometry
from torax import sim as sim_lib
from torax.config import runtime_params as general_runtime_params
from torax.sources import default_sources
from torax.sources import runtime_params as source_runtime_params
from torax.sources import source_models as source_models_lib
from torax.stepper import runtime_params as stepper_runtime_params
from torax.tests.test_lib import explicit_stepper
from torax.transport_model import constant as constant_transport_model


def get_runtime_params() -> general_runtime_params.GeneralRuntimeParams:
  # This config based approach is deprecated.
  # Over time more will be built with pure Python constructors in `get_sim`.
  return general_runtime_params.GeneralRuntimeParams(
      profile_conditions=general_runtime_params.ProfileConditions(
          set_pedestal=False,
      ),
      numerics=general_runtime_params.Numerics(
          dtmult=0.9,
          t_final=0.1,
          ion_heat_eq=True,
          el_heat_eq=False,
      ),
  )


def get_geometry(
    runtime_params: general_runtime_params.GeneralRuntimeParams,
) -> geometry.Geometry:
  del runtime_params  # Unused.
  return geometry.build_circular_geometry()


def get_transport_model_builder() -> (
    constant_transport_model.ConstantTransportModel
):
  return constant_transport_model.ConstantTransportModelBuilder()


def get_sources_builder() -> source_models_lib.SourceModelsBuilder:
  """Returns the source models used in the simulation."""
  config = default_sources.get_default_source_config()
  # multiplier for ion-electron heat exchange term for sensitivity
  config['qei_source'].Qei_mult = 0.0
  # remove bootstrap current
  config['j_bootstrap'].bootstrap_mult = 0.0
  config['generic_ion_el_heat_source'] = dataclasses.replace(
      config['generic_ion_el_heat_source'],
      # total heating (including accounting for radiation) r
      Ptot=200.0e6,  # pylint: disable=unexpected-keyword-arg
      is_explicit=True,
  )
  config['fusion_heat_source'].mode = source_runtime_params.Mode.ZERO
  config['ohmic_heat_source'].mode = source_runtime_params.Mode.ZERO

  return source_models_lib.SourceModelsBuilder(config)


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
  runtime_params = get_runtime_params()
  geo = get_geometry(runtime_params)
  return sim_lib.build_sim_object(
      runtime_params=runtime_params,
      geo=geo,
      source_models_builder=get_sources_builder(),
      transport_model_builder=get_transport_model_builder(),
      stepper_builder=get_stepper_builder(),
  )

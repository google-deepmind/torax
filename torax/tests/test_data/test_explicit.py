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

from torax import sim as sim_lib
from torax.config import numerics as numerics_lib
from torax.config import profile_conditions as profile_conditions_lib
from torax.config import runtime_params as general_runtime_params
from torax.geometry import geometry_provider
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.pedestal_model import pedestal_model as pedestal_model_lib
from torax.pedestal_model import set_tped_nped
from torax.sources import runtime_params as source_runtime_params
from torax.sources import source_models as source_models_lib
from torax.tests.test_lib import default_sources
from torax.tests.test_lib import explicit_stepper
from torax.transport_model import constant as constant_transport_model


def get_runtime_params() -> general_runtime_params.GeneralRuntimeParams:
  # This config based approach is deprecated.
  # Over time more will be built with pure Python constructors in `get_sim`.
  return general_runtime_params.GeneralRuntimeParams(
      profile_conditions=profile_conditions_lib.ProfileConditions(
          ne_bound_right=0.5,
      ),
      numerics=numerics_lib.Numerics(
          dtmult=0.9,
          t_final=0.1,
          ion_heat_eq=True,
          el_heat_eq=False,
      ),
  )


def get_geometry_provider() -> geometry_provider.ConstantGeometryProvider:
  return geometry_provider.ConstantGeometryProvider(
      geometry_pydantic_model.CircularConfig().build_geometry()
  )


def get_transport_model_builder() -> (
    constant_transport_model.ConstantTransportModelBuilder
):
  return constant_transport_model.ConstantTransportModelBuilder()


def get_sources_builder() -> source_models_lib.SourceModelsBuilder:
  """Returns the source models used in the simulation."""
  source_models_builder = default_sources.get_default_sources_builder()
  # multiplier for ion-electron heat exchange term for sensitivity
  source_models_builder.runtime_params['qei_source'].Qei_mult = 0.0
  # remove bootstrap current
  source_models_builder.runtime_params['j_bootstrap'].bootstrap_mult = 0.0
  # pylint: disable=unexpected-keyword-arg
  source_models_builder.source_builders[
      'generic_ion_el_heat_source'
  ].runtime_params = dataclasses.replace(
      source_models_builder.runtime_params['generic_ion_el_heat_source'],
      # total heating (including accounting for radiation) r
      Ptot=200.0e6,  # pytype: disable=wrong-keyword-args
      is_explicit=True,
  )
  # pylint: enable=unexpected-keyword-arg
  source_models_builder.runtime_params['fusion_heat_source'].mode = (
      source_runtime_params.Mode.ZERO
  )
  source_models_builder.runtime_params['ohmic_heat_source'].mode = (
      source_runtime_params.Mode.ZERO
  )
  return source_models_builder


def get_stepper() -> explicit_stepper.ExplicitStepperModel:
  """Returns a builder for the stepper that includes its runtime params."""
  builder = explicit_stepper.ExplicitStepperModel.from_dict(
      dict(
          predictor_corrector=False,
          use_pereverzev=False,
      )
  )
  return builder


def get_pedestal_model_builder() -> pedestal_model_lib.PedestalModelBuilder:
  return set_tped_nped.SetTemperatureDensityPedestalModelBuilder()


def get_sim() -> sim_lib.Sim:
  # This approach is currently lightweight because so many objects require
  # config for construction, but over time we expect to transition to most
  # config taking place via constructor args in this function.
  runtime_params = get_runtime_params()
  geo_provider = get_geometry_provider()
  return sim_lib.Sim.create(
      runtime_params=runtime_params,
      geometry_provider=geo_provider,
      source_models_builder=get_sources_builder(),
      transport_model_builder=get_transport_model_builder(),
      stepper=get_stepper(),
      pedestal_model_builder=get_pedestal_model_builder(),
  )

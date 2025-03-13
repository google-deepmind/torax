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

from torax import sim as sim_lib
from torax.config import numerics as numerics_lib
from torax.config import profile_conditions as profile_conditions_lib
from torax.config import runtime_params as general_runtime_params
from torax.geometry import geometry_provider
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.pedestal_model import pydantic_model as pedestal_pydantic_model
from torax.sources import pydantic_model as source_pydantic_model
from torax.sources import runtime_params as source_runtime_params
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


def get_sources() -> source_pydantic_model.Sources:
  """Returns the source models used in the simulation."""
  sources = default_sources.get_default_sources()
  sources_dict = sources.to_dict()
  sources_dict = sources_dict['source_model_config']
  # multiplier for ion-electron heat exchange term for sensitivity
  sources_dict['qei_source']['Qei_mult'] = 0.0
  # remove bootstrap current
  sources_dict['j_bootstrap']['bootstrap_mult'] = 0.0
  # total heating (including accounting for radiation) r
  sources_dict['generic_ion_el_heat_source']['Ptot'] = 200.0e6
  sources_dict['generic_ion_el_heat_source']['is_explicit'] = True
  sources_dict['fusion_heat_source']['mode'] = source_runtime_params.Mode.ZERO
  sources_dict['ohmic_heat_source']['mode'] = source_runtime_params.Mode.ZERO

  sources = source_pydantic_model.Sources.from_dict(sources_dict)
  return sources


def get_stepper() -> explicit_stepper.ExplicitStepperModel:
  """Returns a builder for the stepper that includes its runtime params."""
  builder = explicit_stepper.ExplicitStepperModel.from_dict(
      dict(
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
  geo_provider = get_geometry_provider()
  return sim_lib.Sim.create(
      runtime_params=runtime_params,
      geometry_provider=geo_provider,
      sources=get_sources(),
      transport_model_builder=get_transport_model_builder(),
      stepper=get_stepper(),
      pedestal=pedestal_pydantic_model.Pedestal(),
  )

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

"""Tests semi-implicit convection as carried out with FiPy.

Semi-implicit convection can lead to numerical instability at boundary
condition. No pedestal, implicit + pereverzev-corrigan, Ti+Te,
Pei standard dens, chi from CGM.
"""

from torax import geometry
from torax import sim as sim_lib
from torax.config import runtime_params as general_runtime_params
from torax.sources import default_sources
from torax.sources import runtime_params as source_runtime_params
from torax.sources import source_models as source_models_lib
from torax.stepper import linear_theta_method
from torax.stepper import runtime_params as stepper_runtime_params
from torax.transport_model import critical_gradient as cgm_transport_model


def get_runtime_params() -> general_runtime_params.GeneralRuntimeParams:
  return general_runtime_params.GeneralRuntimeParams(
      # This is test_cgm_heat but modified to not use the pedestal feature,
      # to exercise the convection term at the boundary. This causes FiPy to
      # explode. The time was reduced compared to test_cgm_heat to avoid test
      # time bottlenecks
      profile_conditions=general_runtime_params.ProfileConditions(
          set_pedestal=False,
      ),
      numerics=general_runtime_params.Numerics(
          t_final=0.5,
      ),
  )


def get_geometry(
    runtime_params: general_runtime_params.GeneralRuntimeParams,
) -> geometry.Geometry:
  del runtime_params  # Unused.
  return geometry.build_circular_geometry()


def get_transport_model() -> cgm_transport_model.CriticalGradientModel:
  return cgm_transport_model.CriticalGradientModel()


def get_sources() -> source_models_lib.SourceModels:
  """Returns the source models used in the simulation."""
  source_models = default_sources.get_default_sources()
  # remove bootstrap current
  source_models.j_bootstrap.runtime_params.bootstrap_mult = 0.0
  source_models.sources['fusion_heat_source'].runtime_params.mode = (
      source_runtime_params.Mode.ZERO
  )
  source_models.sources['ohmic_heat_source'].runtime_params.mode = (
      source_runtime_params.Mode.ZERO
  )
  return source_models


def get_stepper_builder() -> linear_theta_method.LinearThetaMethodBuilder:
  """Returns a builder for the stepper that includes its runtime params."""
  builder = linear_theta_method.LinearThetaMethodBuilder(
      runtime_params=stepper_runtime_params.RuntimeParams(
          predictor_corrector=False,
          # Use FiPy's less stable boundary condition handling, to verify that
          # this makes us reproduce FiPy's behavior.
          convection_dirichlet_mode='semi-implicit',
          convection_neumann_mode='semi-implicit',
          use_pereverzev=True,
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
      stepper_builder=get_stepper_builder(),
      source_models=get_sources(),
      transport_model=get_transport_model(),
  )

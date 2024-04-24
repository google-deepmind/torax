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

"""Newton-Raphson zero iteration test. Linear solver initial condition.

Tests that Newton-Raphson method behaves the same as linear method
when Newton-Raphson is set to use zero iterations and use linear
method for the initial guess.

implicit + pereverzev-corrigan, Ti+Te+Psi, Pei standard dens, pedestal, chi from
qlknn
Used for a nonlinear run with zero iterations. dt adaptation is turned off to
avoid backtracking
to tiny timesteps (and test timeouts)
"""

from torax import config as config_lib
from torax import geometry
from torax import sim as sim_lib
from torax.fvm import enums
from torax.sources import source_config
from torax.sources import source_models as source_models_lib
from torax.stepper import nonlinear_theta_method
from torax.stepper import stepper as stepper_lib
from torax.transport_model import qlknn_wrapper
from torax.transport_model import transport_model as transport_model_lib


def make_linear_newton_raphson_stepper(
    transport_model: transport_model_lib.TransportModel,
    source_models: source_models_lib.SourceModels,
) -> stepper_lib.Stepper:
  """Makes a Newton Raphson stepper that behaves linearly.

  The solver is configured to use a linear initial guess, then run for 0
  iterations.
  Under these conditions, we can test that the nonlinear stepper behaves the
  same as
  the linear solver.

  Args:
    transport_model: Transport model.
    source_models: TORAX sources/sinks used to compute profile terms in the
      state evolution equations.

  Returns:
    Stepper: the stepper.
  """
  return nonlinear_theta_method.NewtonRaphsonThetaMethod(
      transport_model,
      source_models=source_models,
      initial_guess_mode=enums.InitialGuessMode.LINEAR,
      maxiter=0,
  )


def get_config() -> config_lib.Config:
  # This config based approach is deprecated.
  # Over time more will be built with pure Python constructors in `get_sim`.
  return config_lib.Config(
      profile_conditions=config_lib.ProfileConditions(
          Ti_bound_left=8,
          Te_bound_left=8,
      ),
      numerics=config_lib.Numerics(
          current_eq=True,
          adaptive_dt=False,
          # to shorten current diffusion time for the test
          resistivity_mult=100,
          t_final=2,
          bootstrap_mult=0,  # remove bootstrap current
      ),
      # set flat Ohmic current to provide larger range of current evolution for
      # test
      nu=0,
      solver=config_lib.SolverConfig(
          predictor_corrector=False,
          use_pereverzev=True,
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


def get_transport_model() -> qlknn_wrapper.QLKNNTransportModel:
  return qlknn_wrapper.QLKNNTransportModel()


def get_sim() -> sim_lib.Sim:
  # This approach is currently lightweight because so many objects require
  # config for construction, but over time we expect to transition to most
  # config taking place via constructor args in this function.
  config = get_config()
  geo = get_geometry(config)
  return sim_lib.build_sim_from_config(
      config=config,
      geo=geo,
      stepper_builder=make_linear_newton_raphson_stepper,
      transport_model=get_transport_model(),
  )

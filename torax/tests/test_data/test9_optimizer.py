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

"""Config for test9_optimizer.

Tests that we get the same result as the linear solver, when we use
a nonlinear solver but then hack it to act like a linear solver by
using 0 iterations and an initial guess based on the linear solver.
"""

from torax import config as config_lib
from torax import geometry
from torax import sim as sim_lib
from torax.fvm import enums
from torax.sources import source_config
from torax.sources import source_profiles
from torax.stepper import nonlinear_theta_method
from torax.stepper import stepper as stepper_lib
from torax.transport_model import transport_model as transport_model_lib


def make_linear_optimizer_stepper(
    transport_model: transport_model_lib.TransportModel,
    sources: source_profiles.Sources,
) -> stepper_lib.Stepper:
  """Makes an optimizer stepper that behaves linearly.

  The optimizer is configured to use a linear initial guess, then run for 0
  iterations.
  Under these conditions, we can test that the optimizer behaves the same as
  the linear solver.

  Args:
    transport_model: Transport model.
    sources: TORAX sources/sinks used to compute profile terms in the state
      evolution equations.

  Returns:
    Stepper: the stepper.
  """
  return nonlinear_theta_method.OptimizerThetaMethod(
      transport_model,
      sources=sources,
      initial_guess_mode=enums.InitialGuessMode.LINEAR,
      maxiter=0,
  )


def get_config() -> config_lib.Config:
  # This config based approach is deprecated.
  # Over time more will be built with pure Python constructors in `get_sim`.
  return config_lib.Config(
      Ti_bound_left=8,
      Te_bound_left=8,
      current_eq=True,
      adaptive_dt=False,
      # to shorten current diffusion time for the test
      resistivity_mult=100,
      # set flat Ohmic current to provide larger range of current evolution for
      # test
      nu=0,
      t_final=2,
      solver=config_lib.SolverConfig(
          predictor_corrector=False,
          coupling_use_explicit_source=True,
          use_pereverzev=True,
      ),
      bootstrap_mult=0,  # remove bootstrap current
      transport=config_lib.TransportConfig(
          transport_model="qlknn",
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


def get_sim() -> sim_lib.Sim:
  # This approach is currently lightweight because so many objects require
  # config for construction, but over time we expect to transition to most
  # config taking place via constructor args in this function.
  config = get_config()
  geo = get_geometry(config)
  return sim_lib.build_sim_from_config(
      config, geo, make_linear_optimizer_stepper
  )

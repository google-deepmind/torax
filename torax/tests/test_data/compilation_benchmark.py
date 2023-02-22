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

"""Compilating time benchmark script.

This config is not run as an actual automated test, but is convenient to have
as a manual test to invoke from time to time. It is configured to incur high
but not extreme compile time: high enough that compile time problems are
obvious, not so high that iterating on solutions to compile time problems
is infeasible.
"""

from torax import config as config_lib
from torax import geometry
from torax import sim as sim_lib
from torax.stepper import nonlinear_theta_method


def get_config() -> config_lib.Config:
  # This config based approach is deprecated.
  # Over time more will be built with pure Python constructors in `get_sim`.
  return config_lib.Config(
      set_pedestal=True,
      Qei_mult=1,
      ion_heat_eq=True,
      el_heat_eq=True,
      dens_eq=True,
      current_eq=True,
      resistivity_mult=100,  # to shorten current diffusion time for the test
      bootstrap_mult=1,  # remove bootstrap current
      nu=0,
      fGW=0.85,  # initial density (Greenwald fraction)
      S_pellet_tot=1.0e22,
      S_puff_tot=0.5e22,
      S_nbi_tot=0.3e22,
      ne_bound_right=0.2,
      neped=1.0,
      t_final=0.0007944 * 2,
      Ptot=53.0e6,  # total external heating
      transport=config_lib.TransportConfig(
          DVeff=False,
          transport_model='qlknn',
      ),
      solver=config_lib.SolverConfig(
          use_pereverzev=False,
      ),
  )


def get_geometry(config: config_lib.Config) -> geometry.Geometry:
  return geometry.build_circular_geometry(config)


def get_sim() -> sim_lib.Sim:
  # This approach is currently lightweight because so many objects require
  # config for construction, but over time we expect to transition to most
  # config taking place via constructor args in this function.
  config = get_config()
  return sim_lib.build_sim_from_config(
      config,
      get_geometry(config),
      nonlinear_theta_method.NewtonRaphsonThetaMethod,
  )

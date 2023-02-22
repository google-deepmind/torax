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

"""test40: Mock-up of ITER baseline with Tped=4."""

from torax import config as config_lib
from torax import geometry
from torax import sim as sim_lib
from torax.sources import source_config
from torax.stepper import linear_theta_method


def get_config() -> config_lib.Config:
  return config_lib.Config(
      Zeff=1.7,
      dtmult=100,
      nr=25,  # radial grid points
      maxdt=1.0e-0,
      set_pedestal=True,
      Ti_bound_right=0.2,
      Te_bound_right=0.2,
      Tiped=4,
      Teped=4,
      Qei_mult=1,
      ion_heat_eq=True,
      el_heat_eq=True,
      dens_eq=True,
      npeak=1.5,  # Peaking factor of density profile
      current_eq=True,
      resistivity_mult=500,  # to shorten current diffusion time for the test
      bootstrap_mult=1,  # remove bootstrap current
      fext=0.0,  # total "external" current fraction
      # width of "external" Gaussian current profile (normalized radial
      # coordinate)
      wext=0.0,
      # radius of "external" Gaussian current profile (normalized radial
      # coordinate)
      rext=0.0,
      # set flat Ohmic current to provide larger range of current evolution for
      # test
      nu=0,
      fGW=0.85,  # initial density (Greenwald fraction)
      w=0.18202270915319393,
      S_pellet_tot=1.0e22,
      S_puff_tot=0.5e22,
      S_nbi_tot=0.3e22,
      ne_bound_right=0.1,
      neped=0.65,
      t_final=10.0,
      Ptot=53.0e6,  # total external heating
      transport=config_lib.TransportConfig(
          transport_model="qlknn",
          DVeff=True,
          coll_mult=0.25,
          apply_inner_patch=True,
          De_inner=0.5,
          chii_inner=1.0,
          chie_inner=0.75,
          rho_inner=0.3,  # radius below which patch transport is applied
      ),
      solver=config_lib.SolverConfig(
          predictor_corrector=False,
          coupling_use_explicit_source=True,
          convection_dirichlet_mode="semi-implicit",
          convection_neumann_mode="semi-implicit",
          use_pereverzev=True,
      ),
      sources=dict(
          ohmic_heat_source=source_config.SourceConfig(
              source_type=source_config.SourceType.ZERO,
          ),
      ),
  )


def get_geometry(config: config_lib.Config) -> geometry.Geometry:
  return geometry.build_chease_geometry(
      config,
      geometry_file="ITER_hybrid_citrin_equil_cheasedata.mat2cols",
      Ip_from_parameters=True,
  )


def get_sim() -> sim_lib.Sim:
  # This approach is currently lightweight because so many objects require
  # config for construction, but over time we expect to transition to most
  # config taking place via constructor args in this function.
  config = get_config()
  geo = get_geometry(config)
  return sim_lib.build_sim_from_config(
      config, geo, linear_theta_method.LinearThetaMethod
  )

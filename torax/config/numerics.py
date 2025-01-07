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

"""Numerics parameters used throughout TORAX simulations."""

from __future__ import annotations

import chex
from torax import array_typing
from torax import interpolated_param
from torax.config import base
from torax.geometry import geometry
from typing_extensions import override


# pylint: disable=invalid-name


@chex.dataclass
class Numerics(base.RuntimeParametersConfig):
  """Generic numeric parameters for the simulation."""

  # simulation control
  # start of simulation, in seconds
  t_initial: float = 0.0
  # end of simulation, in seconds
  t_final: float = 5.0
  # If True, ensures that if the simulation runs long enough, one step
  # occurs exactly at `t_final`.
  exact_t_final: bool = False

  # maximum and minimum timesteps allowed in simulation
  maxdt: float = 1e-1  #  only used with chi_time_step_calculator
  mindt: float = 1e-8  #  if adaptive timestep is True, error raised if dt<mindt

  # prefactor in front of chi_timestep_calculator base timestep dt=dx^2/(2*chi).
  # In most use-cases can be increased further above this conservative default
  dtmult: float = 0.9 * 10

  fixed_dt: float = 1e-2  # timestep used for fixed_time_step_calculator

  # Iterative reduction of dt if nonlinear step does not converge,
  # If nonlinear step does not converge, then the step is redone
  # iteratively at successively lower dt until convergence is reached
  adaptive_dt: bool = True
  dt_reduction_factor: float = 3

  # Solve the ion heat equation (ion temperature evolves over time)
  ion_heat_eq: bool = True
  # Solve the electron heat equation (electron temperature evolves over time)
  el_heat_eq: bool = True
  # Solve the current equation (psi evolves over time driven by the solver;
  # q and s evolve over time as a function of psi)
  current_eq: bool = False
  # Solve the density equation (n evolves over time)
  dens_eq: bool = False
  # Enable time-dependent prescribed profiles.
  # This option is provided to allow initialization of density profiles scaled
  # to a Greenwald fraction, and freeze this density even if the current is time
  # evolving. Otherwise the density will evolve to always maintain that GW frac.
  enable_prescribed_profile_evolution: bool = True

  # Calculate Phibdot in the geometry dataclasses. This is used in calc_coeffs
  # to calculate terms related to time-dependent geometry. Can set to false to
  # zero out for testing purposes.
  calcphibdot: bool = True

  # q-profile correction factor. Used only in ad-hoc circular geometry model
  q_correction_factor: float = 1.25
  # 1/multiplication factor for sigma (conductivity) to reduce current
  # diffusion timescale to be closer to heat diffusion timescale
  resistivity_mult: interpolated_param.TimeInterpolatedInput = 1.0

  # density profile info
  # Reference value for normalization
  nref: float = 1e20

  # numerical (e.g. no. of grid points, other info needed by solver)
  # effective source to dominate PDE in internal boundary condtion location
  # if T != Tped
  largeValue_T: float = 2.0e10
  # effective source to dominate density PDE in internal boundary condtion
  # location if n != neped
  largeValue_n: float = 2.0e8

  @override
  def make_provider(
      self, torax_mesh: geometry.Grid1D | None = None
  ) -> NumericsProvider:
    return NumericsProvider(**self.get_provider_kwargs(torax_mesh))

  def __post_init__(self):
    if self.dtmult <= 0.0:
      raise ValueError(f'dtmult must be positive, got {self.dtmult}')


@chex.dataclass
class NumericsProvider(base.RuntimeParametersProvider['DynamicNumerics']):
  """Generic numeric parameters for the simulation."""

  runtime_params_config: Numerics
  resistivity_mult: interpolated_param.InterpolatedVarSingleAxis

  @override
  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> DynamicNumerics:
    """Builds a DynamicNumerics."""
    return DynamicNumerics(
        t_initial=self.runtime_params_config.t_initial,
        t_final=self.runtime_params_config.t_final,
        exact_t_final=self.runtime_params_config.exact_t_final,
        maxdt=self.runtime_params_config.maxdt,
        mindt=self.runtime_params_config.mindt,
        dtmult=self.runtime_params_config.dtmult,
        fixed_dt=self.runtime_params_config.fixed_dt,
        dt_reduction_factor=self.runtime_params_config.dt_reduction_factor,
        enable_prescribed_profile_evolution=self.runtime_params_config.enable_prescribed_profile_evolution,
        calcphibdot=self.runtime_params_config.calcphibdot,
        q_correction_factor=self.runtime_params_config.q_correction_factor,
        resistivity_mult=self.resistivity_mult.get_value(t),
        nref=self.runtime_params_config.nref,
        largeValue_T=self.runtime_params_config.largeValue_T,
        largeValue_n=self.runtime_params_config.largeValue_n,
    )


@chex.dataclass
class DynamicNumerics:
  """Generic numeric parameters for the simulation."""

  t_initial: float
  t_final: float
  exact_t_final: bool
  maxdt: float
  mindt: float
  dtmult: float
  fixed_dt: float
  dt_reduction_factor: float
  q_correction_factor: float
  resistivity_mult: array_typing.ScalarFloat
  nref: float
  largeValue_T: float
  largeValue_n: float
  enable_prescribed_profile_evolution: bool
  calcphibdot: bool

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
import pydantic
from torax import array_typing
from torax import interpolated_param
from torax.config import base
from torax.torax_pydantic import torax_pydantic
from typing_extensions import override
from typing_extensions import Self

# pylint: disable=invalid-name


class NumericsPydantic(torax_pydantic.BaseModelFrozen):
  """Generic numeric parameters for the simulation.

  The `from_dict(...)` method can accept a dictionary defined by
  https://torax.readthedocs.io/en/latest/configuration.html#numerics.

  Attributes:
    t_initial: Simulation start time, in units of seconds.
    t_final: Simulation end time, in units of seconds.
    exact_t_final: If True, ensures that the simulation end time is exactly
      `t_final`, by adapting the final `dt` to match.
    maxdt: Maximum timesteps allowed in the simulation. This is only used with
      the `chi_time_step_calculator` time_step_calculator.
    mindt: Minimum timestep allowed in simulation.
    dtmult: Prefactor in front of chi_timestep_calculator base timestep
      dt=dx^2/(2*chi). In most use-cases can be increased further above this.
    fixed_dt: Timestep used for `fixed_time_step_calculator`.
    adaptive_dt: Iterative reduction of dt if nonlinear step does not converge,
      if nonlinear step does not converge, then the step is redone iteratively
      at successively lower dt until convergence is reached.
    dt_reduction_factor: Factor by which to reduce dt if adaptive_dt is True.
    ion_heat_eq: Solve the ion heat equation (ion temperature evolves over
      time).
    el_heat_eq: Solve the electron heat equation (electron temperature evolves
      over time)
    current_eq: Solve the current equation (current evolves over time).
    dens_eq: Solve the density equation (n evolves over time).
    calcphibdot: Calculate Phibdot in the geometry dataclasses. This is used in
      calc_coeffs to calculate terms related to time-dependent geometry. Can set
      to false to zero out for testing purposes.
    resistivity_mult:  1/multiplication factor for sigma (conductivity) to
      reduce current diffusion timescale to be closer to heat diffusion
      timescale
    nref: Reference density value for normalizations.
    largeValue_T: Prefactor for adaptive source term for setting temperature
      internal boundary conditions.
    largeValue_n: Prefactor for adaptive source term for setting density
      internal boundary conditions.
  """

  t_initial: torax_pydantic.Second = 0.0
  t_final: torax_pydantic.Second = 5.0
  exact_t_final: bool = False
  maxdt: torax_pydantic.Second = 1e-1
  mindt: torax_pydantic.Second = 1e-8
  dtmult: pydantic.PositiveFloat = 9.0
  fixed_dt: torax_pydantic.Second = 1e-2
  adaptive_dt: bool = True
  dt_reduction_factor: pydantic.PositiveFloat = 3
  ion_heat_eq: bool = True
  el_heat_eq: bool = True
  current_eq: bool = False
  dens_eq: bool = False
  calcphibdot: bool = True
  resistivity_mult: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  nref: pydantic.PositiveFloat = 1e20
  largeValue_T: pydantic.PositiveFloat = 2.0e10
  largeValue_n: pydantic.PositiveFloat = 2.0e8

  @pydantic.model_validator(mode='after')
  def model_validation(self) -> Self:
    if self.t_initial > self.t_final:
      raise ValueError(
          't_initial must be less than or equal to t_final. '
          f't_initial: {self.t_initial}, t_final: {self.t_final}'
      )

    if self.mindt > self.maxdt:
      raise ValueError(
          'maxdt must be greater than or equal to mindt. '
          f'maxdt: {self.maxdt}, mindt: {self.mindt}'
      )
    return self


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

  # Calculate Phibdot in the geometry dataclasses. This is used in calc_coeffs
  # to calculate terms related to time-dependent geometry. Can set to false to
  # zero out for testing purposes.
  calcphibdot: bool = True
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
      self, torax_mesh: torax_pydantic.Grid1D | None = None
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
        calcphibdot=self.runtime_params_config.calcphibdot,
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
  resistivity_mult: array_typing.ScalarFloat
  nref: float
  largeValue_T: float
  largeValue_n: float
  calcphibdot: bool

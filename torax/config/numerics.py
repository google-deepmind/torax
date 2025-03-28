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
import chex
import pydantic
from torax import array_typing
from torax.torax_pydantic import torax_pydantic
from typing_extensions import Self


# pylint: disable=invalid-name
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
  enable_sanity_checks: bool


class Numerics(torax_pydantic.BaseModelFrozen):
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
    enable_sanity_checks: If True, enables runtime validation checks for 
      configuration parameters, simulation state, and step outputs to detect
      issues early such as NaN values, negative temperatures/densities,
      and other physically unrealistic values.
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
  enable_sanity_checks: bool = False

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

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> DynamicNumerics:
    """Builds a DynamicNumerics."""
    return DynamicNumerics(
        t_initial=self.t_initial,
        t_final=self.t_final,
        exact_t_final=self.exact_t_final,
        maxdt=self.maxdt,
        mindt=self.mindt,
        dtmult=self.dtmult,
        fixed_dt=self.fixed_dt,
        dt_reduction_factor=self.dt_reduction_factor,
        calcphibdot=self.calcphibdot,
        resistivity_mult=self.resistivity_mult.get_value(t),
        nref=self.nref,
        largeValue_T=self.largeValue_T,
        largeValue_n=self.largeValue_n,
        enable_sanity_checks=self.enable_sanity_checks,
    )

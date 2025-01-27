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
from torax.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name


class Numerics(torax_pydantic.BaseModelMutable):
  """Generic numeric parameters for the simulation."""

  # simulation control
  # start of simulation, in seconds
  t_initial: pydantic.NonNegativeFloat = 0.0
  # end of simulation, in seconds
  t_final: pydantic.PositiveFloat = 5.0
  # If True, ensures that if the simulation runs long enough, one step
  # occurs exactly at `t_final`.
  exact_t_final: bool = False

  # maximum and minimum timesteps allowed in simulation
  maxdt: pydantic.PositiveFloat = (
      1e-1  #  only used with chi_time_step_calculator
  )
  mindt: pydantic.PositiveFloat = (
      1e-8  #  if adaptive timestep is True, error raised if dt<mindt
  )

  # prefactor in front of chi_timestep_calculator base timestep dt=dx^2/(2*chi).
  # In most use-cases can be increased further above this conservative default
  dtmult: pydantic.PositiveFloat = 0.9 * 10

  fixed_dt: pydantic.PositiveFloat = (
      1e-2  # timestep used for fixed_time_step_calculator
  )

  # Iterative reduction of dt if nonlinear step does not converge,
  # If nonlinear step does not converge, then the step is redone
  # iteratively at successively lower dt until convergence is reached
  adaptive_dt: bool = True
  dt_reduction_factor: pydantic.PositiveFloat = 3

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
  q_correction_factor: pydantic.PositiveFloat = 1.25
  # 1/multiplication factor for sigma (conductivity) to reduce current
  # diffusion timescale to be closer to heat diffusion timescale
  resistivity_mult: torax_pydantic.TimeVaryingScalar = pydantic.Field(
      default_factory=lambda: 1.0, validate_default=True
  )
  # density profile info
  # Reference value for normalization
  nref: pydantic.PositiveFloat = 1e20

  # numerical (e.g. no. of grid points, other info needed by solver)
  # effective source to dominate PDE in internal boundary condtion location
  # if T != Tped
  largeValue_T: pydantic.PositiveFloat = 2.0e10
  # effective source to dominate density PDE in internal boundary condtion
  # location if n != neped
  largeValue_n: pydantic.PositiveFloat = 2.0e8

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
        enable_prescribed_profile_evolution=self.enable_prescribed_profile_evolution,
        calcphibdot=self.calcphibdot,
        q_correction_factor=self.q_correction_factor,
        resistivity_mult=self.resistivity_mult.get_value(t),
        nref=self.nref,
        largeValue_T=self.largeValue_T,
        largeValue_n=self.largeValue_n,
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

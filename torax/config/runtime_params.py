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

"""General runtime input parameters used throughout TORAX simulations."""

from __future__ import annotations

import dataclasses

import chex
from torax import interpolated_param


# Type-alias for clarity. While the InterpolatedVar1ds can vary across any
# field, in here, we mainly use it to handle time-dependent parameters.
TimeInterpolatedScalar = interpolated_param.TimeInterpolatedScalar
# Type-alias for clarity for time-and-rho-dependent parameters.
TimeInterpolatedArray = (
    interpolated_param.TimeInterpolatedArray
)
# Type-alias for brevity.
InterpolationMode = interpolated_param.InterpolationMode
InterpolatedVar1d = interpolated_param.InterpolatedVar1d


# pylint: disable=invalid-name


@chex.dataclass
class PlasmaComposition:
  # amu of main ion (if multiple isotope, make average)
  Ai: float = 2.5
  # charge of main ion
  Zi: float = 1.0
  # needed for qlknn and fusion power
  Zeff: TimeInterpolatedScalar = 1.0
  Zimp: TimeInterpolatedScalar = (
      10.0  # impurity charge state assumed for dilution
  )


@chex.dataclass
class ProfileConditions:
  """Prescribed values and boundary conditions for the core profiles."""

  # total plasma current in MA
  # Note that if Ip_from_parameters=False in geometry, then this Ip will be
  # overwritten by values from the geometry data
  Ip: TimeInterpolatedScalar = 15.0

  # Temperature boundary conditions at r=Rmin
  Ti_bound_right: TimeInterpolatedScalar = 1.0
  Te_bound_right: TimeInterpolatedScalar = 1.0
  # Prescribed or evolving values for temperature at different times.
  # The outer mapping is for times and the inner mapping is for values of
  # temperature along the rho grid.
  Ti: TimeInterpolatedArray = dataclasses.field(
      default_factory=lambda: {0: {0: 15.0, 1: 1.0}}
  )
  Te: TimeInterpolatedArray = dataclasses.field(
      default_factory=lambda: {0: {0: 15.0, 1: 1.0}}
  )

  # Peaking factor of density profile.
  # If density evolves with PDE (dens_eq=True), then is initial condition
  npeak: TimeInterpolatedScalar = 1.5

  # Initial line averaged density.
  # In units of reference density if nbar_is_fGW = False.
  # In Greenwald fraction if nbar_is_fGW = True.
  # nGW = Ip/(pi*a^2) with a in m, nGW in 10^20 m-3, Ip in MA
  nbar: TimeInterpolatedScalar = 0.85
  # Toggle units of nbar
  nbar_is_fGW: bool = True

  # Density boundary condition for r=Rmin.
  # In units of reference density if ne_bound_right_is_fGW = False.
  # In Greenwald fraction if ne_bound_right_is_fGW = True.
  ne_bound_right: TimeInterpolatedScalar = 0.5
  ne_bound_right_is_fGW: bool = False

  # Internal boundary condition (pedestal)
  # Do not set internal boundary condition if this is False
  set_pedestal: TimeInterpolatedScalar = True
  # ion pedestal top temperature in keV
  Tiped: TimeInterpolatedScalar = 5.0
  # electron pedestal top temperature in keV
  Teped: TimeInterpolatedScalar = 5.0
  # pedestal top electron density
  # In units of reference density if neped_is_fGW = False.
  # In Greenwald fraction if neped_is_fGW = True.
  neped: TimeInterpolatedScalar = 0.7
  neped_is_fGW: bool = False
  # Set ped top location.
  Ped_top: TimeInterpolatedScalar = 0.91

  # current profiles (broad "Ohmic" + localized "external" currents)
  # peaking factor of "Ohmic" current: johm = j0*(1 - r^2/a^2)^nu
  nu: float = 3.0
  # toggles if "Ohmic" current is treated as total current upon initialization,
  # or if non-inductive current should be included in initial jtot calculation
  initial_j_is_total_current: bool = False
  # toggles if the initial psi calculation is based on the "nu" current formula,
  # or from the psi available in the numerical geometry file. This setting is
  # ignored for the ad-hoc circular geometry, which has no numerical geometry.
  initial_psi_from_j: bool = False


@chex.dataclass
class Numerics:
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

  # q-profile correction factor. Used only in ad-hoc circular geometry model
  q_correction_factor: float = 1.25
  # 1/multiplication factor for sigma (conductivity) to reduce current
  # diffusion timescale to be closer to heat diffusion timescale
  resistivity_mult: TimeInterpolatedScalar = 1.0

  # density profile info
  # Reference value for normalization
  nref: float = 1e20

  # numerical (e.g. no. of grid points, other info needed by solver)
  # effective source to dominate PDE in internal boundary condtion location
  # if T != Tped
  largeValue_T: float = 1.0e10
  # effective source to dominate density PDE in internal boundary condtion
  # location if n != neped
  largeValue_n: float = 1.0e8


# NOMUTANTS -- It's expected for the tests to pass with different defaults.
@chex.dataclass
class GeneralRuntimeParams:
  """General runtime input parameters for the `torax` module."""

  plasma_composition: PlasmaComposition = dataclasses.field(
      default_factory=PlasmaComposition
  )
  profile_conditions: ProfileConditions = dataclasses.field(
      default_factory=ProfileConditions
  )
  numerics: Numerics = dataclasses.field(default_factory=Numerics)

  # 'File directory where the simulation outputs will be saved. If not '
  # 'provided, this will default to /tmp/torax_results_<YYYYMMDD_HHMMSS>/.',
  output_dir: str | None = None

  # pylint: enable=invalid-name

  def sanity_check(self) -> None:
    """Checks that various configuration parameters are valid."""
    # TODO(b/330172917) do more extensive config parameter sanity checking

    # These are floats, not jax types, so we can use direct asserts.
    assert self.numerics.dtmult > 0.0
    assert isinstance(self.plasma_composition, PlasmaComposition)
    assert isinstance(self.numerics, Numerics)

  def __post_init__(self):
    self.sanity_check()

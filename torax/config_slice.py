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

"""Inputs to the TORAX steppers based on the input config.

When running a TORAX simulation, the steppers are (by default) JAX-compiled
function, meaning it has two types of arguments: "dynamic" and "static".

The "dynamic" arguments can change from call to call. These arguments must be
arrays, scalars, or standard (possibly nested) Python containers. See the JAX
docs for more info on allowed types. They cannot influence the logical branches
the JointStateStepper may take (again, see the sharp bits in the JAX docs to
learn more about the how these "dynamic" args can be used within the function).

Note that the "dynamic" arguments are NOT necessarily time-dependent. They do
not need to vary from time step to time step (though they can). They can change
from time step to time step, or from simulation run to simulation run, without
triggering a recompile. Changing these params without needing to recompile the
stepper is the defining quality of the dynamic arguments.

The "static" arguments are compile-time constant. Any changes to them would
trigger a recompilation of the stepper. These arguments don't have the same
restrictions as the dynamic arguments both in terms of types and how they are
used.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Callable

import chex
from torax import config as config_lib
from torax.runtime_params import config_slice_args
from torax.sources import runtime_params as sources_params
from torax.stepper import runtime_params as stepper_params
from torax.transport_model import runtime_params as transport_model_params


# Many of the variables follow scientific or mathematical notation, so disable
# pylint complaints.
# pylint: disable=invalid-name


@chex.dataclass(frozen=True)
class DynamicConfigSlice:
  """Input params that are ok to use as inputs to a JAX-compiled function.

  This PyTree of params is input to the sim.JointStateStepper, which updates
  the joint state and evolves the mesh state. This config includes various
  "dynamic" parameters which can change from step to step, or from
  simulation run to simulation run, without requiring the JointStateStepper to
  recompile.

  Note that "dynamic" does NOT mean time dependent necessarily (though these
  params can be time dependent). Here "dynamic" means these params can change
  without trigerring or requiring a recompile.

  While the parameters are not necessarily time-dependent, that is how the class
  gets its name: a config "slice" refers to a subset of the overall TORAX config
  at a specific time t.
  """

  transport: transport_model_params.DynamicRuntimeParams
  stepper: stepper_params.DynamicRuntimeParams
  plasma_composition: DynamicPlasmaComposition
  profile_conditions: DynamicProfileConditions
  numerics: DynamicNumerics
  sources: Mapping[str, sources_params.DynamicRuntimeParams]


@chex.dataclass
class DynamicPlasmaComposition:
  # amu of main ion (if multiple isotope, make average)
  Ai: float
  # charge of main ion
  Zi: float
  # needed for qlknn and fusion power
  Zeff: float
  Zimp: float  # impurity charge state assumed for dilution


@chex.dataclass
class DynamicProfileConditions:
  """Prescribed values and boundary conditions for the core profiles."""

  # total plasma current in MA
  # Note that if Ip_from_parameters=False in geometry, then this Ip will be
  # overwritten by values from the geometry data
  Ip: float

  # Temperature boundary conditions at r=Rmin
  Ti_bound_right: float
  Te_bound_right: float
  # Prescribed values for r=0. When evolving, then is initial condition.
  Te_bound_left: float
  Ti_bound_left: float

  # Peaking factor of density profile.
  # If density evolves with PDE (dens_eq=True), then is initial condition
  npeak: float

  # Initial line averaged density.
  # In units of reference density if nbar_is_fGW = False.
  # In Greenwald fraction if nbar_is_fGW = True.
  # nGW = Ip/(pi*a^2) with a in m, nGW in 10^20 m-3, Ip in MA
  nbar: float
  # Toggle units of nbar
  nbar_is_fGW: bool

  # Density boundary condition for r=Rmin, units of nref
  # In units of reference density if ne_bound_right_is_fGW = False.
  # In Greenwald fraction if ne_bound_right_is_fGW = True.
  ne_bound_right: float
  ne_bound_right_is_fGW: bool

  # Internal boundary condition (pedestal)
  # Do not set internal boundary condition if this is False
  set_pedestal: bool
  # ion pedestal top temperature in keV for Ti and Te
  Tiped: float
  # electron pedestal top temperature in keV for Ti and Te
  Teped: float
  # pedestal top electron density
  # In units of reference density if neped_is_fGW = False.
  # In Greenwald fraction if neped_is_fGW = True.
  neped: float
  neped_is_fGW: bool
  # Set ped top location.
  Ped_top: float

  # current profiles (broad "Ohmic" + localized "external" currents)
  # peaking factor of prescribed (initial) "Ohmic" current:
  # johm = j0*(1 - r^2/a^2)^nu
  nu: float
  # toggles if "Ohmic" current is treated as total current upon initialization,
  # or if non-inductive current should be included in initial jtot calculation
  initial_j_is_total_current: bool
  # toggles if the initial psi calculation is based on the "nu" current formula,
  # or from the psi available in the numerical geometry file. This setting is
  # ignored for the ad-hoc circular geometry, which has no numerical geometry.
  initial_psi_from_j: bool


@chex.dataclass
class DynamicNumerics:
  """Generic numeric parameters for the simulation."""

  # simulation control
  # start of simulation, in seconds
  t_initial: float
  # end of simulation, in seconds
  t_final: float
  # If True, ensures that if the simulation runs long enough, one step
  # occurs exactly at `t_final`.
  exact_t_final: bool

  # maximum and minimum timesteps allowed in simulation
  maxdt: float  #  only used with chi_time_step_calculator
  mindt: float  #  if adaptive timestep is True, error raised if dt<mindt

  # prefactor in front of chi_timestep_calculator base timestep dt=dx^2/(2*chi).
  # In most use-cases can be increased further above this conservative default
  dtmult: float

  use_fixed_dt: bool  # use fixed_time_step_calculator
  fixed_dt: float  # timestep used for fixed_time_step_calculator
  dt_reduction_factor: float

  # q-profile correction factor. Used only in ad-hoc circular geometry model
  q_correction_factor: float
  # 1/multiplication factor for sigma (conductivity) to reduce current
  # diffusion timescale to be closer to heat diffusion timescale
  resistivity_mult: float

  # density profile info
  # Reference value for normalization
  nref: float

  # numerical (e.g. no. of grid points, other info needed by solver)
  # effective source to dominate PDE in internal boundary condtion location
  # if T != Tped
  largeValue_T: float
  # effective source to dominate density PDE in internal boundary condtion
  # location if n != neped
  largeValue_n: float

  # Enable time-dependent prescribed profiles.
  # This option is provided to allow initialization of density profiles scaled
  # to a Greenwald fraction, and freeze this density even if the current is time
  # evolving. Otherwise the density will evolve to always maintain that GW frac.
  enable_prescribed_profile_evolution: bool


@chex.dataclass(frozen=True)
class StaticConfigSlice:
  """Static arguments to JointStateStepper which cannot be changed.

  If any changes are made to these arguments, then the JointStateStepper must be
  recompiled.

  NOTE: These are not the only parameters which can trigger a recompile! For
  instance, if the geometry changes its shape (i.e. nr or hires_fac change),
  that can also trigger a recompile. This is just to note that this list is not
  an exhaustive list of what can cause recompilations.

  TODO(b/335596447): Add function to help users detect whether their
  change in config will trigger a recompile.
  """

  stepper: stepper_params.StaticRuntimeParams
  # radial grid points (num cells)
  nr: int
  # Solve the ion heat equation (ion temperature evolves over time)
  ion_heat_eq: bool
  # Solve the electron heat equation (electron temperature evolves over time)
  el_heat_eq: bool
  # Solve the current equation (psi evolves over time driven by the solver;
  # q and s evolve over time as a function of psi)
  current_eq: bool
  # Solve the density equation (n evolves over time)
  dens_eq: bool

  # Iterative reduction of dt if nonlinear step does not converge,
  # If nonlinear step does not converge, then the step is redone
  # iteratively at successively lower dt until convergence is reached
  adaptive_dt: bool

# pylint: enable=invalid-name


def build_dynamic_config_slice(
    config: config_lib.Config,
    transport: transport_model_params.RuntimeParams | None = None,
    sources: dict[str, sources_params.RuntimeParams] | None = None,
    stepper: stepper_params.RuntimeParams | None = None,
    t: chex.Numeric | None = None,
) -> DynamicConfigSlice:
  """Builds a DynamicConfigSlice based on the input config."""
  transport = transport or transport_model_params.RuntimeParams()
  sources = sources or {}
  stepper = stepper or stepper_params.RuntimeParams()
  t = config.numerics.t_initial if t is None else t
  # For each dataclass attribute under DynamicConfigSlice, build those objects
  # explicitly, and then for all scalar attributes, fetch their values directly
  # from the input config using config_slice_args.get_init_kwargs.
  return DynamicConfigSlice(
      transport=transport.build_dynamic_params(t),
      stepper=stepper.build_dynamic_params(t),
      sources=_build_dynamic_sources(sources, t),
      plasma_composition=DynamicPlasmaComposition(
          **config_slice_args.get_init_kwargs(
              input_config=config.plasma_composition,
              output_type=DynamicPlasmaComposition,
              t=t,
          )
      ),
      profile_conditions=DynamicProfileConditions(
          **config_slice_args.get_init_kwargs(
              input_config=config.profile_conditions,
              output_type=DynamicProfileConditions,
              t=t,
          )
      ),
      numerics=DynamicNumerics(
          **config_slice_args.get_init_kwargs(
              input_config=config.numerics,
              output_type=DynamicNumerics,
              t=t,
          )
      ),
      **config_slice_args.get_init_kwargs(
          input_config=config,
          output_type=DynamicConfigSlice,
          t=t,
          skip=(
              'transport',
              'stepper',
              'sources',
              'plasma_composition',
              'profile_conditions',
              'numerics',
          ),
      ),
  )


def _build_dynamic_sources(
    sources: dict[str, sources_params.RuntimeParams],
    t: chex.Numeric,
) -> dict[str, sources_params.DynamicRuntimeParams]:
  """Builds a dict of DynamicSourceConfigSlice based on the input config."""
  return {
      source_name: input_source_config.build_dynamic_params(t)
      for source_name, input_source_config in sources.items()
  }


def build_static_config_slice(
    config: config_lib.Config,
    stepper: stepper_params.RuntimeParams | None = None,
) -> StaticConfigSlice:
  """Builds a StaticConfigSlice based on the input config."""
  # t set to None because there shouldnt be time-dependent params in the static
  # config.
  stepper = stepper or stepper_params.RuntimeParams()
  return StaticConfigSlice(
      stepper=stepper.build_static_params(),
      nr=config.numerics.nr,
      ion_heat_eq=config.numerics.ion_heat_eq,
      el_heat_eq=config.numerics.el_heat_eq,
      current_eq=config.numerics.current_eq,
      dens_eq=config.numerics.dens_eq,
      adaptive_dt=config.numerics.adaptive_dt,
  )


class DynamicConfigSliceProvider:
  """Provides a DynamicConfigSlice to use during time t of the sim.

  The DynamicConfigSlice may change from time step to time step, so this class
  interpolates any time-dependent params in the input config to the values they
  should be at time t.

  See `run_simulation()` for how this callable is used.
  """

  def __init__(
      self,
      config: config_lib.Config,
      transport_getter: Callable[[], transport_model_params.RuntimeParams],
      sources_getter: Callable[[], dict[str, sources_params.RuntimeParams]],
      stepper_getter: Callable[[], stepper_params.RuntimeParams],
  ):
    self._input_config = config
    self._transport_runtime_params_getter = transport_getter
    self._sources_getter = sources_getter
    self._stepper_getter = stepper_getter

  def __call__(
      self,
      t: chex.Numeric,
  ) -> DynamicConfigSlice:
    """Returns a DynamicConfigSlice to use during time t of the sim."""
    return build_dynamic_config_slice(
        config=self._input_config,
        transport=self._transport_runtime_params_getter(),
        sources=self._sources_getter(),
        stepper=self._stepper_getter(),
        t=t,
    )

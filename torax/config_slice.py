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
import dataclasses
import types
import typing
from typing import Any, Protocol

import chex
from jax import numpy as jnp
from torax import config as config_lib
from torax import interpolated_param
from torax import jax_utils


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

  transport: DynamicTransportConfigSlice
  solver: DynamicSolverConfigSlice
  plasma_composition: DynamicPlasmaComposition
  profile_conditions: DynamicProfileConditions
  numerics: DynamicNumerics
  sources: Mapping[str, DynamicSourceConfigSlice]

  # density profile info
  # Reference value for normalization
  nref: float

  # external heat source parameters
  w: float  # Gaussian width
  rsource: float  # Source Gaussian central location
  Ptot: float  # total heating
  el_heat_fraction: float  # fraction of heating to electrons (rest are to ions)

  # particle source parameters
  # Gaussian width of pellet deposition [normalized radial coord],
  # (continuous pellet model)
  pellet_width: float
  # Pellet source Gaussian central location [normalized radial coord]
  # (continuous pellet model)
  pellet_deposition_location: float
  # total pellet particles/s (continuous pellet model)
  # TODO(b/323504363): improve numerical strategy, avoid these large numbers
  S_pellet_tot: float

  # exponential decay length of gas puff ionization [normalized radial coord]
  puff_decay_length: float
  # total gas puff particles/s
  # TODO(b/323504363): improve numerical strategy, avoid these large numbers
  S_puff_tot: float

  # NBI particle source Gaussian width in normalized radial coord
  nbi_particle_width: float
  # NBI particle source Gaussian central location in normalized radial coord
  nbi_deposition_location: float
  # NBI total particle source
  S_nbi_tot: float

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
  # toggles if external current is provided absolutely or as a fraction of Ip
  use_absolute_jext: bool
  # total "external" current in MA. Used if use_absolute_jext=True.
  Iext: float
  # total "external" current fraction. Used if use_absolute_jext=False.
  fext: float
  # width of "external" Gaussian current profile
  wext: float
  # normalized radius of "external" Gaussian current profile
  rext: float

  def sanity_check(self):
    """Checks that all parameters are valid."""
    jax_utils.error_if_negative(self.wext, 'wext')

  def __post_init__(self):
    self.sanity_check()


@chex.dataclass(frozen=True)
class DynamicTransportConfigSlice:
  """Input params for the transport model which can be used as compiled args."""

  # Allowed chi and diffusivity bounds
  chimin: float  # minimum chi
  chimax: float  # maximum chi (can be helpful for stability)
  Demin: float  # minimum electron density diffusivity
  Demax: float  # maximum electron density diffusivity
  Vemin: float  # minimum electron density convection
  Vemax: float  # minimum electron density convection

  # set inner core transport coefficients (ad-hoc MHD/EM transport)
  apply_inner_patch: bool
  De_inner: float
  Ve_inner: float
  chii_inner: float
  chie_inner: float
  rho_inner: float  # normalized radius below which patch is applied

  # set outer core transport coefficients.
  # Useful for L-mode near-edge region where QLKNN10D is not applicable.
  # Only used when set_pedestal = False
  apply_outer_patch: bool
  De_outer: float
  Ve_outer: float
  chii_outer: float
  chie_outer: float
  rho_outer: float  # normalized radius above which patch is applied

  # For Critical Gradient Model (CGM)
  # Exponent of chi power law: chi \propto (R/LTi - R/LTi_crit)^alpha
  CGMalpha: float
  # Stiffness parameter
  CGMchistiff: float
  # Ratio of electron to ion transport coefficient (ion higher: ITG)
  CGMchiei_ratio: float
  CGM_D_ratio: float

  # QLKNN model configuration
  # Collisionality multiplier in QLKNN for sensitivity testing.
  # Default is 0.25 (correction factor to a more recent QLK collision operator)
  coll_mult: float
  include_ITG: bool  # to toggle ITG modes on or off
  include_TEM: bool  # to toggle TEM modes on or off
  include_ETG: bool  # to toggle ETG modes on or off
  # The QLK version this specific QLKNN was trained on tends to underpredict
  # ITG electron heat flux in shaped, high-beta scenarios.
  # This is a correction factor
  ITG_flux_ratio_correction: float
  # effective D / effective V approach for particle transport
  DVeff: bool
  # minimum |R/Lne| below which effective V is used instead of effective D
  An_min: float
  # ensure that smag - alpha > -0.2 always, to compensate for no slab modes
  avoid_big_negative_s: bool
  # reduce magnetic shear by 0.5*alpha to capture main impact of alpha
  smag_alpha_correction: bool
  # if q < 1, modify input q and smag as if q~1 as if there are sawteeth
  q_sawtooth_proxy: bool
  # Width of HWHM Gaussian smoothing kernel operating on transport model outputs
  smoothing_sigma: float

  # for constant chi model
  # coefficient in ion heat equation diffusion term in m^2/s
  chii_const: float
  # coefficient in electron heat equation diffusion term in m^2/s
  chie_const: float
  # diffusion coefficient in electron density equation in m^2/s
  De_const: float
  # convection coefficient in electron density equation in m^2/s
  Ve_const: float


@chex.dataclass(frozen=True)
class DynamicSolverConfigSlice:
  """Input params for the solver which can be used as compiled args."""

  # (deliberately) large heat conductivity for Pereverzev rule
  chi_per: float
  # (deliberately) large particle diffusion for Pereverzev rule
  d_per: float
  # Number of corrector steps for the predictor-corrector linear solver.
  # 0 means a pure linear solve with no corrector steps.
  corrector_steps: int
  # log internal iterations in Newton-Raphson solver
  log_iterations: bool


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
  """Perscribed values and boundary conditions for the core profiles."""

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
  # Multiplication factor for bootstrap current
  bootstrap_mult: float
  # multiplier for ion-electron heat exchange term for sensitivity testing
  Qei_mult: float

  # numerical (e.g. no. of grid points, other info needed by solver)
  # effective source to dominate PDE in internal boundary condtion location
  # if T != Tped
  largeValue_T: float
  # effective source to dominate density PDE in internal boundary condtion
  # location if n != neped
  largeValue_n: float


@chex.dataclass(frozen=True)
class DynamicExponentialFormulaConfigSlice:
  """Runtime config for an exponential source profile."""

  # floats to parameterize the formula.
  total: float
  c1: float
  c2: float
  # If True, uses r_norm when calculating the source profiles.
  use_normalized_r: bool


@chex.dataclass(frozen=True)
class DynamicGaussianFormulaConfigSlice:
  # floats to parameterize the formula.
  total: float
  c1: float
  c2: float
  # If True, uses r_norm when calculating the source profiles.
  use_normalized_r: bool


@chex.dataclass(frozen=True)
class DynamicFormulaConfigSlice:
  """Contains all formula configs."""

  exponential: DynamicExponentialFormulaConfigSlice
  gaussian: DynamicGaussianFormulaConfigSlice
  custom_params: dict[str, chex.Numeric]


@chex.dataclass(frozen=True)
class DynamicSourceConfigSlice:
  """Dynamic params for a single TORAX source.

  These params can be changed without triggering a recompile. TORAX sources are
  stateless, so these params are their inputs to determine their output
  profiles.
  """

  # Method to get the source profile. See source_config.py for more info on
  # possible types. This maps to the enum value for the SourceType enum. The
  # enum itself is not JAX-friendly.
  source_type: int
  # If True, this source depends on the mesh state at the start of the time
  # step, or does not depend on the mesh state at all, to compute it's value
  # for the time step. If False, then the source will depend on the "live"
  # state that is updated within the JointStateStepper call.
  is_explicit: bool
  # Parameters used only when the source is using a prescribed formula to
  # compute its profile.
  formula: DynamicFormulaConfigSlice


@chex.dataclass(frozen=True)
class StaticConfigSlice:
  """Static arguments to JointStateStepper which cannot be changed.

  If any changes are made to these arguments, then the JointStateStepper must be
  recompiled.

  NOTE: These are not the only parameters which can trigger a recompile! For
  instance, if the geometry changes its shape (i.e. nr or hires_fac change),
  that can also trigger a recompile. This is just to note that this list is not
  an exhaustive list of what can cause recompilations.

  TODO( b/312726008): Add function to help users detect whether their
  change in config will trigger a recompile.
  """

  solver: StaticSolverConfigSlice
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


@chex.dataclass(frozen=True)
class StaticSolverConfigSlice:
  """Static params for the solver."""

  # Theta for theta-method. 0 is fully explicit, 1 is fully implicit.
  theta_imp: float
  # See `fvm.convection_terms` docstring, `dirichlet_mode` argument
  convection_dirichlet_mode: str
  # See `fvm.convection_terms` docstring, `neumann_mode` argument
  convection_neumann_mode: str
  # use pereverzev terms for linear solver. Is only applied in the nonlinear
  # solver for the optional initial guess from the linear solver
  use_pereverzev: bool
  # Enables predictor_corrector iterations with the linear solver.
  # If False, compilation is faster
  predictor_corrector: bool


# pylint: enable=invalid-name


def build_dynamic_config_slice(
    config: config_lib.Config,
    t: chex.Numeric | None = None,
) -> DynamicConfigSlice:
  """Builds a DynamicConfigSlice based on the input config."""
  t = config.numerics.t_initial if t is None else t
  # For each dataclass attribute under DynamicConfigSlice, build those objects
  # explicitly, and then for all scalar attributes, fetch their values directly
  # from the input config using _get_init_kwargs.
  return DynamicConfigSlice(
      transport=DynamicTransportConfigSlice(
          **_get_init_kwargs(
              input_config=config.transport,
              output_type=DynamicTransportConfigSlice,
              t=t,
          )
      ),
      solver=DynamicSolverConfigSlice(
          **_get_init_kwargs(
              input_config=config.solver,
              output_type=DynamicSolverConfigSlice,
              t=t,
          )
      ),
      sources=_build_dynamic_sources(config, t),
      plasma_composition=DynamicPlasmaComposition(
          **_get_init_kwargs(
              input_config=config.plasma_composition,
              output_type=DynamicPlasmaComposition,
              t=t,
          )
      ),
      profile_conditions=DynamicProfileConditions(
          **_get_init_kwargs(
              input_config=config.profile_conditions,
              output_type=DynamicProfileConditions,
              t=t,
          )
      ),
      numerics=DynamicNumerics(
          **_get_init_kwargs(
              input_config=config.numerics,
              output_type=DynamicNumerics,
              t=t,
          )
      ),
      **_get_init_kwargs(
          input_config=config,
          output_type=DynamicConfigSlice,
          t=t,
          skip=(
              'transport',
              'solver',
              'sources',
              'plasma_composition',
              'profile_conditions',
              'numerics',
          ),
      ),
  )


def _build_dynamic_sources(
    config: config_lib.Config,
    t: chex.Numeric,
) -> dict[str, DynamicSourceConfigSlice]:
  """Builds a dict of DynamicSourceConfigSlice based on the input config."""
  source_configs = {}
  for source_name, input_source_config in config.sources.items():
    source_configs[source_name] = DynamicSourceConfigSlice(
        source_type=input_source_config.source_type.value,
        is_explicit=input_source_config.is_explicit,
        formula=DynamicFormulaConfigSlice(
            exponential=DynamicExponentialFormulaConfigSlice(
                **_get_init_kwargs(
                    input_config=input_source_config.formula.exponential,
                    output_type=DynamicExponentialFormulaConfigSlice,
                    t=t,
                )
            ),
            gaussian=DynamicGaussianFormulaConfigSlice(
                **_get_init_kwargs(
                    input_config=input_source_config.formula.gaussian,
                    output_type=DynamicGaussianFormulaConfigSlice,
                    t=t,
                )
            ),
            custom_params={
                key: _interpolate_param(value, t)
                for key, value in input_source_config.formula.custom_params.items()
            },
        ),
    )
  return source_configs


def build_static_config_slice(config: config_lib.Config) -> StaticConfigSlice:
  """Builds a StaticConfigSlice based on the input config."""
  # t set to None because there shouldnt be time-dependent params in the static
  # config.
  return StaticConfigSlice(
      solver=StaticSolverConfigSlice(
          **_get_init_kwargs(config.solver, StaticSolverConfigSlice, t=None)
      ),
      nr=config.numerics.nr,
      ion_heat_eq=config.numerics.ion_heat_eq,
      el_heat_eq=config.numerics.el_heat_eq,
      current_eq=config.numerics.current_eq,
      dens_eq=config.numerics.dens_eq,
      adaptive_dt=config.numerics.adaptive_dt,
  )


def _input_is_a_float(
    field_name: str, input_config_fields_to_types: dict[str, Any]
) -> bool:
  try:
    return field_name in input_config_fields_to_types and issubclass(
        input_config_fields_to_types[field_name], float
    )
  except:  # pylint: disable=bare-except
    # issubclass does not play nicely with generics, but if a type is a
    # generic at this stage, it is not a float.
    return False


def _input_is_an_interpolated_param(
    field_name: str,
    input_config_fields_to_types: dict[str, Any],
) -> bool:
  """Returns True if the input config field is an InterpolatedParam."""
  if field_name not in input_config_fields_to_types:
    return False

  def _check(ft):
    """Checks if the input field type is an InterpolatedParam."""
    try:
      return (
          # If the type comes as a string rather than an object, the Union check
          # below won't work, so we check for the full name here.
          ft == 'InterpParamOrInterpParamInput'
          or
          # Common alias for InterpParamOrInterpParamInput.
          ft == 'TimeDependentField'
          or
          # Otherwise, only check if it is actually the InterpolatedParam.
          ft == 'interpolated_param.InterpolatedParam'
          or issubclass(ft, interpolated_param.InterpolatedParamBase)
      )
    except:  # pylint: disable=bare-except
      # issubclass does not play nicely with generics, but if a type is a
      # generic at this stage, it is not an InterpolatedParam.
      return False

  field_type = input_config_fields_to_types[field_name]
  if isinstance(field_type, types.UnionType):
    # Look at all the args of the union and see if any match properly
    for arg in typing.get_args(field_type):
      if _check(arg):
        return True
  else:
    return _check(field_type)


def _interpolate_param(
    param_or_param_input: interpolated_param.InterpParamOrInterpParamInput,
    t: chex.Numeric,
) -> jnp.ndarray:
  if not isinstance(param_or_param_input, interpolated_param.InterpolatedParam):
    # The param is a InterpolatedParamInput, so we need to convert it to an
    # InterpolatedParam first.
    param_or_param_input = interpolated_param.InterpolatedParam(
        value=param_or_param_input,
    )
  return param_or_param_input.get_value(t)


def _get_init_kwargs(
    input_config: ...,
    output_type: ...,
    t: chex.Numeric | None = None,
    skip: tuple[str, ...] = (),
) -> dict[str, Any]:
  """Builds init() kwargs based on the input config for all non-dict fields."""
  kwargs = {}
  input_config_fields_to_types = {
      field.name: field.type for field in dataclasses.fields(input_config)
  }
  for field in dataclasses.fields(output_type):
    if field.name in skip:
      continue
    if not hasattr(input_config, field.name):
      raise ValueError(f'Missing field {field.name}')
    config_val = getattr(input_config, field.name)
    # If the input config type is an InterpolatedParam, we need to interpolate
    # it at time t to populate the correct values in the output config.
    # dataclass fields can either be the actual type OR the string name of the
    # type. Check for both.
    if _input_is_an_interpolated_param(
        field.name, input_config_fields_to_types
    ):
      if t is None:
        raise ValueError('t must be specified for interpolated params')
      config_val = _interpolate_param(config_val, t)
    elif _input_is_a_float(field.name, input_config_fields_to_types):
      config_val = float(config_val)
    kwargs[field.name] = config_val
  return kwargs


class DynamicConfigSliceProvider(Protocol):
  """Provides a DynamicConfigSlice to use during one time step of the sim.

  The DynamicConfigSlice may change from time step to time step, so the
  simulator needs to know which DynamicConfigSlice to feed to the
  SimulationStepCallable. See `run_simulation()` for how this callable is
  used.

  This class is a typing.Protocol, meaning any class or function that implements
  this API can be used as an argument for functions that require a
  DynamicConfigSliceProvider (it uses the Python concept of "duck-typing").
  """

  def __call__(
      self,
      t: chex.Numeric,
  ) -> DynamicConfigSlice:
    """Returns a DynamicConfigSlice to use during time t of the sim."""


class TimeDependentDynamicConfigSliceProvider(DynamicConfigSliceProvider):
  """Provides a DynamicConfigSlice to use during time t of the sim.

  Interpolates any time-dependent params in the input config to the values they
  should be at time t.
  """

  def __init__(
      self,
      config: config_lib.Config,
  ):
    self._input_config = config

  def __call__(
      self,
      t: chex.Numeric,
  ) -> DynamicConfigSlice:
    return build_dynamic_config_slice(self._input_config, t)

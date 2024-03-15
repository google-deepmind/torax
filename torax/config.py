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

"""Definition of configuration dataclass.

Specifies parameter names and default values for all physics and solver
parameters.
"""

from collections.abc import Iterable, Mapping
import dataclasses
import enum
import typing
from typing import Any
import chex
from torax import interpolated_param
from torax.sources import source_config


# Type-alias for clarity. While the InterpolatedParams can vary across any
# field, in Config, we mainly use it to handle time-dependent parameters.
TimeDependentField = interpolated_param.InterpParamOrInterpParamInput
# Type-alias for brevity. Helps users only import this module.
InterpolationMode = interpolated_param.InterpolationMode
InterpolationParam = interpolated_param.InterpolatedParam


def _check_config_param_in_set(
    param_name: str,
    param_value: Any,
    valid_values: Iterable[Any],
) -> None:
  if param_value not in valid_values:
    raise ValueError(
        f'{param_name} invalid. Must give {" or ".join(valid_values)}. '
        f'Provided: {param_value}.'
    )


@chex.dataclass
class SolverConfig:
  """Configuration parameters for the differential equation solver."""
  # theta value in the theta method.
  # 0 = explicit, 1 = fully implicit, 0.5 = Crank-Nicolson
  theta_imp: float = 1.0
  # Enables predictor_corrector iterations with the linear solver.
  # If False, compilation is faster
  predictor_corrector: bool = True
  # Number of corrector steps for the predictor-corrector linear solver.
  # 0 means a pure linear solve with no corrector steps.
  corrector_steps: int = 1
  # See `fvm.convection_terms` docstring, `dirichlet_mode` argument
  convection_dirichlet_mode: str = 'ghost'
  # See `fvm.convection_terms` docstring, `neumann_mode` argument
  convection_neumann_mode: str = 'ghost'
  # use pereverzev terms for linear solver. Is only applied in the nonlinear
  # solver for the optional initial guess from the linear solver
  use_pereverzev: bool = False
  # (deliberately) large heat conductivity for Pereverzev rule
  chi_per: float = 20.0
  # (deliberately) large particle diffusion for Pereverzev rule
  d_per: float = 10.0
  # log internal iterations in Newton-Raphson solver
  log_iterations: bool = False

  def __post_init__(self):
    assert self.theta_imp >= 0.0
    assert self.theta_imp <= 1.0
    assert self.corrector_steps >= 0
    _check_config_param_in_set(
        'convection_dirichlet_mode',
        self.convection_dirichlet_mode,
        ['ghost', 'direct', 'semi-implicit'],
    )
    _check_config_param_in_set(
        'convection_neumann_mode',
        self.convection_neumann_mode,
        ['ghost', 'semi-implicit'],
    )


# pylint: disable=invalid-name
@chex.dataclass
class TransportConfig:
  """Configuration parameters for the turbulent transport model."""

  transport_model: str = 'constant'  # 'constant', 'CGM', or 'qlknn'

  # Allowed chi and diffusivity bounds
  chimin: float = 0.05  # minimum chi
  chimax: float = 100.0  # maximum chi (can be helpful for stability)
  Demin: float = 0.05  # minimum electron density diffusivity
  Demax: float = 100.0  # maximum electron density diffusivity
  Vemin: float = -50.0  # minimum electron density convection
  Vemax: float = 50.0  # minimum electron density convection

  # set inner core transport coefficients (ad-hoc MHD/EM transport)
  apply_inner_patch: bool = False
  De_inner: float = 0.2
  Ve_inner: float = 0.0
  chii_inner: float = 1.0
  chie_inner: float = 1.0
  rho_inner: float = 0.3  # normalized radius below which patch is applied

  # set outer core transport coefficients.
  # Useful for L-mode near-edge region where QLKNN10D is not applicable.
  # Only used when set_pedestal = False
  apply_outer_patch: bool = False
  De_outer: float = 0.2
  Ve_outer: float = 0.0
  chii_outer: float = 1.0
  chie_outer: float = 1.0
  rho_outer: float = 0.9  # normalized radius above which patch is applied

  # For Critical Gradient Model (CGM)
  # Exponent of chi power law: chi \propto (R/LTi - R/LTi_crit)^alpha
  CGMalpha: float = 2.0
  # Stiffness parameter
  CGMchistiff: float = 2.0
  # Ratio of electron to ion transport coefficient (ion higher: ITG)
  CGMchiei_ratio: float = 2.0
  CGM_D_ratio: float = 5.0

  # QLKNN model configuration
  # Collisionality multiplier in QLKNN for sensitivity testing.
  # Default is 0.25 (correction factor to a more recent QLK collision operator)
  coll_mult: float = 0.25
  include_ITG: bool = True  # to toggle ITG modes on or off
  include_TEM: bool = True  # to toggle TEM modes on or off
  include_ETG: bool = True  # to toggle ETG modes on or off
  # The QLK version this specific QLKNN was trained on tends to underpredict
  # ITG electron heat flux in shaped, high-beta scenarios.
  # This is a correction factor
  ITG_flux_ratio_correction: float = 2.0
  # effective D / effective V approach for particle transport
  DVeff: bool = False
  # minimum |R/Lne| below which effective V is used instead of effective D
  An_min: float = 0.05
  # ensure that smag - alpha > -0.2 always, to compensate for no slab modes
  avoid_big_negative_s: bool = True
  # reduce magnetic shear by 0.5*alpha to capture main impact of alpha
  smag_alpha_correction: bool = True
  # if q < 1, modify input q and smag as if q~1 as if there are sawteeth
  q_sawtooth_proxy: bool = True
  # Width of HWHM Gaussian smoothing kernel operating on transport model outputs
  smoothing_sigma: float = 0.0

  # for constant chi model
  # coefficient in ion heat equation diffusion term in m^2/s
  chii_const: float = 1.0
  # coefficient in electron heat equation diffusion term in m^2/s
  chie_const: float = 1.0
  # diffusion coefficient in electron density equation in m^2/s
  De_const: float = 1.0
  # convection coefficient in electron density equation in m^2/s
  Ve_const: float = -0.33

  def __post_init__(self):
    assert self.De_const >= 0.0
    assert self.chii_const >= 0.0
    assert self.chie_const >= 0.0
    assert self.CGM_D_ratio >= 0.0
    assert self.chimin >= 0.0
    assert self.chimax >= 0.0 and self.chimax > self.chimin
    assert self.Demin >= 0.0 and self.Demax > self.Demin
    assert self.Vemax > self.Vemin
    assert self.De_inner >= 0.0
    assert self.De_inner >= 0.0
    assert self.chii_inner >= 0.0
    assert self.chie_inner >= 0.0
    assert self.rho_inner >= 0.0 and self.rho_inner <= 1.0
    assert self.rho_outer >= 0.0 and self.rho_outer <= 1.0
    assert self.rho_outer > self.rho_inner
    _check_config_param_in_set(
        'transport_model', self.transport_model, ['qlknn', 'CGM', 'constant']
    )


# NOMUTANTS -- It's expected for the tests to pass with different defaults.
@chex.dataclass
class Config:
  """Configuration parameters for the `torax` module."""

  # physical inputs
  # major radius (R) in meters
  Rmaj: TimeDependentField = 6.2
  # minor radius (a) in meters
  Rmin: TimeDependentField = 2.0
  # amu of main ion (if multiple isotope, make average)
  Ai: float = 2.5
  # charge of main ion
  Zi: float = 1.0
  # total plasma current in MA
  # Note that if Ip_from_parameters=False in geometry, then this Ip will be
  # overwritten by values from the geometry data
  Ip: TimeDependentField = 15.0
  # Toroidal magnetic field on axis [T]
  B0: TimeDependentField = 5.3
  # needed for qlknn and fusion power
  Zeff: TimeDependentField = 1.0
  Zimp: TimeDependentField = 10.0  # impurity charge state assumed for dilution

  # density profile info
  # Reference value for normalization
  nref: float = 1e20
  # line averaged density (not used if f_GW = True)
  # in units of reference density
  nbar: float = 1.0
  # set initial condition density according to Greenwald fraction.
  # Otherwise from nbar
  set_fGW: bool = True
  # Initial condition Greenwald fraction (nGW = Ip/(pi*a^2))
  # with a in m, nGW in 10^20 m-3, Ip in MA
  fGW: float = 0.85
  # Peaking factor of density profile
  npeak: float = 1.5

  # temperature boundary conditions
  # initial condition ion temperature for r=0
  Ti_bound_left: float = 15.0
  # boundary condition ion temperature for r=Rmin
  Ti_bound_right: TimeDependentField = 1.0
  # initial condition electron temperature for r=0
  Te_bound_left: float = 15.0
  # boundary condition electron temperature for r=Rmin
  Te_bound_right: TimeDependentField = 1.0
  # density boundary condition for r=Rmin, units of nref
  ne_bound_right: TimeDependentField = 0.5

  # external heat source parameters
  w: TimeDependentField = 0.25  # Gaussian width in normalized radial coordinate
  # Source Gaussian central location (in normalized r)
  rsource: TimeDependentField = 0.0
  Ptot: TimeDependentField = 120e6  # total heating
  el_heat_fraction: TimeDependentField = 0.66666  # electron heating fraction
  # multiplier for ion-electron heat exchange term for sensitivity testing
  Qei_mult: float = 1.0

  # particle source parameters
  # Gaussian width of pellet deposition [normalized radial coord],
  # (continuous pellet model)
  pellet_width: TimeDependentField = 0.1
  # Pellet source Gaussian central location [normalized radial coord]
  # (continuous pellet model)
  pellet_deposition_location: TimeDependentField = 0.85
  # total pellet particles/s (continuous pellet model)
  # TODO(b/323504363): improve numerical strategy, avoid these large numbers
  S_pellet_tot: TimeDependentField = 2e22

  # exponential decay length of gas puff ionization [normalized radial coord]
  puff_decay_length: TimeDependentField = 0.05
  # total gas puff particles/s
  # TODO(b/323504363): improve numerical strategy, avoid these large numbers
  S_puff_tot: TimeDependentField = 1e22

  # NBI particle source Gaussian width in normalized radial coord
  nbi_particle_width: TimeDependentField = 0.25
  # NBI particle source Gaussian central location in normalized radial coord
  nbi_deposition_location: TimeDependentField = 0.0
  # NBI total particle source
  S_nbi_tot: TimeDependentField = 1e22

  # current profiles (broad "Ohmic" + localized "external" currents)
  # peaking factor of "Ohmic" current: johm = j0*(1 - r^2/a^2)^nu
  nu: float = 3.0
  # total "external" current fraction
  fext: TimeDependentField = 0.2
  # width of "external" Gaussian current profile
  wext: TimeDependentField = 0.05
  # normalized radius of "external" Gaussian current profile
  rext: TimeDependentField = 0.4
  # q-profile correction factor. Used only in ad-hoc circular geometry model
  q_correction_factor: float = 1.38
  # 1/multiplication factor for sigma (conductivity) to reduce current
  # diffusion timescale to be closer to heat diffusion timescale
  resistivity_mult: TimeDependentField = 100.0
  # Multiplication factor for bootstrap current
  bootstrap_mult: float = 1.0

  # numerical (e.g. no. of grid points, other info needed by solver)
  # radial grid points (num cells)
  nr: int = 25

  # maximum and minimum timesteps allowed in simulation
  maxdt: float = 1e-1  #  only used with chi_time_step_calculator
  mindt: float = 1e-8  #  if adaptive timestep is True, error raised if dt<mindt

  # prefactor in front of chi_timestep_calculator base timestep dt=dx^2/(2*chi).
  # In most use-cases can be increased further above this conservative default
  dtmult: float = 0.9 * 10

  use_fixed_dt: bool = False  # use fixed_time_step_calculator
  fixed_dt: float = 1e-2  # timestep used for fixed_time_step_calculator

  # Iterative reduction of dt if nonlinear step does not converge,
  # If nonlinear step does not converge, then the step is redone
  # iteratively at successively lower dt until convergence is reached
  adaptive_dt: bool = True
  dt_reduction_factor: float = 3

  # simulation control
  # start of simulation, in seconds
  t_initial: float = 0.0
  # end of simulation, in seconds
  t_final: float = 5.0
  # If True, ensures that if the simulation runs long enough, one step
  # occurs exactly at `t_final`.
  exact_t_final: bool = False

  # Internal boundary condition (pedestal)
  # Do not set internal boundary condition if this is False
  set_pedestal: TimeDependentField = True
  # ion pedestal top temperature in keV for Ti and Te
  Tiped: TimeDependentField = 5.0
  # electron pedestal top temperature in keV for Ti and Te
  Teped: TimeDependentField = 5.0
  # pedestal top electron density in units of nref
  neped: TimeDependentField = 0.7
  # Set ped top location.
  Ped_top: TimeDependentField = 0.91
  # effective source to dominate PDE in internal boundary condtion location
  # if T != Tped
  largeValue_T: float = 1.0e10
  # effective source to dominate density PDE in internal boundary condtion
  # location if n != neped
  largeValue_n: float = 1.0e8

  # solver parameters
  solver: SolverConfig = dataclasses.field(default_factory=SolverConfig)

  # Solve the ion heat equation (ion temperature evolves over time)
  ion_heat_eq: bool = True
  # Solve the electron heat equation (electron temperature evolves over time)
  el_heat_eq: bool = True
  # Solve the current equation (psi evolves over time driven by the solver;
  # q and s evolve over time as a function of psi)
  current_eq: bool = False
  # Solve the density equation (n evolves over time)
  dens_eq: bool = False

  # 'File directory where the simulation outputs will be saved. If not '
  # 'provided, this will default to /tmp/torax_results_<YYYYMMDD_HHMMSS>/.',
  output_dir: str | None = None

  # pylint: enable=invalid-name

  # Transport parameters.
  transport: TransportConfig = dataclasses.field(
      default_factory=TransportConfig
  )

  # Runtime configs for all source/sink terms.
  # Note that the sources field is overridden in the __post_init__. See impl for
  # details on how this field is updated.
  sources: Mapping[str, source_config.SourceConfig] = dataclasses.field(
      default_factory=source_config.get_default_sources_config
  )

  def sanity_check(self) -> None:
    """Checks that various configuration parameters are valid."""
    # TODO do more extensive config parameter sanity checking

    # These are floats, not jax types, so we can use direct asserts.
    assert self.dtmult > 0.0
    assert isinstance(self.transport, TransportConfig)
    assert isinstance(self.solver, SolverConfig)
    if (
        not self.set_pedestal
        and self.transport.apply_outer_patch
        and self.solver.convection_neumann_mode != 'ghost'
        and self.solver.convection_dirichlet_mode != 'ghost'
    ):
      raise ValueError(
          'To avoid numerical instability use ghost convection modes'
      )

  def __post_init__(self):
    # The sources config should have the default values from
    # source_config.get_default_sources_config. The additional values provided
    # via the config constructor should OVERRIDE these defaults.
    sources = dict(source_config.get_default_sources_config())
    sources.update(self.sources)  # Update with the user inputs.
    self.sources = sources
    self.sanity_check()


def recursive_replace(obj: ..., **changes) -> ...:
  """Recursive version of `dataclasses.replace`.

  This allows updating of nested dataclasses.
  Assumes all dict-valued keys in `changes` are themselves changes to apply
  to fields of obj.

  Args:
    obj: Any dataclass instance.
    **changes: Dict of updates to apply to fields of `obj`.

  Returns:
    A copy of `obj` with the changes applied.
  """

  flattened_changes = {}
  if dataclasses.is_dataclass(obj):
    keys_to_types = {
        field.name: field.type for field in dataclasses.fields(obj)
    }
  else:
    # obj is another dict-like object that does not have typed fields.
    keys_to_types = None
  for key, value in changes.items():
    if isinstance(value, dict):
      if dataclasses.is_dataclass(getattr(obj, key)):
        # If obj[key] is another dataclass, recurse and populate that dataclass
        # with the input changes.
        flattened_changes[key] = recursive_replace(getattr(obj, key), **value)
      elif keys_to_types is not None:
        # obj[key] is likely just a dict, and each key needs to be treated
        # separately.
        # In order to support this, there needs to be some added type
        # information for what the values of the dict should be.
        typing_args = typing.get_args(keys_to_types[key])
        if len(typing_args) == 2:  # the keys type, the values type.
          inner_dict = {}
          value_type = typing_args[1]
          for inner_key, inner_value in value.items():
            if dataclasses.is_dataclass(value_type):
              inner_dict[inner_key] = recursive_replace(
                  value_type(), **inner_value
              )
            else:
              inner_dict[inner_key] = value_type(inner_value)
          flattened_changes[key] = inner_dict
        else:
          # If we don't have additional type information, just try using the
          # value as is.
          flattened_changes[key] = value
      else:
        # keys_to_types is None, so again, we don't have additional information.
        flattened_changes[key] = value
    else:
      # For any value that should be an enum value but is not an enum already
      # (could come a YAML file for instance and might be a string or int),
      # this converts that value to an enum.
      try:
        if (
            # if obj is a dataclass
            keys_to_types is not None
            and
            # and this param should be an enum
            issubclass(keys_to_types[key], enum.Enum)
            and
            # but it is not already one.
            not isinstance(value, enum.Enum)
        ):
          if isinstance(value, str):
            value = keys_to_types[key][value.upper()]
          else:
            value = keys_to_types[key](value)
      except TypeError:
        # Ignore these errors. issubclass doesn't work with typing.Optional
        # types. Note that this means that optional enum fields might not be
        # cast properly, so avoid these when defining configs.
        pass
      flattened_changes[key] = value
  return dataclasses.replace(obj, **flattened_changes)

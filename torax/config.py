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

from collections.abc import Iterable
import dataclasses
import enum
import typing
from typing import Any
import chex
from torax import interpolated_param


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
class PlasmaComposition:
  # amu of main ion (if multiple isotope, make average)
  Ai: float = 2.5
  # charge of main ion
  Zi: float = 1.0
  # needed for qlknn and fusion power
  Zeff: TimeDependentField = 1.0
  Zimp: TimeDependentField = 10.0  # impurity charge state assumed for dilution


@chex.dataclass
class ProfileConditions:
  """Prescribed values and boundary conditions for the core profiles."""

  # total plasma current in MA
  # Note that if Ip_from_parameters=False in geometry, then this Ip will be
  # overwritten by values from the geometry data
  Ip: TimeDependentField = 15.0

  # Temperature boundary conditions at r=Rmin
  Ti_bound_right: TimeDependentField = 1.0
  Te_bound_right: TimeDependentField = 1.0
  # Prescribed values for r=0. When evolving, then is initial condition.
  Te_bound_left: TimeDependentField = 15.0
  Ti_bound_left: TimeDependentField = 15.0

  # Peaking factor of density profile.
  # If density evolves with PDE (dens_eq=True), then is initial condition
  npeak: TimeDependentField = 1.5

  # Initial line averaged density.
  # In units of reference density if nbar_is_fGW = False.
  # In Greenwald fraction if nbar_is_fGW = True.
  # nGW = Ip/(pi*a^2) with a in m, nGW in 10^20 m-3, Ip in MA
  nbar: TimeDependentField = 0.85
  # Toggle units of nbar
  nbar_is_fGW: bool = True

  # Density boundary condition for r=Rmin.
  # In units of reference density if ne_bound_right_is_fGW = False.
  # In Greenwald fraction if ne_bound_right_is_fGW = True.
  ne_bound_right: TimeDependentField = 0.5
  ne_bound_right_is_fGW: bool = False

  # Internal boundary condition (pedestal)
  # Do not set internal boundary condition if this is False
  set_pedestal: TimeDependentField = True
  # ion pedestal top temperature in keV
  Tiped: TimeDependentField = 5.0
  # electron pedestal top temperature in keV
  Teped: TimeDependentField = 5.0
  # pedestal top electron density
  # In units of reference density if neped_is_fGW = False.
  # In Greenwald fraction if neped_is_fGW = True.
  neped: TimeDependentField = 0.7
  neped_is_fGW: bool = False
  # Set ped top location.
  Ped_top: TimeDependentField = 0.91


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

  use_fixed_dt: bool = False  # use fixed_time_step_calculator
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
  q_correction_factor: float = 1.38
  # 1/multiplication factor for sigma (conductivity) to reduce current
  # diffusion timescale to be closer to heat diffusion timescale
  resistivity_mult: TimeDependentField = 1.0

  # density profile info
  # Reference value for normalization
  nref: float = 1e20

  # numerical (e.g. no. of grid points, other info needed by solver)
  # radial grid points (num cells)
  nr: int = 25  # TODO(b/330172917): Move this to geometry.
  # effective source to dominate PDE in internal boundary condtion location
  # if T != Tped
  largeValue_T: float = 1.0e10
  # effective source to dominate density PDE in internal boundary condtion
  # location if n != neped
  largeValue_n: float = 1.0e8


# NOMUTANTS -- It's expected for the tests to pass with different defaults.
@chex.dataclass
class Config:
  """Configuration parameters for the `torax` module."""

  plasma_composition: PlasmaComposition = dataclasses.field(
      default_factory=PlasmaComposition
  )
  profile_conditions: ProfileConditions = dataclasses.field(
      default_factory=ProfileConditions
  )
  numerics: Numerics = dataclasses.field(default_factory=Numerics)

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

  # solver parameters
  solver: SolverConfig = dataclasses.field(default_factory=SolverConfig)

  # 'File directory where the simulation outputs will be saved. If not '
  # 'provided, this will default to /tmp/torax_results_<YYYYMMDD_HHMMSS>/.',
  output_dir: str | None = None

  # pylint: enable=invalid-name

  def sanity_check(self) -> None:
    """Checks that various configuration parameters are valid."""
    # TODO(b/330172917) do more extensive config parameter sanity checking

    # These are floats, not jax types, so we can use direct asserts.
    assert self.numerics.dtmult > 0.0
    assert isinstance(self.solver, SolverConfig)
    assert isinstance(self.plasma_composition, PlasmaComposition)
    assert isinstance(self.numerics, Numerics)

  def __post_init__(self):
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

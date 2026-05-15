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

"""Profile condition parameters used throughout TORAX simulations."""

import dataclasses
import enum
import logging
from typing import Annotated, Callable, Final, Sequence

import chex
import jax
import numpy as np
import pydantic
from torax._src import array_typing
from torax._src.fvm import cell_variable
from torax._src.internal_boundary_conditions import internal_boundary_conditions as internal_boundary_conditions_lib
from torax._src.physics import fast_ion as fast_ion_lib
from torax._src.torax_pydantic import torax_pydantic
from typing_extensions import Self

# pylint: disable=invalid-name

# Order of magnitude validations to catch common config errors.
_MIN_IP_AMPS: Final[float] = 1e3
_MIN_DENSITY_M3: Final[float] = 1e10
_MAX_DENSITY_GW: Final[float] = 1e2
_MAX_TEMPERATURE_KEV: Final[float] = 1e3
_MAX_TEMPERATURE_BC_KEV: Final[float] = 5e1


class PrescribedFastIon(torax_pydantic.BaseModelFrozen):
  """User-facing config for prescribed fast ion density and temperature.

  This is the Pydantic configuration class that users interact with. Fields
  support time-varying interpolation (e.g. ``{0.0: 1e19, 1.0: 2e19}``).
  At simulation time, these are evaluated into concrete arrays via
  ``PrescribedFastIonData`` for use in JAX-compiled functions.

  Attributes:
    source: Source name (e.g. 'icrh').
    species: Species name (e.g. 'He3'). Must be one of
      ``fast_ion_lib.FAST_ION_SPECIES``.
    n: Prescribed density profile [m^-3].
    n_right_bc: Right boundary condition for density [m^-3].
    T: Prescribed temperature profile [keV].
    T_right_bc: Right boundary condition for temperature [keV].
  """

  source: Annotated[str, torax_pydantic.JAX_STATIC]
  species: Annotated[str, torax_pydantic.JAX_STATIC]
  n: torax_pydantic.TimeVaryingArray
  n_right_bc: torax_pydantic.TimeVaryingScalar
  T: torax_pydantic.TimeVaryingArray
  T_right_bc: torax_pydantic.TimeVaryingScalar


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PrescribedFastIonData:
  """Evaluated prescribed fast ion data for a single species at time t.

  This is the JAX-compatible runtime counterpart of ``PrescribedFastIon``.
  It holds concrete array values evaluated at a specific time ``t``, and is
  stored on ``RuntimeParams`` for use inside JIT-compiled simulation steps.

  Attributes:
    source: Source name (e.g. 'icrh').
    species: Species name (e.g. 'He3').
    n: Prescribed density profile [m^-3].
    n_right_bc: Right boundary condition for density [m^-3].
    T: Prescribed temperature profile [keV].
    T_right_bc: Right boundary condition for temperature [keV].
  """

  source: str = dataclasses.field(metadata={'static': True})
  species: str = dataclasses.field(metadata={'static': True})
  n: array_typing.FloatVector
  n_right_bc: array_typing.FloatScalar
  T: array_typing.FloatVector
  T_right_bc: array_typing.FloatScalar


class InitialPsiMode(enum.StrEnum):
  """How to calculate the initial psi value."""

  PROFILE_CONDITIONS = 'profile_conditions'
  GEOMETRY = 'geometry'
  J = 'j'


class NeBoundaryConditionMode(enum.StrEnum):
  """Mode for the electron density right boundary condition.

  Attributes:
    PRESCRIBED: The boundary condition is prescribed directly via `n_e_right_bc`
      or taken from the `n_e` profile at rho_norm=1.
    DENSITY_FRACTION: The boundary condition is computed as `n_e(reference_rho,
      t) * multiplier`, where `reference_rho` and `multiplier` are
      user-specified. t is the time at the beginning of each time step interval.
  """

  PRESCRIBED = 'prescribed'
  DENSITY_FRACTION = 'density_fraction'


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class RuntimeParams:
  """Prescribed values and boundary conditions for the core profiles."""

  Ip: array_typing.FloatScalar
  v_loop_lcfs: array_typing.FloatScalar
  T_i_right_bc: array_typing.FloatScalar
  T_e_right_bc: array_typing.FloatScalar
  # Temperature profiles defined on the cell grid.
  T_e: array_typing.FloatVector
  T_i: array_typing.FloatVector
  # If provided as array, Psi profile defined on the cell grid.
  psi: array_typing.FloatVector | None
  psidot: array_typing.FloatVector | None
  toroidal_angular_velocity: array_typing.FloatVector | None
  toroidal_angular_velocity_right_bc: array_typing.FloatScalar | None
  # Electron density profile on the cell grid.
  n_e: array_typing.FloatVector
  nbar: array_typing.FloatScalar
  n_e_nbar_is_fGW: bool
  n_e_right_bc: array_typing.FloatScalar
  n_e_right_bc_is_fGW: bool
  n_e_right_bc_mode: NeBoundaryConditionMode = dataclasses.field(
      metadata={'static': True}
  )
  n_e_right_bc_reference_rho: array_typing.FloatScalar | None
  n_e_right_bc_multiplier: array_typing.FloatScalar | None
  internal_boundary_conditions: (
      internal_boundary_conditions_lib.InternalBoundaryConditions
  )
  current_profile_nu: float
  initial_j_is_total_current: bool = dataclasses.field(
      metadata={'static': True}
  )
  initial_psi_from_j: bool = dataclasses.field(metadata={'static': True})
  normalize_n_e_to_nbar: bool = dataclasses.field(metadata={'static': True})
  use_v_loop_lcfs_boundary_condition: bool = dataclasses.field(
      metadata={'static': True}
  )
  n_e_right_bc_is_absolute: bool = dataclasses.field(metadata={'static': True})
  initial_psi_mode: InitialPsiMode = dataclasses.field(
      metadata={'static': True}
  )
  prescribed_fast_ions: tuple[PrescribedFastIonData, ...] = ()


class ProfileConditions(torax_pydantic.BaseModelFrozen):
  """Generic numeric parameters for the simulation.

  The `from_dict(...)` method can accept a dictionary defined by
  https://torax.readthedocs.io/en/latest/configuration.html#profile-conditions.

  Attributes:
    Ip: Total plasma current in A. Note that if Ip_from_parameters=False in
      geometry, then this Ip will be overwritten by values from the geometry
      data. If use_v_loop_lcfs_boundary_condition, only used as an initial
      condition.
    use_v_loop_lcfs_boundary_condition: Boundary condition at LCFS for Vloop ( =
      dspsi_lcfs/dt ). If use_v_loop_lcfs_boundary_condition is True, then the
      specified Vloop at the LCFS is used as the boundary condition for the psi
      equation; otherwise, Ip is used as the boundary condition.
    v_loop_lcfs: Boundary condition at LCFS for Vloop ( = dpsi_lcfs/dt ).
    T_i_right_bc: Temperature boundary conditions at r=a_minor. If this is
      `None` the boundary condition will instead be taken from `T_i` and `T_e`
      at rhon=1.
    T_e_right_bc: Temperature boundary conditions at r=a_minor. If this is
      `None` the boundary condition will instead be taken from `T_i` and `T_e`
      at rhon=1.
    T_i: Prescribed or evolving values for temperature at different times.
    T_e: Prescribed or evolving values for temperature at different times.
    psi: Initial values for psi. If provided, the initial psi will be taken from
      here. Otherwise, the initial psi will be calculated from either the
      geometry or the "current_profile_nu formula" dependant on the
      `initial_psi_from_j` field.
    psidot: Prescribed values for the time derivative of poloidal flux (loop
      voltage). If provided, and if `evolve_current` is False, this prescribed
      `psidot` will be used instead of the internally calculated one. This is
      useful for cases where an unphysical transient `psidot` from the initial
      `psi` condition needs to be overridden.
    toroidal_angular_velocity: Prescribed or evolving values for toroidal
      angular velocity. If None, toroidal_angular_velocity will be initialized
      to zero.
    toroidal_angular_velocity_right_bc: Toroidal velocity boundary condition for
      r=a_minor. If this is `None` the boundary condition will instead be taken
      from `toroidal_angular_velocity` at rhon=1. If `toroidal_angular_velocity
      is also `None`, then the boundary condition will be set to zero.
    n_e: Prescribed or evolving values for electron density at different times.
    normalize_n_e_to_nbar: Whether to renormalize the density profile to have
      the desired line averaged density `nbar`.
    nbar: Line averaged density. In units of m^-3 if n_e_nbar_is_fGW = False. In
      Greenwald fraction if n_e_nbar_is_fGW = True. nGW = Ip/(pi*a^2) with a in
      m, nGW in 10^20 m-3, Ip in MA
    n_e_nbar_is_fGW: Toggle units of nbar
    n_e_right_bc: Density boundary condition for r=a_minor. In units of m^-3 if
      n_e_right_bc_is_fGW = False. In Greenwald fraction if `n_e_right_bc_is_fGW
      = True`. If `n_e_right_bc` is `None` then the boundary condition will
      instead be taken from `n_e` at rho_norm=1. In this case,
      `n_e_right_bc_is_absolute` in the StaticRuntimeParams will be set to
      `False` and n_e_right_bc_is_fGW` will be set to `n_e_nbar_is_fGW`. If
      `n_e_right_bc` is not `None` then `n_e_right_bc_is_absolute` will be set
      to `True`.
    n_e_right_bc_is_fGW: Toggle units of n_e_right_bc.
    internal_boundary_conditions: Config for internal boundary conditions. See
      `InternalBoundaryConditionsConfig` for more details.
    current_profile_nu: Peaking factor of "Ohmic" current: j_ohmic = j0*(1 -
      r^2/a^2)^current_profile_nu
    initial_j_is_total_current: Toggles if "Ohmic" current is treated as total
      current upon initialization, or if non-inductive current should be
      included in initial j_total calculation.
    initial_psi_from_j: Toggles if the initial psi calculation is based on the
      "current_profile_nu" current formula, or from the psi available in the
      numerical geometry file. This setting is ignored for the ad-hoc circular
      geometry, which has no numerical geometry.
    initial_psi_mode: Mode of the initial psi value. If `PROFILE_CONDITIONS` is
      used, then the initial psi is taken from the `psi` attribute. If
      `GEOMETRY` is used, then the initial psi is taken from the geometry. If
      `J` is used, then the psi calculation is based on the `current_profile_nu`
      current formula. For now if `PROFILE_CONDITIONS` is used, but `psi` is not
      provided then the `initial_psi_from_j` setting will be used to match the
      legacy behavior.
  """

  Ip: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(15e6)
  use_v_loop_lcfs_boundary_condition: Annotated[
      bool, torax_pydantic.JAX_STATIC
  ] = False
  v_loop_lcfs: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.0)
  )
  T_i_right_bc: torax_pydantic.PositiveTimeVaryingScalar | None = None
  T_e_right_bc: torax_pydantic.PositiveTimeVaryingScalar | None = None
  T_i: torax_pydantic.PositiveTimeVaryingArray = (
      torax_pydantic.ValidatedDefault({0: {0: 15.0, 1: 1.0}})
  )
  T_e: torax_pydantic.PositiveTimeVaryingArray = (
      torax_pydantic.ValidatedDefault({0: {0: 15.0, 1: 1.0}})
  )
  psi: torax_pydantic.TimeVaryingArray | None = None
  psidot: torax_pydantic.TimeVaryingArray | None = None
  toroidal_angular_velocity: torax_pydantic.TimeVaryingArray | None = None
  toroidal_angular_velocity_right_bc: (
      torax_pydantic.TimeVaryingScalar | None
  ) = None
  n_e: torax_pydantic.PositiveTimeVaryingArray = (
      torax_pydantic.ValidatedDefault({0: {0: 1.2e20, 1: 0.8e20}})
  )
  normalize_n_e_to_nbar: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  nbar: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      0.85e20
  )
  n_e_nbar_is_fGW: bool = False
  n_e_right_bc: torax_pydantic.TimeVaryingScalar | None = None
  n_e_right_bc_is_fGW: bool = False
  # TODO(b/495851657): V2 API: move boundary condition attributes to a
  # dedicated nested object, and merge with attributes currently in edge model,
  # e.g. update_temperatures.
  n_e_right_bc_mode: Annotated[
      NeBoundaryConditionMode, torax_pydantic.JAX_STATIC
  ] = NeBoundaryConditionMode.PRESCRIBED
  n_e_right_bc_reference_rho: torax_pydantic.TimeVaryingScalar | None = None
  n_e_right_bc_multiplier: torax_pydantic.TimeVaryingScalar | None = None
  internal_boundary_conditions: (
      internal_boundary_conditions_lib.InternalBoundaryConditionsConfig
  ) = torax_pydantic.ValidatedDefault(
      internal_boundary_conditions_lib.InternalBoundaryConditionsConfig()
  )
  current_profile_nu: float = 1.0
  initial_j_is_total_current: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  # TODO(b/434175938): Remove this before the V2 API release in place of
  # initial_psi_source.
  initial_psi_from_j: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  initial_psi_mode: Annotated[InitialPsiMode, torax_pydantic.JAX_STATIC] = (
      InitialPsiMode.PROFILE_CONDITIONS
  )
  fast_ions: list[PrescribedFastIon] | None = None

  @pydantic.model_validator(mode='after')
  def _validate_fast_ion_species(self) -> Self:
    """Validates that prescribed fast ion species are known."""
    if self.fast_ions is not None:
      for pfi in self.fast_ions:
        if pfi.species not in fast_ion_lib.FAST_ION_SPECIES:
          raise ValueError(
              f'Prescribed fast ion species {pfi.species!r} for source'
              f' {pfi.source!r} is not a supported species. Supported'
              f' species: {fast_ion_lib.FAST_ION_SPECIES}'
          )
    return self

  @pydantic.model_validator(mode='after')
  def after_validator(self) -> Self:
    error_messages = []

    def _sanity_check_profile_boundary_conditions(
        values,
        value_name,
        errors_list,
    ):
      """Check that the profile is defined at rho=1.0 for various cases."""
      if not values.right_boundary_conditions_defined:
        error_message = (
            f'As no right boundary condition was set for {value_name}, the'
            f' profile for {value_name} must include a rho=1.0 boundary'
            ' condition.'
        )
        errors_list.append(error_message)

    if self.T_i_right_bc is None:
      _sanity_check_profile_boundary_conditions(self.T_i, 'T_i', error_messages)
    if self.T_e_right_bc is None:
      _sanity_check_profile_boundary_conditions(self.T_e, 'T_e', error_messages)
    if (
        self.n_e_right_bc is None
        and self.n_e_right_bc_mode == NeBoundaryConditionMode.PRESCRIBED
    ):
      _sanity_check_profile_boundary_conditions(self.n_e, 'n_e', error_messages)

    # Validate density_fraction mode attributes.
    if self.n_e_right_bc_mode == NeBoundaryConditionMode.DENSITY_FRACTION:
      if self.n_e_right_bc_reference_rho is None:
        error_messages.append(
            'n_e_right_bc_reference_rho must be set when'
            ' n_e_right_bc_mode is "density_fraction".'
        )
      if self.n_e_right_bc_multiplier is None:
        error_messages.append(
            'n_e_right_bc_multiplier must be set when'
            ' n_e_right_bc_mode is "density_fraction".'
        )
      if self.n_e_right_bc_is_fGW:
        error_messages.append(
            'n_e_right_bc_is_fGW must be False when n_e_right_bc_mode is'
            ' "density_fraction". The density_fraction mode computes an'
            ' absolute boundary condition in SI units (m^-3).'
        )
      if self.n_e_right_bc is not None:
        logging.warning(
            'n_e_right_bc is set but will be ignored because'
            ' n_e_right_bc_mode is "density_fraction".'
        )
    elif self.n_e_right_bc_mode == NeBoundaryConditionMode.PRESCRIBED:
      if self.n_e_right_bc_reference_rho is not None:
        logging.warning(
            'n_e_right_bc_reference_rho is set but will be ignored because'
            ' n_e_right_bc_mode is "prescribed".'
        )
      if self.n_e_right_bc_multiplier is not None:
        logging.warning(
            'n_e_right_bc_multiplier is set but will be ignored because'
            ' n_e_right_bc_mode is "prescribed".'
        )

    # Validate plasma current input order of magnitude.
    if np.any(self.Ip.value < _MIN_IP_AMPS):
      failed_val, time_at_fail = _get_first_failing_value_and_time_or_rho(
          self.Ip.value, self.Ip.time, lambda x: x < _MIN_IP_AMPS
      )
      error_messages.append(
          f'Plasma current Ip at time {time_at_fail}s is {failed_val:.2e} A,'
          f' which is below the minimum threshold of {_MIN_IP_AMPS:.0e} A.'
          ' Possible cause: erroneous input Ip in MA instead of A.'
      )

    # Validate plasma density inputs order of magnitude.
    if not self.n_e_nbar_is_fGW and self.normalize_n_e_to_nbar:
      if np.any(self.nbar.value < _MIN_DENSITY_M3):
        failed_val, time_at_fail = _get_first_failing_value_and_time_or_rho(
            self.nbar.value, self.nbar.time, lambda x: x < _MIN_DENSITY_M3
        )
        error_messages.append(
            'Density inputs set in m^-3 units. Line-averaged density nbar at'
            f' time {time_at_fail}s is {failed_val:.2e} m^-3  which is below'
            f' the minimum threshold of {_MIN_DENSITY_M3:.0e} m^-3. Possible'
            ' cause: erroneous input nbar in normalized units instead of m^-3.'
        )

    if self.n_e_nbar_is_fGW and self.normalize_n_e_to_nbar:
      if np.any(self.nbar.value > _MAX_DENSITY_GW):
        failed_val, time_at_fail = _get_first_failing_value_and_time_or_rho(
            self.nbar.value, self.nbar.time, lambda x: x > _MAX_DENSITY_GW
        )
        error_messages.append(
            'Density inputs set in Greenwald fraction. Line-averaged density'
            f' nbar time {time_at_fail}s is {failed_val:.2e} fGW which is'
            f' above the maximum threshold of {_MAX_DENSITY_GW:.0e} fGW.'
            ' Possible cause: erroneous input nbar in m^-3 instead of fGW.'
        )

    if not self.n_e_nbar_is_fGW and not self.normalize_n_e_to_nbar:
      for n_e_time, n_e_value in self.n_e.value.items():
        if np.any(n_e_value[1] < _MIN_DENSITY_M3):
          failed_val, rho_norm_at_fail = (
              _get_first_failing_value_and_time_or_rho(
                  n_e_value[1], n_e_value[0], lambda x: x < _MIN_DENSITY_M3
              )
          )
          error_messages.append(
              f'Density inputs set in m^-3 units. n_e at time {n_e_time}s and'
              f' rho_norm {rho_norm_at_fail} is {failed_val:.2e} m^-3  which is'
              f' below the minimum threshold of {_MIN_DENSITY_M3:.0e} m^-3.'
              ' Possible cause: erroneous input n_e in normalized units instead'
              ' of m^-3.'
          )
        break

    if self.n_e_nbar_is_fGW and not self.normalize_n_e_to_nbar:
      for n_e_time, n_e_value in self.n_e.value.items():
        if np.any(n_e_value[1] > _MAX_DENSITY_GW):
          (
              failed_val,
              rho_norm_at_fail,
          ) = _get_first_failing_value_and_time_or_rho(
              n_e_value[1], n_e_value[0], lambda x: x > _MAX_DENSITY_GW
          )
          error_messages.append(
              'Density inputs set in Greenwald fraction. n_e at time'
              f' {n_e_time}s and rho_norm {rho_norm_at_fail} is'
              f' {failed_val:.2e} m^-3  which is above the maximum threshold of'
              f' {_MAX_DENSITY_GW:.0e} fGW. Possible cause: erroneous input n_e'
              ' in m^-3 instead of fGW.'
          )
          break

    if self.n_e_right_bc is not None:
      if not self.n_e_right_bc_is_fGW:
        if np.any(self.n_e_right_bc.value < _MIN_DENSITY_M3):
          failed_val, time_at_fail = _get_first_failing_value_and_time_or_rho(
              self.n_e_right_bc.value,
              self.n_e_right_bc.time,
              lambda x: x < _MIN_DENSITY_M3,
          )
          error_messages.append(
              'Density boundary condition inputs set in m^-3 units,'
              f' n_e_right_bc at time {time_at_fail}s is {failed_val:.2e} m^-3'
              f' which is below the minimum threshold of {_MIN_DENSITY_M3:.0e}'
              ' m^-3. Possible cause: erroneous input n_e_right_bc in'
              ' normalized units instead of m^-3.'
          )
      else:
        if np.any(self.n_e_right_bc.value > _MAX_DENSITY_GW):
          failed_val, time_at_fail = _get_first_failing_value_and_time_or_rho(
              self.n_e_right_bc.value,
              self.n_e_right_bc.time,
              lambda x: x > _MAX_DENSITY_GW,
          )
          error_messages.append(
              'Density boundary condition inputs set in Greenwald fraction,'
              f' n_e_right_bc at time {time_at_fail}s is {failed_val:.2e} fGW'
              f' which is above the maximum threshold of {_MAX_DENSITY_GW:.0e}'
              ' fGW. Possible cause: erroneous input n_e_right_bc in m^-3'
              ' instead of fGW.'
          )

    # Validate temperature inputs order of magnitude.
    for T_e_time, T_e_value in self.T_e.value.items():
      if np.any(T_e_value[1] > _MAX_TEMPERATURE_KEV):
        (
            failed_val,
            rho_norm_at_fail,
        ) = _get_first_failing_value_and_time_or_rho(
            T_e_value[1], T_e_value[0], lambda x: x > _MAX_TEMPERATURE_KEV
        )
        error_messages.append(
            f'T_e at time {T_e_time}s and rho_norm {rho_norm_at_fail} is'
            f' {failed_val:.2e} which is above the maximum threshold of'
            f' {_MAX_TEMPERATURE_KEV:.0e} keV. Possible cause: erroneous'
            ' input T_e in eV instead of keV.'
        )
        break

    for T_i_time, T_i_value in self.T_i.value.items():
      if np.any(T_i_value[1] > _MAX_TEMPERATURE_KEV):
        (
            failed_val,
            rho_norm_at_fail,
        ) = _get_first_failing_value_and_time_or_rho(
            T_i_value[1], T_i_value[0], lambda x: x > _MAX_TEMPERATURE_KEV
        )
        error_messages.append(
            f'T_i at time {T_i_time}s and rho_norm {rho_norm_at_fail} is'
            f' {failed_val:.2e} which is above the maximum threshold of'
            f' {_MAX_TEMPERATURE_KEV:.0e} keV. Possible cause: erroneous'
            ' input T_i in eV instead of keV.'
        )
        break

    if self.T_e_right_bc is not None:
      if np.any(self.T_e_right_bc.value > _MAX_TEMPERATURE_BC_KEV):
        failed_val, time_at_fail = _get_first_failing_value_and_time_or_rho(
            self.T_e_right_bc.value,
            self.T_e_right_bc.time,
            lambda x: x > _MAX_TEMPERATURE_BC_KEV,
        )
        error_messages.append(
            f'T_e_right_bc at time {time_at_fail}s is {failed_val:.2e} keV'
            ' which is above the maximum threshold of'
            f' {_MAX_TEMPERATURE_BC_KEV:.0e} keV. Possible cause: erroneous'
            ' input T_e_right_bc in eV instead of keV.'
        )

    if self.T_i_right_bc is not None:
      if np.any(self.T_i_right_bc.value > _MAX_TEMPERATURE_BC_KEV):
        failed_val, time_at_fail = _get_first_failing_value_and_time_or_rho(
            self.T_i_right_bc.value,
            self.T_i_right_bc.time,
            lambda x: x > _MAX_TEMPERATURE_BC_KEV,
        )
        error_messages.append(
            f'T_i_right_bc at time {time_at_fail}s is {failed_val:.2e} keV'
            ' which is above the maximum threshold of'
            f' {_MAX_TEMPERATURE_BC_KEV:.0e} keV. Possible cause: erroneous'
            ' input T_i_right_bc in eV instead of keV.'
        )

    if error_messages:
      error_message_preamble = (
          f'{len(error_messages)} errors were found in profile conditions'
          ' config:\n\n'
      )
      final_error_message = '\n\n'.join(error_messages)
      raise ValueError(error_message_preamble + final_error_message)

    return self

  def build_runtime_params(self, t: chex.Numeric) -> RuntimeParams:
    """Builds a RuntimeParams object for time t."""

    runtime_params = {
        x.name: getattr(self, x.name)
        for x in dataclasses.fields(RuntimeParams)
        if x.name
        not in (
            'n_e_right_bc_is_absolute',
            'prescribed_fast_ions',
            'internal_boundary_conditions',
        )
    }

    if self.T_e_right_bc is None:
      runtime_params['T_e_right_bc'] = self.T_e.get_value(
          t, grid_type='face_right'
      )

    if self.T_i_right_bc is None:
      runtime_params['T_i_right_bc'] = self.T_i.get_value(
          t, grid_type='face_right'
      )

    if self.toroidal_angular_velocity_right_bc is None:
      if self.toroidal_angular_velocity is None:
        runtime_params['toroidal_angular_velocity_right_bc'] = 0.0
      else:
        runtime_params['toroidal_angular_velocity_right_bc'] = (
            self.toroidal_angular_velocity.get_value(t, grid_type='face_right')
        )

    if self.n_e_right_bc_mode == NeBoundaryConditionMode.DENSITY_FRACTION:
      # In density_fraction mode, the actual n_e_right_bc value is computed
      # later in build_runtime_params.py. Set a placeholder here and mark
      # it as absolute (SI units, not Greenwald fraction).
      runtime_params['n_e_right_bc'] = 1e19
      runtime_params['n_e_right_bc_is_absolute'] = True
    elif self.n_e_right_bc is None:
      runtime_params['n_e_right_bc'] = self.n_e.get_value(
          t, grid_type='face_right'
      )
      runtime_params['n_e_right_bc_is_fGW'] = self.n_e_nbar_is_fGW
      runtime_params['n_e_right_bc_is_absolute'] = False
    else:
      runtime_params['n_e_right_bc_is_absolute'] = True

    # Evaluate prescribed fast ions at time t.
    prescribed_fast_ions = _build_prescribed_fast_ions(self.fast_ions, t)
    runtime_params['prescribed_fast_ions'] = prescribed_fast_ions

    # Evaluate internal boundary conditions at time t. Required as IBCs are a
    # nested class within the config.
    runtime_params['internal_boundary_conditions'] = (
        self.internal_boundary_conditions.build_runtime_params(t)
    )

    def _get_value(x):
      if isinstance(
          x,
          (
              torax_pydantic.TimeVaryingScalar,
              torax_pydantic.TimeVaryingArray,
          ),
      ):
        return x.get_value(t)
      else:
        return x

    runtime_params = {k: _get_value(v) for k, v in runtime_params.items()}
    return RuntimeParams(**runtime_params)


def _build_prescribed_fast_ions(
    fast_ions: list[PrescribedFastIon] | None,
    t: chex.Numeric,
) -> tuple[PrescribedFastIonData, ...]:
  """Evaluates prescribed fast ion configs at time t.

  Args:
    fast_ions: List of prescribed fast ion configs.
    t: Time at which to evaluate the time-varying fields.

  Returns:
    Tuple of PrescribedFastIonData evaluated at time t.
  """
  if fast_ions is None:
    return ()
  return tuple(
      PrescribedFastIonData(
          source=pfi.source,
          species=pfi.species,
          n=pfi.n.get_value(t),
          n_right_bc=pfi.n_right_bc.get_value(t),
          T=pfi.T.get_value(t),
          T_right_bc=pfi.T_right_bc.get_value(t),
      )
      for pfi in fast_ions
  )


def apply_prescribed_fast_ions(
    fast_ions: Sequence[fast_ion_lib.FastIon],
    prescribed: tuple[PrescribedFastIonData, ...],
    face_centers: jax.Array,
) -> tuple[fast_ion_lib.FastIon, ...]:
  """Overrides matching fast ions with prescribed data.

  For each prescribed fast ion, if a matching (source, species) pair exists in
  the fast_ions list, it is replaced with the prescribed values. Prescribed
  entries that do not match any existing fast ion are ignored.

  Args:
    fast_ions: Existing fast ion objects (from sources or initialization).
    prescribed: Prescribed fast ion data from profile_conditions.
    face_centers: Face-grid coordinates for building CellVariables.

  Returns:
    Updated tuple of FastIon objects with prescribed overrides applied.
  """
  if not prescribed:
    return tuple(fast_ions)

  prescribed_by_key = {(p.source, p.species): p for p in prescribed}
  result = []
  for fi in fast_ions:
    key = (fi.source, fi.species)
    if key in prescribed_by_key:
      p = prescribed_by_key[key]
      result.append(
          fast_ion_lib.FastIon(
              species=fi.species,
              source=fi.source,
              n=cell_variable.CellVariable(
                  value=p.n,
                  face_centers=face_centers,
                  right_face_grad_constraint=None,
                  right_face_constraint=p.n_right_bc,
              ),
              T=cell_variable.CellVariable(
                  value=p.T,
                  face_centers=face_centers,
                  right_face_grad_constraint=None,
                  right_face_constraint=p.T_right_bc,
              ),
          )
      )
    else:
      result.append(fi)
  return tuple(result)


def _get_first_failing_value_and_time_or_rho(
    values: np.ndarray,
    time_or_rho: np.ndarray,
    comparator: Callable[[np.ndarray], np.ndarray],
) -> tuple[float, float]:
  """Returns the first failing value and time or rho for a given comparator.

  If values came from a TimeVaryingScalar, then time_or_rho will be the time.
  If values came from a TimeVaryingArray, then time_or_rho will be the rho.

  Args:
    values: The values to check.
    time_or_rho: The time or rho associated with the values.
    comparator: A function that takes an array of values and returns an array of
      bools indicating whether each value passes the check.

  Returns:
    A tuple of (first_failing_value, time_or_rho)
  """
  fail_indices = np.where(comparator(values))[0]
  if fail_indices.size == 0:
    raise ValueError(
        '_get_first_failing_time was called with no failing values.'
    )
  return values[fail_indices[0]], time_or_rho[fail_indices[0]]

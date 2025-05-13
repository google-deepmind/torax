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

import chex
import pydantic
from torax import array_typing
from torax.torax_pydantic import torax_pydantic
from typing_extensions import Self
# pylint: disable=invalid-name


@chex.dataclass
class DynamicProfileConditions:
  """Prescribed values and boundary conditions for the core profiles."""

  Ip: array_typing.ScalarFloat
  vloop_lcfs: array_typing.ScalarFloat
  T_i_right_bc: array_typing.ScalarFloat
  T_e_right_bc: array_typing.ScalarFloat
  # Temperature profiles defined on the cell grid.
  T_e: array_typing.ArrayFloat
  T_i: array_typing.ArrayFloat
  # If provided as array, Psi profile defined on the cell grid.
  psi: array_typing.ArrayFloat | None
  # Electron density profile on the cell grid.
  n_e: array_typing.ArrayFloat
  normalize_n_e_to_nbar: bool
  nbar: array_typing.ScalarFloat
  n_e_nbar_is_fGW: bool
  n_e_right_bc: array_typing.ScalarFloat
  n_e_right_bc_is_fGW: bool
  current_profile_nu: float
  initial_j_is_total_current: bool
  initial_psi_from_j: bool


@chex.dataclass(frozen=True)
class StaticRuntimeParams:
  """Static params for profile conditions."""

  use_vloop_lcfs_boundary_condition: bool
  normalize_n_e_to_nbar: bool
  # Whether to use absolute ne_bound_right or ne[-1] for setting BC.
  # Set by ne_bound_right condition.
  n_e_right_bc_is_absolute: bool


class ProfileConditions(torax_pydantic.BaseModelFrozen):
  """Generic numeric parameters for the simulation.

  The `from_dict(...)` method can accept a dictionary defined by
  https://torax.readthedocs.io/en/latest/configuration.html#profile-conditions.

  Attributes:
    Ip: Total plasma current in MA. Note that if Ip_from_parameters=False in
      geometry, then this Ip will be overwritten by values from the geometry
      data. If use_vloop_lcfs_boundary_condition, only used as an initial
      condition.
    use_vloop_lcfs_boundary_condition: Boundary condition at LCFS for Vloop ( =
      dspsi_lcfs/dt ). If use_vloop_lcfs_boundary_condition is True, then the
      specfied Vloop at the LCFS is used as the boundary condition for the psi
      equation; otherwise, Ip is used as the boundary condition.
    vloop_lcfs: Boundary condition at LCFS for Vloop ( = dpsi_lcfs/dt ).
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
    n_e: Prescribed or evolving values for electron density at different times.
    normalize_n_e_to_nbar: Whether to renormalize the density profile to have
      the desired line averaged density `nbar`.
    nbar: Line averaged density. In units of reference density if
      n_e_nbar_is_fGW = False. In Greenwald fraction if n_e_nbar_is_fGW = True.
      nGW = Ip/(pi*a^2) with a in m, nGW in 10^20 m-3, Ip in MA
    n_e_nbar_is_fGW: Toggle units of nbar
    n_e_right_bc: Density boundary condition for r=a_minor. In units of
      reference density if n_e_right_bc_is_fGW = False. In Greenwald fraction if
      `n_e_right_bc_is_fGW = True`. If `n_e_right_bc` is `None` then the
      boundary condition will instead be taken from `n_e` at rhon=1. In this
      case, `n_e_right_bc_is_absolute` in the StaticRuntimeParams will be set to
      `False` and n_e_right_bc_is_fGW` will be set to `n_e_nbar_is_fGW`. If
      `n_e_right_bc` is not `None` then `n_e_right_bc_is_absolute` will be set
      to `True`.
    n_e_right_bc_is_fGW: Toggle units of n_e_right_bc.
    current_profile_nu: Peaking factor of "Ohmic" current: j_ohmic = j0*(1 -
      r^2/a^2)^current_profile_nu
    initial_j_is_total_current: Toggles if "Ohmic" current is treated as total
      current upon initialization, or if non-inductive current should be
      included in initial j_total calculation.
    initial_psi_from_j: Toggles if the initial psi calculation is based on the
      "current_profile_nu" current formula, or from the psi available in the
      numerical geometry file. This setting is ignored for the ad-hoc circular
      geometry, which has no numerical geometry.
  """

  Ip: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(15.0)
  use_vloop_lcfs_boundary_condition: bool = False
  vloop_lcfs: torax_pydantic.TimeVaryingScalar = (
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
  n_e: torax_pydantic.PositiveTimeVaryingArray = (
      torax_pydantic.ValidatedDefault(
          {0: {0: 1.2, 1: 0.8}}
      )
  )
  normalize_n_e_to_nbar: bool = False
  nbar: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(0.85)
  n_e_nbar_is_fGW: bool = False
  n_e_right_bc: torax_pydantic.TimeVaryingScalar | None = None
  n_e_right_bc_is_fGW: bool = False
  set_pedestal: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(True)
  )
  current_profile_nu: float = 3.0
  initial_j_is_total_current: bool = False
  initial_psi_from_j: bool = False

  @pydantic.model_validator(mode='after')
  def after_validator(self) -> Self:

    def _sanity_check_profile_boundary_conditions(
        values,
        value_name,
    ):
      """Check that the profile is defined at rho=1.0 for various cases."""
      error_message = (
          f'As no right boundary condition was set for {value_name}, the'
          f' profile for {value_name} must include a rho=1.0 boundary'
          ' condition.'
      )
      if not values.right_boundary_conditions_defined:
        raise ValueError(error_message)

    if self.T_i_right_bc is None:
      _sanity_check_profile_boundary_conditions(self.T_i, 'T_i')
    if self.T_e_right_bc is None:
      _sanity_check_profile_boundary_conditions(self.T_e, 'T_e')
    if self.n_e_right_bc is None:
      _sanity_check_profile_boundary_conditions(self.n_e, 'n_e')
    return self

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> DynamicProfileConditions:
    """Builds a DynamicProfileConditions."""

    dynamic_params = {
        x.name: getattr(self, x.name)
        for x in dataclasses.fields(DynamicProfileConditions)
    }

    if self.T_e_right_bc is None:
      dynamic_params['T_e_right_bc'] = self.T_e.get_value(
          t, grid_type='face_right'
      )

    if self.T_i_right_bc is None:
      dynamic_params['T_i_right_bc'] = self.T_i.get_value(
          t, grid_type='face_right'
      )

    if self.n_e_right_bc is None:
      dynamic_params['n_e_right_bc'] = self.n_e.get_value(
          t, grid_type='face_right'
      )
      dynamic_params['n_e_right_bc_is_fGW'] = self.n_e_nbar_is_fGW

    def _get_value(x):
      if isinstance(
          x, (torax_pydantic.TimeVaryingScalar, torax_pydantic.TimeVaryingArray)
      ):
        return x.get_value(t)
      else:
        return x

    dynamic_params = {k: _get_value(v) for k, v in dynamic_params.items()}
    return DynamicProfileConditions(**dynamic_params)

  def build_static_params(self) -> StaticRuntimeParams:
    """Builds static runtime params from the config."""
    return StaticRuntimeParams(
        use_vloop_lcfs_boundary_condition=self.use_vloop_lcfs_boundary_condition,
        normalize_n_e_to_nbar=self.normalize_n_e_to_nbar,
        n_e_right_bc_is_absolute=False if self.n_e_right_bc is None else True,
    )

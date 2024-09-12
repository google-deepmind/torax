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

from __future__ import annotations

import dataclasses
import logging

import chex
from torax import array_typing
from torax import geometry
from torax import interpolated_param
from torax.config import base
from torax.config import config_args
from typing_extensions import override
import xarray as xr


# pylint: disable=invalid-name
@chex.dataclass
class ProfileConditions(
    base.RuntimeParametersConfig['ProfileConditionsProvider']
):
  """Prescribed values and boundary conditions for the core profiles."""

  # total plasma current in MA
  # Note that if Ip_from_parameters=False in geometry, then this Ip will be
  # overwritten by values from the geometry data
  Ip: interpolated_param.TimeInterpolated = 15.0

  # Temperature boundary conditions at r=Rmin. If this is `None` the boundary
  # condition will instead be taken from `Ti` and `Te` at rhon=1.
  Ti_bound_right: interpolated_param.TimeInterpolated | None = (
      None
  )
  Te_bound_right: interpolated_param.TimeInterpolated | None = (
      None
  )
  # Prescribed or evolving values for temperature at different times.
  Ti: interpolated_param.InterpolatedVarTimeRhoInput = dataclasses.field(
      default_factory=lambda: {0: {0: 15.0, 1: 1.0}}
  )
  Te: interpolated_param.InterpolatedVarTimeRhoInput = dataclasses.field(
      default_factory=lambda: {0: {0: 15.0, 1: 1.0}}
  )

  # Initial values for psi. If provided, the initial psi will be taken from
  # here. Otherwise, the initial psi will be calculated from either the geometry
  # or the "nu formula" dependant on the `initial_psi_from_j` field.
  psi: interpolated_param.InterpolatedVarTimeRhoInput | None = None

  # Prescribed or evolving values for electron density at different times.
  ne: interpolated_param.InterpolatedVarTimeRhoInput = dataclasses.field(
      default_factory=lambda: {0: {0: 1.5, 1: 1.0}}
  )
  # Whether to renormalize the density profile to have the desired line averaged
  # density `nbar`.
  normalize_to_nbar: bool = True

  # Line averaged density.
  # In units of reference density if ne_is_fGW = False.
  # In Greenwald fraction if ne_is_fGW = True.
  # nGW = Ip/(pi*a^2) with a in m, nGW in 10^20 m-3, Ip in MA
  nbar: interpolated_param.TimeInterpolated = 0.85
  # Toggle units of nbar
  ne_is_fGW: bool = True

  # Density boundary condition for r=Rmin.
  # In units of reference density if ne_bound_right_is_fGW = False.
  # In Greenwald fraction if ne_bound_right_is_fGW = True.
  # If `ne_bound_right` is `None` then the boundary condition will instead be
  # taken from `ne` at rhon=1. In this case, `ne_bound_right_is_absolute` will
  # be set to `False` and `ne_bound_right_is_fGW` will be set to `ne_is_fGW`.
  # If `ne_bound_right` is not `None` then `ne_bound_right_is_absolute` will be
  # set to `True`.
  ne_bound_right: interpolated_param.TimeInterpolated | None = (
      None
  )
  ne_bound_right_is_fGW: bool = False
  ne_bound_right_is_absolute: bool = False

  # Internal boundary condition (pedestal)
  # Do not set internal boundary condition if this is False
  set_pedestal: interpolated_param.TimeInterpolated = True
  # ion pedestal top temperature in keV
  Tiped: interpolated_param.TimeInterpolated = 5.0
  # electron pedestal top temperature in keV
  Teped: interpolated_param.TimeInterpolated = 5.0
  # pedestal top electron density
  # In units of reference density if neped_is_fGW = False.
  # In Greenwald fraction if neped_is_fGW = True.
  neped: interpolated_param.TimeInterpolated = 0.7
  neped_is_fGW: bool = False
  # Set ped top location.
  Ped_top: interpolated_param.TimeInterpolated = 0.91

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

  def _sanity_check_profile_boundary_conditions(
      self,
      values: interpolated_param.InterpolatedVarTimeRhoInput,
      value_name: str,
  ):
    """Check that the profile is defined at rho=1.0 for various cases."""
    error_message = (
        f'As no right boundary condition was set for {value_name}, the'
        f' profile for {value_name} must include a rho=1.0 boundary'
        ' condition.'
    )
    match values:
      # In case of constant profile case expect an explicit boundary condition.
      case float():
        raise ValueError(error_message)
      # In case of constant profile case expect an explicit boundary condition.
      case int():
        raise ValueError(error_message)
      case dict():
        # Initial condition dict shortcut.
        if all(isinstance(v, float) for v in values.values()):
          if 1.0 not in values:
            raise ValueError(error_message)
        else:
          # Check for all times that the boundary condition is defined.
          for _, value in values.items():
            if 1.0 not in value:
              raise ValueError(error_message)
      case xr.DataArray():
        if 1.0 not in values.coords[config_args.RHO_NORM]:
          raise ValueError(error_message)
      # Arrays case.
      case _:
        if not isinstance(values, tuple):
          raise ValueError(f'Cannot identify a valid way to map {values}.')
        # Initial condition array shortcut.
        if len(values) == 2:
          rho_norm, _ = values
        elif len(values) == 3:
          _, rho_norm, _ = values
        else:
          raise ValueError('Only array tuples of length 2 or 3 are supported.')
        if 1.0 not in rho_norm:
          raise ValueError(error_message)

  def __post_init__(self):
    if self.Ti_bound_right is None:
      self._sanity_check_profile_boundary_conditions(
          self.Ti,
          'Ti',
      )
    if self.Te_bound_right is None:
      self._sanity_check_profile_boundary_conditions(
          self.Te,
          'Te',
      )
    if self.ne_bound_right is None:
      self._sanity_check_profile_boundary_conditions(
          self.ne,
          'ne',
      )

  @override
  def make_provider(
      self, torax_mesh: geometry.Grid1D | None = None,
  ) -> ProfileConditionsProvider:
    provider_kwargs = self.get_provider_kwargs(torax_mesh)
    if torax_mesh is None:
      raise ValueError('torax_mesh is required for ProfileConditionsProvider.')

    # Overrides for profile conditions provider.
    if self.Te_bound_right is None:
      logging.info('Setting electron temperature boundary condition using Te.')
      Te_bound_right = config_args.get_interpolated_var_2d(
          self.Te, torax_mesh.face_centers[-1]
      )
      provider_kwargs['Te_bound_right'] = Te_bound_right

    if self.Ti_bound_right is None:
      logging.info('Setting ion temperature boundary condition using Ti.')
      Ti_bound_right = config_args.get_interpolated_var_2d(
          self.Ti, torax_mesh.face_centers[-1]
      )
      provider_kwargs['Ti_bound_right'] = Ti_bound_right

    if self.ne_bound_right is None:
      logging.info('Setting electron density boundary condition using ne.')
      ne_bound_right = config_args.get_interpolated_var_2d(
          self.ne, torax_mesh.face_centers[-1]
      )
      self.ne_bound_right_is_absolute = False
      self.ne_bound_right_is_fGW = self.ne_is_fGW
      provider_kwargs['ne_bound_right'] = ne_bound_right
    else:
      self.ne_bound_right_is_absolute = True

    return ProfileConditionsProvider(**provider_kwargs)


@chex.dataclass
class ProfileConditionsProvider(
    base.RuntimeParametersProvider['DynamicProfileConditions']
):
  """Provider to retrieve initial and prescribed values and boundary conditions."""

  runtime_params_config: ProfileConditions
  Ip: interpolated_param.InterpolatedVarSingleAxis
  Ti_bound_right: (
      interpolated_param.InterpolatedVarSingleAxis
      | interpolated_param.InterpolatedVarTimeRho
  )
  Te_bound_right: (
      interpolated_param.InterpolatedVarSingleAxis
      | interpolated_param.InterpolatedVarTimeRho
  )
  Ti: interpolated_param.InterpolatedVarTimeRho
  Te: interpolated_param.InterpolatedVarTimeRho
  psi: interpolated_param.InterpolatedVarTimeRho | None
  ne: interpolated_param.InterpolatedVarTimeRho
  nbar: interpolated_param.InterpolatedVarSingleAxis
  ne_bound_right: (
      interpolated_param.InterpolatedVarSingleAxis
      | interpolated_param.InterpolatedVarTimeRho
  )
  set_pedestal: interpolated_param.InterpolatedVarSingleAxis
  Tiped: interpolated_param.InterpolatedVarSingleAxis
  Teped: interpolated_param.InterpolatedVarSingleAxis
  neped: interpolated_param.InterpolatedVarSingleAxis
  Ped_top: interpolated_param.InterpolatedVarSingleAxis

  @override
  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> DynamicProfileConditions:
    """Builds a DynamicProfileConditions."""
    return DynamicProfileConditions(**self.get_dynamic_params_kwargs(t))


@chex.dataclass
class DynamicProfileConditions:
  """Prescribed values and boundary conditions for the core profiles."""

  Ip: array_typing.ScalarFloat
  Ti_bound_right: array_typing.ScalarFloat
  Te_bound_right: array_typing.ScalarFloat
  # Temperature profiles defined on the cell grid.
  Te: array_typing.ArrayFloat
  Ti: array_typing.ArrayFloat
  # If provided as array, Psi profile defined on the cell grid.
  psi: array_typing.ArrayFloat | None
  # Electron density profile on the cell grid.
  ne: array_typing.ArrayFloat
  normalize_to_nbar: array_typing.ScalarBool
  nbar: array_typing.ScalarFloat
  ne_is_fGW: array_typing.ScalarBool
  ne_bound_right: array_typing.ScalarFloat
  ne_bound_right_is_fGW: array_typing.ScalarBool
  ne_bound_right_is_absolute: array_typing.ScalarBool
  set_pedestal: array_typing.ScalarBool
  Tiped: array_typing.ScalarFloat
  Teped: array_typing.ScalarFloat
  neped: array_typing.ScalarFloat
  neped_is_fGW: array_typing.ScalarBool
  Ped_top: array_typing.ScalarFloat
  nu: array_typing.ScalarFloat
  initial_j_is_total_current: array_typing.ScalarBool
  initial_psi_from_j: array_typing.ScalarBool

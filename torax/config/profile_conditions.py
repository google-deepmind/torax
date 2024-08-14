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
from torax import geometry
from torax import interpolated_param
from torax.config import base
from torax.config import config_args
from typing_extensions import override


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

  @override
  def make_provider(
      self, torax_mesh: geometry.Grid1D | None = None,
  ) -> ProfileConditionsProvider:
    if torax_mesh is None:
      raise ValueError('torax_mesh is required for ProfileConditionsProvider.')
    if self.Te_bound_right is None:
      logging.info('Setting electron temperature boundary condition using Te.')
      Te_bound_right = interpolated_param.InterpolatedVarTimeRho(
          self.Te, torax_mesh.face_centers[-1]
      )
    else:
      Te_bound_right = config_args.get_interpolated_var_single_axis(
          self.Te_bound_right
      )
    if self.Ti_bound_right is None:
      logging.info('Setting ion temperature boundary condition using Ti.')
      Ti_bound_right = interpolated_param.InterpolatedVarTimeRho(
          self.Ti, torax_mesh.face_centers[-1]
      )
    else:
      Ti_bound_right = config_args.get_interpolated_var_single_axis(
          self.Ti_bound_right
      )
    if self.ne_bound_right is None:
      logging.info('Setting electron density boundary condition using ne.')
      ne_bound_right = interpolated_param.InterpolatedVarTimeRho(
          self.ne, torax_mesh.face_centers[-1]
      )
      self.ne_bound_right_is_absolute = False
      self.ne_bound_right_is_fGW = self.ne_is_fGW
    else:
      ne_bound_right = config_args.get_interpolated_var_single_axis(
          self.ne_bound_right
      )
      self.ne_bound_right_is_absolute = True

    if self.psi is None:
      psi = None
    else:
      psi = interpolated_param.InterpolatedVarTimeRho(
          self.psi,
          torax_mesh.cell_centers,
      )

    return ProfileConditionsProvider(
        runtime_params_config=self,
        Ip=config_args.get_interpolated_var_single_axis(self.Ip),
        Ti_bound_right=Ti_bound_right,
        Te_bound_right=Te_bound_right,
        Ti=interpolated_param.InterpolatedVarTimeRho(
            self.Ti, torax_mesh.cell_centers
        ),
        Te=interpolated_param.InterpolatedVarTimeRho(
            self.Te, torax_mesh.cell_centers
        ),
        psi=psi,
        ne=interpolated_param.InterpolatedVarTimeRho(
            self.ne, torax_mesh.cell_centers
        ),
        nbar=config_args.get_interpolated_var_single_axis(self.nbar),
        ne_bound_right=ne_bound_right,
        set_pedestal=config_args.get_interpolated_var_single_axis(
            self.set_pedestal
        ),
        Tiped=config_args.get_interpolated_var_single_axis(self.Tiped),
        Teped=config_args.get_interpolated_var_single_axis(self.Teped),
        neped=config_args.get_interpolated_var_single_axis(self.neped),
        Ped_top=config_args.get_interpolated_var_single_axis(self.Ped_top),
    )


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
    return DynamicProfileConditions(
        Ip=float(self.Ip.get_value(t)),
        Ti_bound_right=float(self.Ti_bound_right.get_value(t)),
        Te_bound_right=float(self.Te_bound_right.get_value(t)),
        Ti=self.Ti.get_value(t),
        Te=self.Te.get_value(t),
        psi=self.psi.get_value(t) if self.psi else None,
        ne=self.ne.get_value(t),
        normalize_to_nbar=self.runtime_params_config.normalize_to_nbar,
        nbar=float(self.nbar.get_value(t)),
        ne_is_fGW=self.runtime_params_config.ne_is_fGW,
        ne_bound_right=float(self.ne_bound_right.get_value(t)),
        ne_bound_right_is_fGW=self.runtime_params_config.ne_bound_right_is_fGW,
        ne_bound_right_is_absolute=self.runtime_params_config.ne_bound_right_is_absolute,
        set_pedestal=bool(self.set_pedestal.get_value(t)),
        Tiped=float(self.Tiped.get_value(t)),
        Teped=float(self.Teped.get_value(t)),
        neped=float(self.neped.get_value(t)),
        neped_is_fGW=self.runtime_params_config.neped_is_fGW,
        Ped_top=float(self.Ped_top.get_value(t)),
        nu=self.runtime_params_config.nu,
        initial_j_is_total_current=self.runtime_params_config.initial_j_is_total_current,
        initial_psi_from_j=self.runtime_params_config.initial_psi_from_j,
    )


@chex.dataclass
class DynamicProfileConditions:
  """Prescribed values and boundary conditions for the core profiles."""

  Ip: float
  Ti_bound_right: float
  Te_bound_right: float
  # Temperature profiles defined on the cell grid.
  Te: chex.Array
  Ti: chex.Array
  # If provided as array, Psi profile defined on the cell grid.
  psi: chex.Array | None
  # Electron density profile on the cell grid.
  ne: chex.Array
  normalize_to_nbar: bool
  nbar: float
  ne_is_fGW: bool
  ne_bound_right: float
  ne_bound_right_is_fGW: bool
  ne_bound_right_is_absolute: bool
  set_pedestal: bool
  Tiped: float
  Teped: float
  neped: float
  neped_is_fGW: bool
  Ped_top: float
  nu: float
  initial_j_is_total_current: bool
  initial_psi_from_j: bool

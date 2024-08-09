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
  Ip: interpolated_param.InterpolatedVarSingleAxisInput = 15.0

  # Temperature boundary conditions at r=Rmin. If provided this will override
  # the temperature boundary conditions being taken from the
  # `TimeRhoInterpolated`s.
  Ti_bound_right: interpolated_param.InterpolatedVarSingleAxisInput | None = (
      None
  )
  Te_bound_right: interpolated_param.InterpolatedVarSingleAxisInput | None = (
      None
  )
  # Prescribed or evolving values for temperature at different times.
  # The outer mapping is for times and the inner mapping is for values of
  # temperature along the rho grid.
  Ti: interpolated_param.InterpolatedVarTimeRhoInput = dataclasses.field(
      default_factory=lambda: {0: {0: 15.0, 1: 1.0}}
  )
  Te: interpolated_param.InterpolatedVarTimeRhoInput = dataclasses.field(
      default_factory=lambda: {0: {0: 15.0, 1: 1.0}}
  )

  # Initial values for psi. If provided, the initial psi will be taken from
  # here. Otherwise, the initial psi will be calculated from either the geometry
  # or the "nu formula".
  psi: interpolated_param.InterpolatedVarTimeRhoInput | None = None

  # Prescribed or evolving values for electron density at different times.
  # The outer mapping is for times and the inner mapping is for values of
  # density along the rho grid.
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
  nbar: interpolated_param.InterpolatedVarSingleAxisInput = 0.85
  # Toggle units of nbar
  ne_is_fGW: bool = True

  # Density boundary condition for r=Rmin.
  # In units of reference density if ne_bound_right_is_fGW = False.
  # In Greenwald fraction if ne_bound_right_is_fGW = True.
  ne_bound_right: interpolated_param.InterpolatedVarSingleAxisInput | None = (
      None
  )
  ne_bound_right_is_fGW: bool = False
  ne_bound_right_is_absolute: bool = False

  # Internal boundary condition (pedestal)
  # Do not set internal boundary condition if this is False
  set_pedestal: interpolated_param.InterpolatedVarSingleAxisInput = True
  # ion pedestal top temperature in keV
  Tiped: interpolated_param.InterpolatedVarSingleAxisInput = 5.0
  # electron pedestal top temperature in keV
  Teped: interpolated_param.InterpolatedVarSingleAxisInput = 5.0
  # pedestal top electron density
  # In units of reference density if neped_is_fGW = False.
  # In Greenwald fraction if neped_is_fGW = True.
  neped: interpolated_param.InterpolatedVarSingleAxisInput = 0.7
  neped_is_fGW: bool = False
  # Set ped top location.
  Ped_top: interpolated_param.InterpolatedVarSingleAxisInput = 0.91

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
      Te_bound_right = config_args.get_interpolated_var_2d(
          self.Te, torax_mesh.face_centers[-1]
      )
    else:
      Te_bound_right = interpolated_param.InterpolatedVarSingleAxis(
          self.Te_bound_right
      )
    if self.Ti_bound_right is None:
      logging.info('Setting ion temperature boundary condition using Ti.')
      Ti_bound_right = config_args.get_interpolated_var_2d(
          self.Ti, torax_mesh.face_centers[-1]
      )
    else:
      Ti_bound_right = interpolated_param.InterpolatedVarSingleAxis(
          self.Ti_bound_right
      )
    if self.ne_bound_right is None:
      logging.info('Setting electron density boundary condition using ne.')
      ne_bound_right = config_args.get_interpolated_var_2d(
          self.ne, torax_mesh.face_centers[-1]
      )
      self.ne_bound_right_is_absolute = False
      self.ne_bound_right_is_fGW = self.ne_is_fGW
    else:
      ne_bound_right = interpolated_param.InterpolatedVarSingleAxis(
          self.ne_bound_right
      )
      self.ne_bound_right_is_absolute = True

    if self.psi is None:
      psi = None
    else:
      psi = config_args.get_interpolated_var_2d(
          self.psi, torax_mesh.cell_centers
      )

    return ProfileConditionsProvider(
        runtime_params_config=self,
        Ip=interpolated_param.InterpolatedVarSingleAxis(self.Ip),
        Ti_bound_right=Ti_bound_right,
        Te_bound_right=Te_bound_right,
        Ti=config_args.get_interpolated_var_2d(
            self.Ti, torax_mesh.cell_centers
        ),
        Te=config_args.get_interpolated_var_2d(
            self.Te, torax_mesh.cell_centers
        ),
        psi=psi,
        ne=config_args.get_interpolated_var_2d(
            self.ne, torax_mesh.cell_centers
        ),
        nbar=interpolated_param.InterpolatedVarSingleAxis(self.nbar),
        ne_bound_right=ne_bound_right,
        set_pedestal=interpolated_param.InterpolatedVarSingleAxis(
            self.set_pedestal
        ),
        Tiped=interpolated_param.InterpolatedVarSingleAxis(self.Tiped),
        Teped=interpolated_param.InterpolatedVarSingleAxis(self.Teped),
        neped=interpolated_param.InterpolatedVarSingleAxis(self.neped),
        Ped_top=interpolated_param.InterpolatedVarSingleAxis(self.Ped_top),
    )


@dataclasses.dataclass
class ProfileConditionsProvider(
    base.RuntimeParametersProvider['DynamicProfileConditions']
):
  """Prescribed values and boundary conditions for the core profiles."""

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

  # total plasma current in MA
  # Note that if Ip_from_parameters=False in geometry, then this Ip will be
  # overwritten by values from the geometry data
  Ip: float

  # Temperature boundary conditions at r=Rmin.
  Ti_bound_right: float
  Te_bound_right: float
  # Radial array used for initial conditions, and prescribed time-dependent
  # conditions when not evolving variable with PDE defined on the cell grid.
  Te: chex.Array
  Ti: chex.Array

  # Radial array and boundary condition used for initial conditions of psi
  # defined on the cell grid.
  psi: chex.Array | None

  # Electron density profile on the cell grid.
  # If density evolves with PDE (dens_eq=True), then is initial condition
  ne: chex.Array
  # Whether to renormalize the density profile.
  normalize_to_nbar: bool

  # Initial line averaged density.
  # In units of reference density if ne_is_fGW = False.
  # In Greenwald fraction if ne_is_fGW = True.
  # nGW = Ip/(pi*a^2) with a in m, nGW in 10^20 m-3, Ip in MA
  nbar: float
  # Toggle units of nbar
  ne_is_fGW: bool

  # Density boundary condition for r=Rmin, units of nref
  # In units of reference density if ne_bound_right_is_fGW = False.
  # In Greenwald fraction if ne_bound_right_is_fGW = True.
  ne_bound_right: float
  ne_bound_right_is_fGW: bool
  # If `ne_bound_right` is set using `ne` then this flag should be `False`.
  ne_bound_right_is_absolute: bool

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

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

"""Dataclass representing runtime parameter inputs to the transport models.

This is the dataclass runtime config exposed to the user. The actual model gets
a time-interpolated version of this config via the DynamicRuntimeParams.
"""

from __future__ import annotations

from typing import TypeAlias

import chex
from torax import geometry
from torax import interpolated_param
from torax import jax_utils
from torax.config import base
from torax.config import config_args


# Type-alias for clarity. While the InterpolatedVarSingleAxiss can vary across
# any field, in these classes, we mainly use it to handle time-dependent
# parameters.
TimeInterpolated: TypeAlias = interpolated_param.TimeInterpolated


# pylint: disable=invalid-name
@chex.dataclass
class RuntimeParams(base.RuntimeParametersConfig['RuntimeParamsProvider']):
  """Runtime parameters for the turbulent transport model.

  This is the dataclass runtime config exposed to the user. The actual model
  gets a time-interpolated version of this config via the DynamicConfigSlice.
  """

  # Allowed chi and diffusivity bounds
  chimin: float = 0.05  # minimum chi
  chimax: float = 100.0  # maximum chi (can be helpful for stability)
  Demin: float = 0.05  # minimum electron density diffusivity
  Demax: float = 100.0  # maximum electron density diffusivity
  Vemin: float = -50.0  # minimum electron density convection
  Vemax: float = 50.0  # minimum electron density convection

  # set inner core transport coefficients (ad-hoc MHD/EM transport)
  apply_inner_patch: TimeInterpolated = False
  De_inner: TimeInterpolated = 0.2
  Ve_inner: TimeInterpolated = 0.0
  chii_inner: TimeInterpolated = 1.0
  chie_inner: TimeInterpolated = 1.0
  # normalized radius below which patch is applied
  rho_inner: TimeInterpolated = 0.3

  # set outer core transport coefficients.
  # Useful for L-mode near-edge region where QLKNN10D is not applicable.
  # Only used when set_pedestal = False
  apply_outer_patch: TimeInterpolated = False
  De_outer: TimeInterpolated = 0.2
  Ve_outer: TimeInterpolated = 0.0
  chii_outer: TimeInterpolated = 1.0
  chie_outer: TimeInterpolated = 1.0
  # normalized radius above which patch is applied
  rho_outer: TimeInterpolated = 0.9

  # Width of HWHM Gaussian smoothing kernel operating on transport model outputs
  smoothing_sigma: float = 0.0

  # Smooth across entire radial domain regardless of inner and outer patches.
  smooth_everywhere: bool = False

  def make_provider(
      self, torax_mesh: geometry.Grid1D | None = None
  ) -> RuntimeParamsProvider:
    # TODO(b/360831279)
    return RuntimeParamsProvider(
        runtime_params_config=self,
        apply_inner_patch=config_args.get_interpolated_var_single_axis(
            self.apply_inner_patch
        ),
        De_inner=config_args.get_interpolated_var_single_axis(self.De_inner),
        Ve_inner=config_args.get_interpolated_var_single_axis(self.Ve_inner),
        chii_inner=config_args.get_interpolated_var_single_axis(
            self.chii_inner
        ),
        chie_inner=config_args.get_interpolated_var_single_axis(
            self.chie_inner
        ),
        rho_inner=config_args.get_interpolated_var_single_axis(self.rho_inner),
        apply_outer_patch=config_args.get_interpolated_var_single_axis(
            self.apply_outer_patch
        ),
        De_outer=config_args.get_interpolated_var_single_axis(self.De_outer),
        Ve_outer=config_args.get_interpolated_var_single_axis(self.Ve_outer),
        chii_outer=config_args.get_interpolated_var_single_axis(
            self.chii_outer
        ),
        chie_outer=config_args.get_interpolated_var_single_axis(
            self.chie_outer
        ),
        rho_outer=config_args.get_interpolated_var_single_axis(self.rho_outer),
    )


@chex.dataclass
class RuntimeParamsProvider(
    base.RuntimeParametersProvider['DynamicRuntimeParams']
):
  """Provides a RuntimeParams to use during time t of the sim."""

  runtime_params_config: RuntimeParams
  apply_inner_patch: interpolated_param.InterpolatedVarSingleAxis
  De_inner: interpolated_param.InterpolatedVarSingleAxis
  Ve_inner: interpolated_param.InterpolatedVarSingleAxis
  chii_inner: interpolated_param.InterpolatedVarSingleAxis
  chie_inner: interpolated_param.InterpolatedVarSingleAxis
  rho_inner: interpolated_param.InterpolatedVarSingleAxis
  apply_outer_patch: interpolated_param.InterpolatedVarSingleAxis
  De_outer: interpolated_param.InterpolatedVarSingleAxis
  Ve_outer: interpolated_param.InterpolatedVarSingleAxis
  chii_outer: interpolated_param.InterpolatedVarSingleAxis
  chie_outer: interpolated_param.InterpolatedVarSingleAxis
  rho_outer: interpolated_param.InterpolatedVarSingleAxis

  def build_dynamic_params(self, t: chex.Numeric) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(
        chimin=self.runtime_params_config.chimin,
        chimax=self.runtime_params_config.chimax,
        Demin=self.runtime_params_config.Demin,
        Demax=self.runtime_params_config.Demax,
        Vemin=self.runtime_params_config.Vemin,
        Vemax=self.runtime_params_config.Vemax,
        apply_inner_patch=bool(self.apply_inner_patch.get_value(t)),
        De_inner=float(self.De_inner.get_value(t)),
        Ve_inner=float(self.Ve_inner.get_value(t)),
        chii_inner=float(self.chii_inner.get_value(t)),
        chie_inner=float(self.chie_inner.get_value(t)),
        rho_inner=float(self.rho_inner.get_value(t)),
        apply_outer_patch=bool(self.apply_outer_patch.get_value(t)),
        De_outer=float(self.De_outer.get_value(t)),
        Ve_outer=float(self.Ve_outer.get_value(t)),
        chii_outer=float(self.chii_outer.get_value(t)),
        chie_outer=float(self.chie_outer.get_value(t)),
        rho_outer=float(self.rho_outer.get_value(t)),
        smoothing_sigma=self.runtime_params_config.smoothing_sigma,
        smooth_everywhere=self.runtime_params_config.smooth_everywhere,
    )


@chex.dataclass(frozen=True)
class DynamicRuntimeParams:
  """Input params for the transport model which can be used as compiled args."""

  chimin: float
  chimax: float
  Demin: float
  Demax: float
  Vemin: float
  Vemax: float
  apply_inner_patch: bool
  De_inner: float
  Ve_inner: float
  chii_inner: float
  chie_inner: float
  rho_inner: float
  apply_outer_patch: bool
  De_outer: float
  Ve_outer: float
  chii_outer: float
  chie_outer: float
  rho_outer: float
  smoothing_sigma: float
  smooth_everywhere: bool

  def sanity_check(self):
    """Make sure all the parameters are valid."""
    # Using the object.__setattr__ call to get around the fact that this
    # dataclass is frozen.
    object.__setattr__(
        self, 'chimin', jax_utils.error_if_negative(self.chimin, 'chimin')
    )
    object.__setattr__(
        self,
        'chimax',
        jax_utils.error_if(
            self.chimax, self.chimax <= self.chimin, 'chimax must be > chimin'
        ),
    )
    object.__setattr__(
        self, 'Demin', jax_utils.error_if_negative(self.Demin, 'Demin')
    )
    object.__setattr__(
        self,
        'Demax',
        jax_utils.error_if(
            self.Demax, self.Demax <= self.Demin, 'Demax must be > Demin'
        ),
    )
    object.__setattr__(
        self,
        'Vemax',
        jax_utils.error_if(
            self.Vemax, self.Vemax <= self.Vemin, 'Vemax must be > Vemin'
        ),
    )
    object.__setattr__(
        self, 'De_inner', jax_utils.error_if_negative(self.De_inner, 'De_inner')
    )
    object.__setattr__(
        self,
        'chii_inner',
        jax_utils.error_if_negative(self.chii_inner, 'chii_inner'),
    )
    object.__setattr__(
        self,
        'chie_inner',
        jax_utils.error_if_negative(self.chie_inner, 'chie_inner'),
    )
    object.__setattr__(
        self,
        'rho_inner',
        jax_utils.error_if_negative(self.rho_inner, 'rho_inner'),
    )
    object.__setattr__(
        self,
        'rho_inner',
        jax_utils.error_if(
            self.rho_inner, self.rho_inner > 1.0, 'rho_inner must be <= 1.'
        ),
    )
    object.__setattr__(
        self,
        'rho_outer',
        jax_utils.error_if(
            self.rho_outer, self.rho_outer > 1.0, 'rho_outer must be <= 1.'
        ),
    )
    object.__setattr__(
        self,
        'rho_outer',
        jax_utils.error_if(
            self.rho_outer,
            self.rho_outer <= self.rho_inner,
            'rho_outer must be > rho_inner',
        ),
    )

  def __post_init__(self):
    self.sanity_check()

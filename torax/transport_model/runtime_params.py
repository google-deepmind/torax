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
import jax
from torax import array_typing
from torax import interpolated_param
from torax import jax_utils
from torax.config import base
from torax.torax_pydantic import torax_pydantic

# Type-alias for clarity. While the InterpolatedVarSingleAxiss can vary across
# any field, in these classes, we mainly use it to handle time-dependent
# parameters.
TimeInterpolatedInput: TypeAlias = interpolated_param.TimeInterpolatedInput


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
  apply_inner_patch: TimeInterpolatedInput = False
  De_inner: TimeInterpolatedInput = 0.2
  Ve_inner: TimeInterpolatedInput = 0.0
  chii_inner: TimeInterpolatedInput = 1.0
  chie_inner: TimeInterpolatedInput = 1.0
  # normalized radius below which patch is applied
  rho_inner: TimeInterpolatedInput = 0.3

  # set outer core transport coefficients.
  # Useful for L-mode near-edge region where QLKNN10D is not applicable.
  # Only used when set_pedestal = False
  apply_outer_patch: TimeInterpolatedInput = False
  De_outer: TimeInterpolatedInput = 0.2
  Ve_outer: TimeInterpolatedInput = 0.0
  chii_outer: TimeInterpolatedInput = 1.0
  chie_outer: TimeInterpolatedInput = 1.0
  # normalized radius above which patch is applied
  rho_outer: TimeInterpolatedInput = 0.9

  # Width of HWHM Gaussian smoothing kernel operating on transport model outputs
  smoothing_sigma: float = 0.0

  # Smooth across entire radial domain regardless of inner and outer patches.
  smooth_everywhere: bool = False

  def make_provider(
      self, torax_mesh: torax_pydantic.Grid1D | None = None
  ) -> RuntimeParamsProvider:
    return RuntimeParamsProvider(**self.get_provider_kwargs(torax_mesh))


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
    return DynamicRuntimeParams(**self.get_dynamic_params_kwargs(t))


@chex.dataclass(frozen=True)
class DynamicRuntimeParams:
  """Input params for the transport model which can be used as compiled args."""

  chimin: jax.Array
  chimax: jax.Array
  Demin: jax.Array
  Demax: jax.Array
  Vemin: jax.Array
  Vemax: jax.Array
  apply_inner_patch: array_typing.ScalarBool
  De_inner: array_typing.ScalarFloat
  Ve_inner: array_typing.ScalarFloat
  chii_inner: array_typing.ScalarFloat
  chie_inner: array_typing.ScalarFloat
  rho_inner: array_typing.ScalarFloat
  apply_outer_patch: array_typing.ScalarBool
  De_outer: array_typing.ScalarFloat
  Ve_outer: array_typing.ScalarFloat
  chii_outer: array_typing.ScalarFloat
  chie_outer: array_typing.ScalarFloat
  rho_outer: array_typing.ScalarFloat
  smoothing_sigma: jax.Array
  smooth_everywhere: jax.Array

  def sanity_check(self):
    """Make sure all the parameters are valid."""
    jax_utils.error_if_negative(self.chimin, 'chimin')
    jax_utils.error_if(
        self.chimax, self.chimax <= self.chimin, 'chimax must be > chimin'
    )
    jax_utils.error_if_negative(self.Demin, 'Demin')
    jax_utils.error_if(
        self.Demax, self.Demax <= self.Demin, 'Demax must be > Demin'
    )
    jax_utils.error_if(
        self.Vemax, self.Vemax <= self.Vemin, 'Vemax must be > Vemin'
    )
    jax_utils.error_if_negative(self.De_inner, 'De_inner')
    jax_utils.error_if_negative(self.chii_inner, 'chii_inner')
    jax_utils.error_if_negative(self.chie_inner, 'chie_inner')
    jax_utils.error_if(
        self.rho_inner, self.rho_inner > 1.0, 'rho_inner must be <= 1.'
    )
    jax_utils.error_if(
        self.rho_outer, self.rho_outer > 1.0, 'rho_outer must be <= 1.'
    )
    jax_utils.error_if(
        self.rho_outer,
        self.rho_outer <= self.rho_inner,
        'rho_outer must be > rho_inner',
    )

  def __post_init__(self):
    self.sanity_check()

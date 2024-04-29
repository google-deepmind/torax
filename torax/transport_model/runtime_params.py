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
a time-interpolated version of this config via the DynamicConfigSlice.
"""

from __future__ import annotations

import chex
from torax import interpolated_param
from torax import jax_utils
from torax.config import config_args


# Type-alias for clarity. While the InterpolatedParams can vary across any
# field, in these classes, we mainly use it to handle time-dependent parameters.
TimeDependentField = interpolated_param.InterpParamOrInterpParamInput


# pylint: disable=invalid-name
@chex.dataclass(eq=True, frozen=True)
class RuntimeParams:
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
  apply_inner_patch: TimeDependentField = False
  De_inner: TimeDependentField = 0.2
  Ve_inner: TimeDependentField = 0.0
  chii_inner: TimeDependentField = 1.0
  chie_inner: TimeDependentField = 1.0
  # normalized radius below which patch is applied
  rho_inner: TimeDependentField = 0.3

  # set outer core transport coefficients.
  # Useful for L-mode near-edge region where QLKNN10D is not applicable.
  # Only used when set_pedestal = False
  apply_outer_patch: TimeDependentField = False
  De_outer: TimeDependentField = 0.2
  Ve_outer: TimeDependentField = 0.0
  chii_outer: TimeDependentField = 1.0
  chie_outer: TimeDependentField = 1.0
  # normalized radius above which patch is applied
  rho_outer: TimeDependentField = 0.9

  # Width of HWHM Gaussian smoothing kernel operating on transport model outputs
  smoothing_sigma: float = 0.0

  def build_dynamic_params(self, t: chex.Numeric) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(
        **config_args.get_init_kwargs(
            input_config=self,
            output_type=DynamicRuntimeParams,
            t=t,
        )
    )


@chex.dataclass(frozen=True)
class DynamicRuntimeParams:
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

  # Width of HWHM Gaussian smoothing kernel operating on transport model outputs
  smoothing_sigma: float

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

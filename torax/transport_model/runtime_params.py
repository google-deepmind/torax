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
import chex
from torax import array_typing
from torax import jax_utils


# pylint: disable=invalid-name
@chex.dataclass(frozen=True)
class DynamicRuntimeParams:
  """Input params for the transport model which can be used as compiled args."""

  chimin: float
  chimax: float
  Demin: float
  Demax: float
  Vemin: float
  Vemax: float
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
  smoothing_sigma: float
  smooth_everywhere: bool

  def __post_init__(self):
    jax_utils.error_if(
        self.rho_outer,
        self.rho_outer <= self.rho_inner,
        'rho_outer must be greater than rho_inner.',
    )

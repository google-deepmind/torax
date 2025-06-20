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
from torax._src import array_typing


# pylint: disable=invalid-name
@chex.dataclass(frozen=True)
class DynamicRuntimeParams:
  """Input params for the transport model which can be used as compiled args."""

  chi_min: float
  chi_max: float
  D_e_min: float
  D_e_max: float
  V_e_min: float
  V_e_max: float
  rho_min: array_typing.ScalarFloat
  rho_max: array_typing.ScalarFloat
  apply_inner_patch: array_typing.ScalarBool
  D_e_inner: array_typing.ScalarFloat
  V_e_inner: array_typing.ScalarFloat
  chi_i_inner: array_typing.ScalarFloat
  chi_e_inner: array_typing.ScalarFloat
  rho_inner: array_typing.ScalarFloat
  apply_outer_patch: array_typing.ScalarBool
  D_e_outer: array_typing.ScalarFloat
  V_e_outer: array_typing.ScalarFloat
  chi_i_outer: array_typing.ScalarFloat
  chi_e_outer: array_typing.ScalarFloat
  rho_outer: array_typing.ScalarFloat
  smoothing_width: float
  smooth_everywhere: bool

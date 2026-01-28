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
a time-interpolated version of this config via the RuntimeParams.
"""
import dataclasses

import jax
from torax._src import array_typing
from torax._src.transport_model import enums


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams:
  """Input params for the transport model which can be used as compiled args."""

  chi_min: float
  chi_max: float
  D_e_min: float
  D_e_max: float
  V_e_min: float
  V_e_max: float
  rho_min: array_typing.FloatScalar
  rho_max: array_typing.FloatScalar
  apply_inner_patch: array_typing.BoolScalar
  D_e_inner: array_typing.FloatScalar
  V_e_inner: array_typing.FloatScalar
  chi_i_inner: array_typing.FloatScalar
  chi_e_inner: array_typing.FloatScalar
  rho_inner: array_typing.FloatScalar
  apply_outer_patch: array_typing.BoolScalar
  D_e_outer: array_typing.FloatScalar
  V_e_outer: array_typing.FloatScalar
  chi_i_outer: array_typing.FloatScalar
  chi_e_outer: array_typing.FloatScalar
  rho_outer: array_typing.FloatScalar
  smoothing_width: float
  smooth_everywhere: bool
  disable_chi_i: array_typing.BoolScalar
  disable_chi_e: array_typing.BoolScalar
  disable_D_e: array_typing.BoolScalar
  disable_V_e: array_typing.BoolScalar
  merge_mode: enums.MergeMode = dataclasses.field(metadata={'static': True})

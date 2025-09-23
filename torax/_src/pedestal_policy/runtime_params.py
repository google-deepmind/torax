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

"""Runtime parameters for PedestalPolicy."""
from __future__ import annotations
import abc
import dataclasses
import jax
from torax._src import array_typing
from torax._src import interpolated_param

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PedestalPolicyRuntimeParams(abc.ABC):
  """Abstract base class for runtime parameters for PedestalPolicy subclasses."""


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ConstantRP(PedestalPolicyRuntimeParams):
  """Runtime parameters for the Constant policy."""

  use_pedestal: array_typing.BoolScalar
  scale_pedestal: array_typing.FloatScalar | None = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TimeVaryingRP(PedestalPolicyRuntimeParams):
  """Runtime parameters for the TimeVarying policy."""

  # JAX arrays containing the time series data
  time: array_typing.Array
  value: array_typing.Array
  # Static metadata for how to interpret the arrays
  is_bool_param: bool = dataclasses.field(metadata={'static': True})
  interpolation_mode: interpolated_param.InterpolationMode = dataclasses.field(
      metadata={'static': True}
  )

  def get_value(self, t: float) -> array_typing.Array:
    """Interpolates the value at time t."""
    interp = interpolated_param.InterpolatedVarSingleAxis(
        value=(self.time, self.value),
        interpolation_mode=self.interpolation_mode,
        is_bool_param=self.is_bool_param,
    )
    return interp.get_value(t)

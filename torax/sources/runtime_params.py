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

"""Configuration for all the sources/sinks modelled in Torax."""

from __future__ import annotations

import enum

import chex
from torax import array_typing
from torax import interpolated_param


TimeInterpolatedInput = interpolated_param.TimeInterpolatedInput


@enum.unique
class Mode(enum.Enum):
  """Defines how to compute the source terms for this source/sink."""

  # Source is set to zero always.
  ZERO = "ZERO"

  # Source values come from a model in code. These terms can be implicit or
  # explicit depending on the model implementation.
  MODEL_BASED = "MODEL_BASED"

  # Source values come from a pre-determined set of values, that may evolve in
  # time. Currently, this is only supported for sources that have a 1D output
  # along the cell grid or face grid.
  PRESCRIBED = "PRESCRIBED"


@chex.dataclass(frozen=True)
class DynamicRuntimeParams:
  """Dynamic params for a single TORAX source.

  These params can be changed without triggering a recompile. TORAX sources are
  stateless, so these params are their inputs to determine their output
  profiles.
  """
  prescribed_values: array_typing.ArrayFloat


@chex.dataclass(frozen=True)
class StaticRuntimeParams:
  """Static params for the sources."""

  mode: str
  is_explicit: bool

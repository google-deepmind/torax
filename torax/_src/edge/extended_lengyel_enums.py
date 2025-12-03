# Copyright 2025 DeepMind Technologies Limited
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

"""Enums for the extended Lengyel model."""

import enum


class ComputationMode(enum.StrEnum):
  """Computation modes for the extended Lengyel model.

  Attributes:
    FORWARD: Calculate impurity concentrations for a given target temperature.
    INVERSE: Calculate target temperature for a given impurity concentration.
  """

  FORWARD = 'forward'
  INVERSE = 'inverse'


class SolverMode(enum.StrEnum):
  """Solver modes for the extended Lengyel model.

  Attributes:
    FIXED_POINT: A simple fixed-point iterative solver.
    NEWTON_RAPHSON: A Newton-Raphson solver (not yet implemented).
    HYBRID: A hybrid solver using a warm start from fixed-point, and then
      Newton-Raphson.
  """

  FIXED_POINT = 'fixed_point'
  NEWTON_RAPHSON = 'newton_raphson'
  HYBRID = 'hybrid'

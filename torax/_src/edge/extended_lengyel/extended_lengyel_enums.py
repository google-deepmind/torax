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


class FixedImpuritySourceOfTruth(enum.StrEnum):
  """Source of truth for fixed impurity concentrations when using an edge model.

  Determines how impurity concentrations are handled between the core plasma
  simulation and the edge model.

  Attributes:
    CORE: * The core impurity profiles are the source of truth. * The edge
      model's impurity concentrations are derived from the core values at the
      last closed flux surface: `c_edge = c_core_face[-1] * enrichment_factor`.
    EDGE: * The edge model's `fixed_impurity_concentrations` are the source of
      truth. * The core impurity profiles (n_e_ratios) are scaled to match the
      values determined by the edge model. runtime_params still sets the profile
      shape: `c_core = c_core / c_core_face[-1] * c_edge / enrichment_factor`.

  Note: For seeded impurities in the extended Lengyel edge model, the source of
  truth is always the edge model, regardless of this setting. This enum only
  controls the behavior for fixed impurities in that case.
  """

  CORE = 'core'
  EDGE = 'edge'

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

"""Pydantic configs for all edge models, currently only extended_lengyel."""

import enum
from typing import Annotated, Literal
import chex
import pydantic
from torax._src.edge import base
from torax._src.edge import extended_lengyel_model
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name


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
    FIXED_STEP: A simple fixed-step iterative solver.
    NEWTON_RAPHSON: A Newton-Raphson solver (not yet implemented).
    HYBRID: A hybrid solver using a warm start from fixed-step, and then
      Newton-Raphson.
  """

  FIXED_STEP = 'fixed_step'
  NEWTON_RAPHSON = 'newton_raphson'
  HYBRID = 'hybrid'


class ExtendedLengyelConfig(base.EdgeModelConfigBase):
  """Configuration for the extended Lengyel edge model."""

  model_name: Annotated[
      Literal['extended_lengyel'], torax_pydantic.JAX_STATIC
  ] = 'extended_lengyel'
  computation_mode: Annotated[ComputationMode, torax_pydantic.JAX_STATIC] = (
      ComputationMode.FORWARD
  )
  solver_mode: Annotated[SolverMode, torax_pydantic.JAX_STATIC] = (
      SolverMode.HYBRID
  )

  # TODO(b/446608829) - to be completed in a later PR.

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> extended_lengyel_model.RuntimeParams:
    # TODO(b/446608829) - to be completed in a later PR.
    return extended_lengyel_model.RuntimeParams()

  def build_edge_model(self) -> extended_lengyel_model.ExtendedLengyelModel:
    return extended_lengyel_model.ExtendedLengyelModel()


EdgeConfig = Annotated[
    ExtendedLengyelConfig,
    pydantic.Field(discriminator='model_name'),
]

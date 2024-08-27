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

import dataclasses
import enum

import chex
from torax import geometry
from torax import interpolated_param
from torax.config import base
from torax.sources import formula_config


TimeInterpolated = interpolated_param.TimeInterpolated


@enum.unique
class Mode(enum.Enum):
  """Defines how to compute the source terms for this source/sink."""

  # Source is set to zero always. This is an explicit source by definition.
  ZERO = 0

  # Source values come from a model in code. These terms can be implicit or
  # explicit depending on the model implementation.
  MODEL_BASED = 1

  # Source values come from a prescribed (possibly time-dependent) formula that
  # is not dependent on the state of the system. These formulas may be dependent
  # on the config and geometry of the system.
  FORMULA_BASED = 2

  # Source values come from a pre-determined set of values, that may evolve in
  # time. Values can be drawn from a file or an array. These sources are always
  # explicit.
  PRESCRIBED = 3


@dataclasses.dataclass
class RuntimeParams(base.RuntimeParametersConfig):
  """Configures a single source/sink term.

  This is a RUNTIME runtime_params, meaning its values can change from run to
  run without triggering a recompile. This config defines the runtime config for
  the entire simulation run. The DynamicRuntimeParams, which is derived from
  this class, only contains information for a single time step.

  Any compile-time configurations for the Sources should go into the Source
  object's constructor.
  """

  # Defines how the source values are computed (from a model, from a file, etc.)
  mode: Mode = Mode.ZERO

  # Defines whether this is an explicit or implicit source.
  # Explicit sources are calculated based on the simulation state at the
  # beginning of a time step, or do not have any dependance on state. Implicit
  # sources depend on updated states as our iterative solvers evolve the state
  # through the course of a time step.
  #
  # NOTE: Not all source types can be implicit or explicit. For example,
  # file-based sources are always explicit. If an incorrect combination of
  # source type and is_explicit is passed in, an error will be thrown when
  # running the simulation.
  is_explicit: bool = False

  # Parameters used only when the source is using a prescribed formula to
  # compute its profile.
  formula: base.RuntimeParametersConfig = dataclasses.field(
      default_factory=formula_config.Exponential
  )

  # Prescribed values for the source. Used only when the source is fully
  # prescribed (i.e. source.mode == Mode.PRESCRIBED).
  # The default here is a vector of all zeros along for all rho and time, and
  # the output vector is along the cell grid.
  # NOTE: For Sources that have different output shapes, make sure to update
  # build_dynamic_params() to handle the new shape. The default implementation
  # assumes a 1D output along the cell grid.
  prescribed_values: interpolated_param.InterpolatedVarTimeRhoInput = (
      dataclasses.field(default_factory=lambda: {0: {0: 0, 1: 0}})
  )

  def make_provider(
      self,
      torax_mesh: geometry.Grid1D | None = None,
  ) -> RuntimeParamsProvider:
    return RuntimeParamsProvider(**self.get_provider_kwargs(torax_mesh))


@chex.dataclass
class RuntimeParamsProvider(
    base.RuntimeParametersProvider['DynamicRuntimeParams']
):
  """Runtime parameter provider for a single source/sink term."""

  runtime_params_config: RuntimeParams
  formula: base.RuntimeParametersProvider
  prescribed_values: interpolated_param.InterpolatedVarTimeRho

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(**self.get_dynamic_params_kwargs(t))


@chex.dataclass(frozen=True)
class DynamicRuntimeParams:
  """Dynamic params for a single TORAX source.

  These params can be changed without triggering a recompile. TORAX sources are
  stateless, so these params are their inputs to determine their output
  profiles.
  """

  # This maps to the enum value for the Mode enum. The enum itself is not
  # JAX-friendly.
  mode: int
  is_explicit: bool
  formula: formula_config.DynamicFormula
  prescribed_values: chex.Array

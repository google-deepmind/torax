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
"""Base model for source pydantic configs."""

import abc
from typing import Annotated

import chex
from torax._src.sources import runtime_params
from torax._src.sources import source as source_lib
from torax._src.sources import source_profiles
from torax._src.torax_pydantic import torax_pydantic


class PrescribedValues(torax_pydantic.ConfigBase):
  """Prescribed profile values for a source in `Mode.PRESCRIBED`.

  Each attribute corresponds to a core profile channel on the cell grid and
  is optional (`None` by default). Only the channels matching the source's
  `AFFECTED_CORE_PROFILES` should be set when `mode == Mode.PRESCRIBED`.

  Attributes:
    power_density_i: Ion heating power density [W/m^3].
    power_density_e: Electron heating power density [W/m^3].
    j_parallel: Parallel current density `<j.B>/B_0` [A/m^2].
    particle_source: Electron particle source density [m^-3 s^-1].
  """

  power_density_i: torax_pydantic.TimeVaryingArray | None = None
  power_density_e: torax_pydantic.TimeVaryingArray | None = None
  j_parallel: torax_pydantic.TimeVaryingArray | None = None
  particle_source: torax_pydantic.TimeVaryingArray | None = None

  def build_source_output(
      self, t: chex.Numeric
  ) -> source_profiles.SourceOutput:
    """Evaluates prescribed time-varying arrays at time `t`."""
    return source_profiles.SourceOutput(
        power_density_i=(
            self.power_density_i.get_value(t)
            if self.power_density_i is not None
            else None
        ),
        power_density_e=(
            self.power_density_e.get_value(t)
            if self.power_density_e is not None
            else None
        ),
        j_parallel=(
            self.j_parallel.get_value(t)
            if self.j_parallel is not None
            else None
        ),
        particle_source=(
            self.particle_source.get_value(t)
            if self.particle_source is not None
            else None
        ),
    )


class SourceConfigBase(torax_pydantic.ConfigBase):
  """Base model holding parameters common to all source model configs.

  Subclasses should define the `model_name` attribute as a `Literal`
  string. This is used as an identifier for the model in the source config
  for Pydantic to "discriminate" against. This should be given a unique value
  for each source model implementation.

  Subclasses must override `build_source` to return the `Source` object for this
  config, and may override `build_runtime_params` if the source uses custom
  runtime parameters.

  Attributes:
    mode: Defines how the source values are computed (from a model, from a file,
      etc.)
    is_explicit: Defines whether this is an explicit or implicit source.
      Explicit sources are calculated based on the simulation state at the
      beginning of a time step, or do not have any dependence on state. Implicit
      sources depend on updated states as our iterative solvers evolve the state
      through the course of a time step. NOTE: Not all source types can be
      implicit or explicit. For example, file-based sources are always explicit.
      If an incorrect combination of source type and is_explicit is passed in,
      an error will be thrown when running the simulation.
    prescribed_values: Prescribed profile values for the source, keyed by output
      field name (`power_density_i`, `power_density_e`, `j_parallel`,
      `particle_source`). Used only when the source is fully prescribed (i.e.
      `source.mode == Mode.PRESCRIBED`).
  """

  mode: Annotated[runtime_params.Mode, torax_pydantic.JAX_STATIC] = (
      runtime_params.Mode.MODEL_BASED
  )
  is_explicit: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  prescribed_values: PrescribedValues = torax_pydantic.ValidatedDefault({})

  @abc.abstractmethod
  def build_source(self) -> source_lib.Source:
    """Builds a source object from the model config."""

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> runtime_params.RuntimeParams:
    """Builds runtime parameters for the source."""
    return runtime_params.RuntimeParams(
        prescribed_values=self.prescribed_values.build_source_output(t),
        mode=self.mode,
        is_explicit=self.is_explicit,
    )

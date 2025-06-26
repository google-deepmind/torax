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

import chex

from torax._src.sources import runtime_params
from torax._src.sources import source as source_lib
from torax._src.torax_pydantic import torax_pydantic


class SourceModelBase(torax_pydantic.BaseModelFrozen, abc.ABC):
  """Base model holding parameters common to all source models.

  Subclasses should define the `model_name` attribute as a `Literal`
  string. This string should match the name of the function that calculates the
  source profile. This is used as an identifier for the model function in the
  source config for Pydantic to "discriminate" against. This should be given a
  unique value for each source model function implementation.

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
    prescribed_values: Tuple of prescribed values for the source, one for each
      affected core profile. Used only when the source is fully prescribed (i.e.
      source.mode == Mode.PRESCRIBED). The default here is a vector of all zeros
      along for all rho and time, and the output vector is along the cell grid.
  """

  mode: runtime_params.Mode = runtime_params.Mode.ZERO
  is_explicit: bool = False
  prescribed_values: tuple[torax_pydantic.TimeVaryingArray, ...] = (
      torax_pydantic.ValidatedDefault(({0: {0: 0, 1: 0}},))
  )

  def build_static_params(self) -> runtime_params.StaticRuntimeParams:
    return runtime_params.StaticRuntimeParams(
        mode=self.mode.value,
        is_explicit=self.is_explicit,
    )

  @abc.abstractmethod
  def build_source(self) -> source_lib.Source:
    """Builds a source object from the model config."""

  @property
  @abc.abstractmethod
  def model_func(self) -> source_lib.SourceProfileFunction:
    """Returns the model function for the source."""

  @abc.abstractmethod
  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> runtime_params.DynamicRuntimeParams:
    """Builds dynamic runtime parameters for the source."""

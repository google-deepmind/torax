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

from collections.abc import Collection
import dataclasses
import enum
from typing import Self

import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import interpolated_param
from torax._src.geometry import geometry
from torax._src.physics import fast_ion as fast_ion_lib

TimeInterpolatedInput = interpolated_param.TimeInterpolatedInput


@enum.unique
class PopulatedSourceOutput(enum.StrEnum):
  """Defines which profile fields on `SourceOutput` a source populates."""

  POWER_DENSITY_I = "power_density_i"
  POWER_DENSITY_E = "power_density_e"
  J_PARALLEL = "j_parallel"
  PARTICLE_SOURCE_E = "particle_source_e"


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SourceOutput:
  """Output of a single Source model evaluation for one time step.

  Attributes:
    power_density_i: Ion power density profile on the cell grid [W/m^3].
    power_density_e: Electron power density profile on the cell grid [W/m^3].
    j_parallel: Parallel current density profile on the cell grid [A/m^2].
    particle_source_e: Electron particle source profile on the cell grid [m^-3
      s^-1].
    fast_ions: Fast ion species distributions produced by this source, if any.
  """

  power_density_i: array_typing.FloatVectorCell | None = None
  power_density_e: array_typing.FloatVectorCell | None = None
  j_parallel: array_typing.FloatVectorCell | None = None
  particle_source_e: array_typing.FloatVectorCell | None = None
  fast_ions: tuple[fast_ion_lib.FastIon, ...] = ()

  @classmethod
  def zeros(
      cls,
      geo: geometry.Geometry,
      populated_source_outputs: Collection[PopulatedSourceOutput] | None = None,
  ) -> Self:
    """Returns a SourceOutput with zero profiles for populated outputs."""
    zero_profile = jnp.zeros(geo.rho_norm.shape)
    if populated_source_outputs is None:
      return cls(
          power_density_i=zero_profile,
          power_density_e=zero_profile,
          j_parallel=zero_profile,
          particle_source_e=zero_profile,
      )
    outputs = set(populated_source_outputs)
    return cls(
        power_density_i=(
            zero_profile
            if PopulatedSourceOutput.POWER_DENSITY_I in outputs
            else None
        ),
        power_density_e=(
            zero_profile
            if PopulatedSourceOutput.POWER_DENSITY_E in outputs
            else None
        ),
        j_parallel=(
            zero_profile
            if PopulatedSourceOutput.J_PARALLEL in outputs
            else None
        ),
        particle_source_e=(
            zero_profile
            if PopulatedSourceOutput.PARTICLE_SOURCE_E in outputs
            else None
        ),
    )


@enum.unique
class Mode(enum.StrEnum):
  """Defines how to compute the source terms for this source/sink."""

  # Source is set to zero always.
  ZERO = "zero"

  # Source values come from a model in code. These terms can be implicit or
  # explicit depending on the model implementation.
  MODEL_BASED = "model_based"

  # Source values come from a pre-determined set of values, that may evolve in
  # time. Currently, this is only supported for sources that have a 1D output
  # along the cell grid or face grid.
  PRESCRIBED = "prescribed"


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams:
  """Runtime params for a single TORAX source."""

  prescribed_values: SourceOutput
  mode: Mode = jax.tree.static()
  is_explicit: bool = jax.tree.static()

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

from collections.abc import Callable, Mapping
import dataclasses
import enum
from typing import Any

import chex
from torax.sources import formula_config


# Sources implement these functions to be able to provide source profiles. The
# SourceConfig also gives a hook for users to provide a custom function.
# Using `Any` instead of the actual argument types below to avoid circular
# dependencies.
SourceProfileFunction = Callable[
    [  # Arguments
        Any,  # config.Config
        Any,  # geometry.Geometry
        Any | None,  # state.CoreProfiles
    ],
    # Returns a JAX array, tuple of arrays, or mapping of arrays.
    chex.ArrayTree,
]


@enum.unique
class SourceType(enum.Enum):
  """Defines how to compute the source terms for this source/sink."""

  # Source is set to zero always. This is an explicit source by definition.
  ZERO = 0

  # Source values come from a model in code. These terms can be implicit or
  # explicit depending on the model implementation.
  MODEL_BASED = 1

  # Source values come from a prescribed (possibly time-dependent) formula that
  # is not dependant on the state of the system. These formulas may be dependent
  # on the config and geometry of the system.
  FORMULA_BASED = 2


@dataclasses.dataclass
class SourceConfig:
  """Configures a single source/sink term.

  This is a RUNTIME config, meaning its values can change from run to run
  without trigerring a recompile. This config defines the runtime config for the
  entire simulation run. The DynamicSourceConfigSlice, which is derived from
  this class, only contains information for a single time step.

  Any compile-time configurations for the Sources should go into the Source
  object's constructor.
  """

  # Defines how the source values are computed (from a model, from a file, etc.)
  source_type: SourceType = SourceType.ZERO

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

  formula: formula_config.FormulaConfig = dataclasses.field(
      default_factory=formula_config.FormulaConfig
  )


# Define helper functions to use as factories in configs below.
# pylint: disable=g-long-lambda
get_model_based_source_config = lambda: SourceConfig(
    source_type=SourceType.MODEL_BASED,
)
get_formula_based_source_config = lambda: SourceConfig(
    source_type=SourceType.FORMULA_BASED,
)
# pylint: enable=g-long-lambda


def get_default_sources_config() -> Mapping[str, SourceConfig]:
  """Returns a mapping of source names to their default runtime configurations.

  This makes an assumption about the names of Source objects used in the
  simulation run, that they match the keys of the dictionary here. If that's not
  the case, callers must modify the dictionary returned here.
  """
  return {
      # Current sources (for psi equation)
      'j_bootstrap': get_model_based_source_config(),
      'jext': get_formula_based_source_config(),
      # Electron density sources/sink (for the ne equation).
      'nbi_particle_source': get_formula_based_source_config(),
      'gas_puff_source': get_formula_based_source_config(),
      'pellet_source': get_formula_based_source_config(),
      # Ion and electron heat sources (for the temp-ion and temp-el eqs).
      'generic_ion_el_heat_source': get_formula_based_source_config(),
      'fusion_heat_source': get_model_based_source_config(),
      'ohmic_heat_source': get_model_based_source_config(),
      # NOTE: For qei_source, the is_explicit field in the config has no effect.
      'qei_source': get_model_based_source_config(),
  }

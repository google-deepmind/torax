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

"""General runtime input parameters used throughout TORAX simulations."""

from __future__ import annotations

import dataclasses

import chex
import pydantic
from torax.config import numerics as numerics_lib
from torax.config import plasma_composition as plasma_composition_lib
from torax.config import profile_conditions as profile_conditions_lib
from torax.torax_pydantic import torax_pydantic


@chex.dataclass
class DynamicGeneralRuntimeParams:
  """General runtime input parameters for the `torax` module."""

  plasma_composition: plasma_composition_lib.DynamicPlasmaComposition
  profile_conditions: profile_conditions_lib.DynamicProfileConditions
  numerics: numerics_lib.DynamicNumerics


class GeneralRuntimeParams(torax_pydantic.BaseModelFrozen):
  """Pydantic model for runtime parameters.

  The `from_dict(...)` method can accept a dictionary defined by
  https://torax.readthedocs.io/en/latest/configuration.html#runtime-params.

  Attributes:
    profile_conditions: Pydantic model for the profile conditions.
    numerics: Pydantic model for the numerics.
    plasma_composition: Pydantic model for the plasma composition.
    output_dir: File directory where the simulation outputs will be saved. If
      not provided, this will default to /tmp/torax_results_<YYYYMMDD_HHMMSS>/.
  """

  profile_conditions: profile_conditions_lib.ProfileConditions = pydantic.Field(
      default_factory=profile_conditions_lib.ProfileConditions
  )
  numerics: numerics_lib.Numerics = pydantic.Field(
      default_factory=numerics_lib.Numerics
  )
  plasma_composition: plasma_composition_lib.PlasmaComposition = pydantic.Field(
      default_factory=plasma_composition_lib.PlasmaComposition
  )
  output_dir: str | None = None

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> DynamicGeneralRuntimeParams:
    return DynamicGeneralRuntimeParams(
        profile_conditions=self.profile_conditions.build_dynamic_params(t),
        numerics=self.numerics.build_dynamic_params(t),
        plasma_composition=self.plasma_composition.build_dynamic_params(t),
    )


@dataclasses.dataclass
class FileRestart:
  # Filename to load initial state from.
  filename: str
  # Time in state file at which to load from.
  time: float
  # Toggle loading initial state from file or not.
  do_restart: bool
  stitch: bool

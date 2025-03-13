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
from torax.config import base
from torax.config import numerics as numerics_lib
from torax.config import plasma_composition as plasma_composition_lib
from torax.config import profile_conditions as profile_conditions_lib
from torax.torax_pydantic import torax_pydantic
from typing_extensions import override


class RuntimeParams(torax_pydantic.BaseModelFrozen):
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

  profile_conditions: profile_conditions_lib.ProfileConditionsPydantic
  numerics: numerics_lib.NumericsPydantic
  plasma_composition: plasma_composition_lib.PlasmaCompositionPydantic = (
      pydantic.Field(
          default_factory=plasma_composition_lib.PlasmaCompositionPydantic
      )
  )
  output_dir: str | None = None


@dataclasses.dataclass
class FileRestart:
  # Filename to load initial state from.
  filename: str
  # Time in state file at which to load from.
  time: float
  # Toggle loading initial state from file or not.
  do_restart: bool
  stitch: bool


# NOMUTANTS -- It's expected for the tests to pass with different defaults.
@chex.dataclass
class GeneralRuntimeParams(base.RuntimeParametersConfig):
  """General runtime input parameters for the `torax` module."""

  plasma_composition: plasma_composition_lib.PlasmaComposition = (
      dataclasses.field(
          default_factory=plasma_composition_lib.PlasmaComposition
      )
  )
  profile_conditions: profile_conditions_lib.ProfileConditions = (
      dataclasses.field(
          default_factory=profile_conditions_lib.ProfileConditions
      )
  )
  numerics: numerics_lib.Numerics = dataclasses.field(
      default_factory=numerics_lib.Numerics
  )

  # 'File directory where the simulation outputs will be saved. If not '
  # 'provided, this will default to /tmp/torax_results_<YYYYMMDD_HHMMSS>/.',
  output_dir: str | None = None

  def make_provider(
      self, torax_mesh: torax_pydantic.Grid1D | None = None
  ) -> GeneralRuntimeParamsProvider:
    return GeneralRuntimeParamsProvider(**self.get_provider_kwargs(torax_mesh))


@chex.dataclass
class GeneralRuntimeParamsProvider(
    base.RuntimeParametersProvider['DynamicGeneralRuntimeParams']
):
  """General runtime input parameters for the `torax` module."""

  runtime_params_config: GeneralRuntimeParams
  plasma_composition: plasma_composition_lib.PlasmaCompositionProvider
  profile_conditions: profile_conditions_lib.ProfileConditionsProvider
  numerics: numerics_lib.NumericsProvider

  @override
  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> DynamicGeneralRuntimeParams:
    dynamic_params_kwargs = self.get_dynamic_params_kwargs(t)
    # TODO(b/362436011)
    del dynamic_params_kwargs['output_dir']
    return DynamicGeneralRuntimeParams(**dynamic_params_kwargs)


@chex.dataclass
class DynamicGeneralRuntimeParams:
  """General runtime input parameters for the `torax` module."""

  plasma_composition: plasma_composition_lib.DynamicPlasmaComposition
  profile_conditions: profile_conditions_lib.DynamicProfileConditions
  numerics: numerics_lib.DynamicNumerics

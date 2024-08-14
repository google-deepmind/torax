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

from collections.abc import Mapping
import dataclasses

import chex
from torax import geometry
from torax import interpolated_param
from torax.config import base
from torax.config import numerics as numerics_lib
from torax.config import plasma_composition as plasma_composition_lib
from torax.config import profile_conditions as profile_conditions_lib
from typing_extensions import override


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

  # pylint: enable=invalid-name

  def _sanity_check_profile_boundary_conditions(
      self, var: interpolated_param.InterpolatedVarTimeRhoInput, var_name: str,
  ):
    """Check that the profile is defined at rho=1.0."""
    values = var

    for time in values:
      if isinstance(values[time], Mapping):
        if 1.0 not in values[time]:
          raise ValueError(
              f'As no right boundary condition was set for {var_name}, the'
              f' profile for {var_name} must include a value at rho=1.0 for'
              ' every provided time.'
          )
      else:
        raise ValueError(
            f'As no right boundary condition was set for {var_name}, the '
            f'profile for {var_name} must include a rho=1.0 boundary condition.'
        )

  def sanity_check(self) -> None:
    """Checks that various configuration parameters are valid."""
    # TODO(b/330172917) do more extensive config parameter sanity checking

    # These are floats, not jax types, so we can use direct asserts.
    assert self.numerics.dtmult > 0.0
    assert isinstance(
        self.plasma_composition, plasma_composition_lib.PlasmaComposition
    )
    assert isinstance(self.numerics, numerics_lib.Numerics)
    if self.profile_conditions.Ti_bound_right is None:
      self._sanity_check_profile_boundary_conditions(
          self.profile_conditions.Ti,
          'Ti',
      )
    if self.profile_conditions.Te_bound_right is None:
      self._sanity_check_profile_boundary_conditions(
          self.profile_conditions.Te,
          'Te',
      )
    if self.profile_conditions.ne_bound_right is None:
      self._sanity_check_profile_boundary_conditions(
          self.profile_conditions.ne,
          'ne',
      )

  def __post_init__(self):
    self.sanity_check()

  def make_provider(
      self, torax_mesh: geometry.Grid1D | None = None
  ) -> GeneralRuntimeParamsProvider:
    return GeneralRuntimeParamsProvider(
        runtime_params_config=self,
        plasma_composition=self.plasma_composition.make_provider(torax_mesh),
        profile_conditions=self.profile_conditions.make_provider(torax_mesh),
        numerics=self.numerics.make_provider(torax_mesh),
    )


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
    return DynamicGeneralRuntimeParams(
        dynamic_plasma_composition=self.plasma_composition.build_dynamic_params(
            t
        ),
        dynamic_profile_conditions=self.profile_conditions.build_dynamic_params(
            t
        ),
        dynamic_numerics=self.numerics.build_dynamic_params(t),
    )


@chex.dataclass
class DynamicGeneralRuntimeParams:
  """General runtime input parameters for the `torax` module."""
  dynamic_plasma_composition: plasma_composition_lib.DynamicPlasmaComposition
  dynamic_profile_conditions: profile_conditions_lib.DynamicProfileConditions
  dynamic_numerics: numerics_lib.DynamicNumerics


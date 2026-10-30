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
"""Impurity radiation heat sink for electron heat equation based on constant fraction of total power density."""

import dataclasses
from typing import Annotated, Literal, Self
import chex
import jax
import jax.numpy as jnp
import pydantic
from torax._src import array_typing
from torax._src import math_utils
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.sources import base
from torax._src.sources import bremsstrahlung_heat_sink
from torax._src.sources import cyclotron_radiation_heat_sink
from torax._src.sources import runtime_params as sources_runtime_params_lib
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.sources import source_state as source_state_lib
from torax._src.sources.impurity_radiation_heat_sink import impurity_radiation_heat_sink
from torax._src.torax_pydantic import torax_pydantic


_HEAT_SINK_SOURCE_IDS = frozenset({
    bremsstrahlung_heat_sink.BremsstrahlungHeatSink.SOURCE_ID,
    cyclotron_radiation_heat_sink.CyclotronRadiationHeatSink.SOURCE_ID,
    impurity_radiation_heat_sink.ImpurityRadiationHeatSink.SOURCE_ID,
})


# pylint: disable=invalid-name
@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class ConstantFractionImpurityRadiationHeatSink(
    impurity_radiation_heat_sink.ImpurityRadiationHeatSink
):
  """Impurity radiation heat sink with constant fraction of total power."""

  def _get_model_value(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      calculated_source_profiles: source_profiles_lib.SourceProfiles | None,
      conductivity: conductivity_base.Conductivity | None,
      source_state: source_state_lib.SourceState,
  ) -> tuple[source_profiles_lib.SourceOutput, source_state_lib.SourceState]:
    """Calculates constant fraction impurity radiation heat sink profile.

    Does not modify source_state; returns it unchanged.

    Args:
      runtime_params: Runtime parameters.
      geo: Geometry of the torus.
      core_profiles: Core plasma profiles.
      calculated_source_profiles: Source profiles calculated so far.
      conductivity: Conductivity model output.
      source_state: Source state carried over from the previous timestep. Unused
        by this model and returned unchanged.

    Returns:
      A tuple of (SourceOutput(profiles=(electron_heat_profile,)),
      source_state).
    """
    del core_profiles, conductivity
    source_params = runtime_params.sources[self.SOURCE_ID]
    assert isinstance(source_params, RuntimeParams)

    if calculated_source_profiles is None:
      raise ValueError(
          'calculated_source_profiles is a required argument for'
          ' ImpurityRadiationConstantFraction._get_model_value. This can occur'
          ' if this source function is used in an explicit source.'
      )

    # Based on source_models.sum_sources_T_e and source_models.calc_and_sum
    # sources_psi, but only summing over heating *input* sources
    # (Pohm + Paux + Palpha + ...) and summing over *both* ion +
    # electron heating

    source_profiles = jnp.zeros_like(geo.rho)
    for name, profile in calculated_source_profiles.T_e.items():
      if name not in _HEAT_SINK_SOURCE_IDS:
        source_profiles += profile
    for name, profile in calculated_source_profiles.T_i.items():
      if name not in _HEAT_SINK_SOURCE_IDS:
        source_profiles += profile

    Q_total_in = source_profiles
    P_total_in = math_utils.volume_integration(Q_total_in, geo)

    # Calculate the heat sink as a fraction of the total power input.
    P_rad = (
        source_params.fraction_P_heating
        * P_total_in
        / geo.volume_face[-1]
        * jnp.ones_like(geo.rho)
    )
    return source_profiles_lib.SourceOutput(profiles=(-P_rad,)), source_state


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(sources_runtime_params_lib.RuntimeParams):
  fraction_P_heating: array_typing.FloatScalar


class ImpurityRadiationHeatSinkConstantFractionConfig(base.SourceConfigBase):
  """Configuration for the ImpurityRadiationHeatSink.

  Attributes:
    fraction_P_heating: Fraction of total power density to be absorbed by the
      impurity. Required when mode is 'model_based'.
  """

  model_name: Annotated[
      Literal['P_in_scaled_flat_profile'], torax_pydantic.JAX_STATIC
  ] = 'P_in_scaled_flat_profile'
  fraction_P_heating: torax_pydantic.UnitIntervalTimeVaryingScalar | None = None

  @pydantic.model_validator(mode='after')
  def _check_fields(self) -> Self:
    if (
        self.mode == sources_runtime_params_lib.Mode.MODEL_BASED
        and self.fraction_P_heating is None
    ):
      raise ValueError(
          "fraction_P_heating must be provided when mode is 'model_based'."
      )
    return self

  def build_runtime_params(
      self,
      t: chex.Numeric,
  ) -> RuntimeParams:
    base_params = super().build_runtime_params(t)
    return RuntimeParams(
        **vars(base_params),
        fraction_P_heating=(
            0.0
            if self.fraction_P_heating is None
            else self.fraction_P_heating.get_value(t)
        ),
    )

  def build_source(self) -> ConstantFractionImpurityRadiationHeatSink:
    return ConstantFractionImpurityRadiationHeatSink()

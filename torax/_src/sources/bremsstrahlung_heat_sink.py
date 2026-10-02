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

# pylint: disable=invalid-name

"""Bremsstrahlung heat sink for electron heat equation.."""
import dataclasses
from typing import Annotated, ClassVar, Final, Literal
import chex
import jax
from jax import numpy as jnp
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.sources import base
from torax._src.sources import runtime_params as sources_runtime_params_lib
from torax._src.sources import source
from torax._src.sources import source_profiles
from torax._src.torax_pydantic import torax_pydantic

# Default value for the model to be used for the Bremsstrahlung heat sink.
# This is also used as an identifier for the model in the default source config
# for Pydantic to "discriminate" against.
DEFAULT_MODEL_NAME: Final[str] = 'wesson'


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(sources_runtime_params_lib.RuntimeParams):
  use_relativistic_correction: bool
  exclude_impurity_bremsstrahlung: bool


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class BremsstrahlungHeatSink(source.Source):
  """Bremsstrahlung heat sink for electron heat equation."""

  SOURCE_ID: ClassVar[str] = 'bremsstrahlung'
  AFFECTED_CORE_PROFILES: ClassVar[tuple[source.AffectedCoreProfile, ...]] = (
      source.AffectedCoreProfile.TEMP_EL,
  )

  def _get_model_value(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      calculated_source_profiles: source_profiles.SourceProfiles | None = None,
      conductivity: conductivity_base.Conductivity | None = None,
  ) -> tuple[source.SourceProfileElement, ...]:
    """Calculates Bremsstrahlung radiation power sink profile.

    Uses the model from Wesson, John, and David J. Campbell. Tokamaks. Vol. 149.
    An optional correction for relativistic effects from Stott PPCF 2005 can be
    enabled with the flag "use_relativistic_correction".

    Args:
      runtime_params: A slice of runtime parameters.
      geo: Magnetic geometry.
      core_profiles: Core plasma profiles.
      calculated_source_profiles: Unused.
      conductivity: Unused.

    Returns:
      Tuple containing the Bremsstrahlung radiation power sink profile [W/m^3].
    """
    del calculated_source_profiles, conductivity
    source_params = runtime_params.sources[self.SOURCE_ID]
    assert isinstance(source_params, RuntimeParams)

    n_e20 = core_profiles.n_e.face_value() / 1e20
    T_e_kev = core_profiles.T_e.face_value()

    # When exclude_impurity_bremsstrahlung is True, use the main-ion-only
    # contribution to Z_eff: Z_eff_main = n_i * Z_i^2 / n_e.
    Z_eff_face = jnp.where(
        source_params.exclude_impurity_bremsstrahlung,
        core_profiles.n_i.face_value() * core_profiles.Z_i_face**2
        / core_profiles.n_e.face_value(),
        core_profiles.Z_eff_face,
    )

    P_brem_profile_face: jax.Array = (
        5.35e-3 * Z_eff_face * n_e20**2 * jnp.sqrt(T_e_kev)
    )  # MW/m^3

    def calc_relativistic_correction() -> chex.Array:
      # Apply the Stott relativistic correction.
      Tm = 511.0  # m_e * c**2 in keV
      correction = (1.0 + 2.0 * T_e_kev / Tm) * (
          1.0
          + (2.0 / Z_eff_face) * (1.0 - 1.0 / (1.0 + T_e_kev / Tm))
      )
      return jnp.asarray(correction)

    # In MW/m^3
    P_brem_profile_face = jnp.where(
        source_params.use_relativistic_correction,
        P_brem_profile_face * calc_relativistic_correction(),
        P_brem_profile_face,
    )

    # In W/m^3
    P_brem_profile_cell = geometry.face_to_cell(P_brem_profile_face) * 1e6

    # As a sink, the power is negative.
    return (-1.0 * P_brem_profile_cell,)


class BremsstrahlungHeatSinkConfig(base.SourceConfigBase):
  """Bremsstrahlung heat sink for electron heat equation.

  Attributes:
    use_relativistic_correction: Whether to use relativistic correction.
    exclude_impurity_bremsstrahlung: If True, only include main-ion
      bremsstrahlung. Automatically set to True by the Sources pydantic
      validator when the Mavrin impurity radiation model is also active.
  """

  model_name: Annotated[Literal['wesson'], torax_pydantic.JAX_STATIC] = 'wesson'
  use_relativistic_correction: bool = False
  exclude_impurity_bremsstrahlung: bool = False

  def build_runtime_params(
      self,
      t: chex.Numeric,
  ) -> 'RuntimeParams':
    return RuntimeParams(
        **dataclasses.asdict(super().build_runtime_params(t)),
        use_relativistic_correction=self.use_relativistic_correction,
        exclude_impurity_bremsstrahlung=self.exclude_impurity_bremsstrahlung,
    )

  def build_source(self) -> BremsstrahlungHeatSink:
    return BremsstrahlungHeatSink()

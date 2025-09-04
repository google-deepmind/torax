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

"""Collisional ion-electron heat source."""
import dataclasses
from typing import Annotated, ClassVar
import chex
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.physics import collisions
from torax._src.sources import base
from torax._src.sources import runtime_params as runtime_params_lib
from torax._src.sources import source
from torax._src.sources import source_profiles
from torax._src.torax_pydantic import torax_pydantic


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(runtime_params_lib.RuntimeParams):
  Qei_multiplier: float


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class QeiSource(source.Source):
  """Collisional ion-electron heat source.

  This is a special-case source because it can provide both implicit and
  explicit terms in our solver. See sim.py for how this is used.
  """

  SOURCE_NAME: ClassVar[str] = 'ei_exchange'

  @property
  def source_name(self) -> str:
    return self.SOURCE_NAME

  @property
  def affected_core_profiles(self) -> tuple[source.AffectedCoreProfile, ...]:
    return (
        source.AffectedCoreProfile.TEMP_ION,
        source.AffectedCoreProfile.TEMP_EL,
    )

  def get_qei(
      self,
      runtime_params: runtime_params_slice.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> source_profiles.QeiInfo:
    """Computes the value of the source."""
    return jax.lax.cond(
        runtime_params.sources[self.source_name].mode
        == runtime_params_lib.Mode.MODEL_BASED,
        lambda: _model_based_qei(
            runtime_params,
            geo,
            core_profiles,
        ),
        lambda: source_profiles.QeiInfo.zeros(geo),
    )

  def get_value(
      self,
      runtime_params: runtime_params_slice.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      calculated_source_profiles: source_profiles.SourceProfiles | None,
      conductivity: conductivity_base.Conductivity | None,
  ) -> tuple[array_typing.FloatVectorCell, ...]:
    raise NotImplementedError('Call get_qei() instead.')

  def get_source_profile_for_affected_core_profile(
      self,
      profile: tuple[array_typing.Array, ...],
      affected_mesh_state: int,
      geo: geometry.Geometry,
  ) -> jax.Array:
    raise NotImplementedError('This method is not valid for QeiSource.')


def _model_based_qei(
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
) -> source_profiles.QeiInfo:
  """Computes Qei via the coll_exchange model."""
  source_params = runtime_params.sources[QeiSource.SOURCE_NAME]
  assert isinstance(source_params, RuntimeParams)
  zeros = jnp.zeros_like(geo.rho_norm)
  qei_coef = collisions.coll_exchange(
      core_profiles=core_profiles,
      Qei_multiplier=source_params.Qei_multiplier,
  )
  implicit_ii = -qei_coef
  implicit_ee = -qei_coef

  if (
      # if only a single heat equation is being evolved
      (
          runtime_params.numerics.evolve_ion_heat
          and not runtime_params.numerics.evolve_electron_heat
      )
      or (
          runtime_params.numerics.evolve_electron_heat
          and not runtime_params.numerics.evolve_ion_heat
      )
  ):
    explicit_i = qei_coef * core_profiles.T_e.value
    explicit_e = qei_coef * core_profiles.T_i.value
    implicit_ie = zeros
    implicit_ei = zeros
  else:
    explicit_i = zeros
    explicit_e = zeros
    implicit_ie = qei_coef
    implicit_ei = qei_coef
  return source_profiles.QeiInfo(
      qei_coef=qei_coef,
      implicit_ii=implicit_ii,
      explicit_i=explicit_i,
      implicit_ee=implicit_ee,
      explicit_e=explicit_e,
      implicit_ie=implicit_ie,
      implicit_ei=implicit_ei,
  )


class QeiSourceConfig(base.SourceModelBase):
  """Configuration for the QeiSource.

  Attributes:
    Qei_multiplier: multiplier for ion-electron heat exchange term for
      sensitivity testing
  """

  Qei_multiplier: float = 1.0
  mode: Annotated[runtime_params_lib.Mode, torax_pydantic.JAX_STATIC] = (
      runtime_params_lib.Mode.MODEL_BASED
  )

  @property
  def model_func(self) -> None:
    return None

  def build_runtime_params(
      self,
      t: chex.Numeric,
  ) -> RuntimeParams:
    return RuntimeParams(
        prescribed_values=tuple(
            [v.get_value(t) for v in self.prescribed_values]
        ),
        mode=self.mode,
        is_explicit=self.is_explicit,
        Qei_multiplier=self.Qei_multiplier,
    )

  def build_source(self) -> QeiSource:
    return QeiSource(model_func=self.model_func)

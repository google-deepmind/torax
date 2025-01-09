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

"""External current source profile."""

from __future__ import annotations

import dataclasses
from typing import ClassVar, Optional

import chex
import jax
from jax import numpy as jnp
from jax.scipy import integrate
import jaxtyping as jt
from torax import array_typing
from torax import interpolated_param
from torax import jax_utils
from torax import state
from torax.config import base
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source
from typing_extensions import override
# pylint: disable=invalid-name


@dataclasses.dataclass(kw_only=True)
class RuntimeParams(runtime_params_lib.RuntimeParams):
  """Runtime parameters for the external current source."""

  # total "external" current in MA. Used if use_absolute_current=True.
  Iext: runtime_params_lib.TimeInterpolatedInput = 3.0
  # total "external" current fraction. Used if use_absolute_current=False.
  fext: runtime_params_lib.TimeInterpolatedInput = 0.2
  # width of "external" Gaussian current profile
  wext: runtime_params_lib.TimeInterpolatedInput = 0.05
  # normalized radius of "external" Gaussian current profile
  rext: runtime_params_lib.TimeInterpolatedInput = 0.4

  # Toggles if external current is provided absolutely or as a fraction of Ip.
  use_absolute_current: bool = False
  mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED

  @property
  def grid_type(self) -> base.GridType:
    return base.GridType.FACE

  def make_provider(
      self,
      torax_mesh: geometry.Grid1D | None = None,
  ) -> RuntimeParamsProvider:
    return RuntimeParamsProvider(**self.get_provider_kwargs(torax_mesh))


@chex.dataclass
class RuntimeParamsProvider(runtime_params_lib.RuntimeParamsProvider):
  """Provides runtime parameters for a given time and geometry."""

  runtime_params_config: RuntimeParams
  Iext: interpolated_param.InterpolatedVarSingleAxis
  fext: interpolated_param.InterpolatedVarSingleAxis
  wext: interpolated_param.InterpolatedVarSingleAxis
  rext: interpolated_param.InterpolatedVarSingleAxis

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(**self.get_dynamic_params_kwargs(t))


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  """Dynamic runtime parameters for the external current source."""

  Iext: array_typing.ScalarFloat
  fext: array_typing.ScalarFloat
  wext: array_typing.ScalarFloat
  rext: array_typing.ScalarFloat
  use_absolute_current: bool

  def sanity_check(self):
    """Checks that all parameters are valid."""
    jax_utils.error_if_negative(self.wext, 'wext')

  def __post_init__(self):
    self.sanity_check()


_trapz = integrate.trapezoid


# pytype bug: does not treat 'source_models.SourceModels' as a forward reference
# pytype: disable=name-error
def calculate_generic_current_face(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    source_name: str,
    unused_state: state.CoreProfiles | None = None,
    unused_source_models: Optional['source_models.SourceModels'] = None,
) -> jax.Array:
  """Calculates the external current density profiles.

  Args:
    static_runtime_params_slice: Static runtime parameters.
    dynamic_runtime_params_slice: Parameter configuration at present timestep.
    geo: Tokamak geometry.
    source_name: Name of the source.
    unused_state: State argument not used in this function but is present to
      adhere to the source API.
    unused_source_models: Source models argument not used in this function but
      is present to adhere to the source API.

  Returns:
    External current density profile along the face grid.
  """
  del (
      static_runtime_params_slice,
      unused_state,
      unused_source_models,
  )  # Unused.
  dynamic_source_runtime_params = dynamic_runtime_params_slice.sources[
      source_name
  ]
  # pytype: enable=name-error
  assert isinstance(dynamic_source_runtime_params, DynamicRuntimeParams)
  Iext = _calculate_Iext(
      dynamic_runtime_params_slice,
      dynamic_source_runtime_params,
  )
  # form of external current on face grid
  generic_current_form_face = jnp.exp(
      -((geo.rho_face_norm - dynamic_source_runtime_params.rext) ** 2)
      / (2 * dynamic_source_runtime_params.wext**2)
  )

  Cext = (
      Iext
      * 1e6
      / _trapz(generic_current_form_face * geo.spr_face, geo.rho_face_norm)
  )

  generic_current_face = (
      Cext * generic_current_form_face
  )  # external current profile
  return generic_current_face


def _calculate_Iext(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    dynamic_source_runtime_params: DynamicRuntimeParams,
) -> chex.Numeric:
  """Calculates the total value of external current."""
  return jnp.where(
      dynamic_source_runtime_params.use_absolute_current,
      dynamic_source_runtime_params.Iext,
      (
          dynamic_runtime_params_slice.profile_conditions.Ip_tot
          * dynamic_source_runtime_params.fext
      ),
  )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class GenericCurrentSource(source.Source):
  """A generic current density source profile."""

  SOURCE_NAME: ClassVar[str] = 'generic_current_source'
  DEFAULT_MODEL_FUNCTION_NAME: ClassVar[str] = 'calc_generic_current_face'
  model_func: source.SourceProfileFunction = calculate_generic_current_face

  @property
  def source_name(self) -> str:
    return self.SOURCE_NAME

  @property
  def affected_core_profiles(self) -> tuple[source.AffectedCoreProfile, ...]:
    return (source.AffectedCoreProfile.PSI,)

  @property
  def output_shape_getter(self) -> source.SourceOutputShapeFunction:
    return source.ProfileType.FACE.get_profile_shape

  @override
  def get_source_profile_for_affected_core_profile(
      self,
      profile: jt.Float[jt.Array, 'rhon_face'],
      affected_core_profile: int,
      geo: geometry.Geometry,
  ) -> jt.Float[jt.Array, 'rhon']:
    return jnp.where(
        affected_core_profile in self.affected_core_profiles_ints,
        # Source profiles are always on cell grid so cast to cell grid.
        geometry.face_to_cell(profile),
        jnp.zeros_like(geo.rho),
    )

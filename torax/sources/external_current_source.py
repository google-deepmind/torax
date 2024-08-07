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
from typing import Optional

import chex
import jax
from jax import numpy as jnp
from jax.scipy import integrate
from torax import geometry
from torax import jax_utils
from torax import state
from torax.config import config_args
from torax.config import runtime_params_slice
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source


# pylint: disable=invalid-name


@dataclasses.dataclass(kw_only=True)
class RuntimeParams(runtime_params_lib.RuntimeParams):
  """Runtime parameters for the external current source."""

  # total "external" current in MA. Used if use_absolute_jext=True.
  Iext: runtime_params_lib.TimeInterpolated = 3.0
  # total "external" current fraction. Used if use_absolute_jext=False.
  fext: runtime_params_lib.TimeInterpolated = 0.2
  # width of "external" Gaussian current profile
  wext: runtime_params_lib.TimeInterpolated = 0.05
  # normalized radius of "external" Gaussian current profile
  rext: runtime_params_lib.TimeInterpolated = 0.4

  # Toggles if external current is provided absolutely or as a fraction of Ip.
  use_absolute_jext: bool = False

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(
        **config_args.get_init_kwargs(
            input_config=self,
            output_type=DynamicRuntimeParams,
            t=t,
        )
    )


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  """Dynamic runtime parameters for the external current source."""

  Iext: float
  fext: float
  wext: float
  rext: float
  use_absolute_jext: bool

  def sanity_check(self):
    """Checks that all parameters are valid."""
    # Using object.__setattr__ to get around the fact this is a frozen dataclass
    object.__setattr__(
        self, 'wext', jax_utils.error_if_negative(self.wext, 'wext')
    )

  def __post_init__(self):
    self.sanity_check()


_trapz = integrate.trapezoid


# pytype bug: does not treat 'source_models.SourceModels' as a forward reference
def _calculate_jext_face(  # pytype: disable=name-error
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    dynamic_source_runtime_params: runtime_params_lib.DynamicRuntimeParams,
    geo: geometry.Geometry,
    unused_state: state.CoreProfiles | None = None,
    unused_source_models: Optional['source_models.SourceModels'] = None,
) -> jax.Array:
  """Calculates the external current density profiles.

  Args:
    dynamic_runtime_params_slice: Parameter configuration at present timestep.
    dynamic_source_runtime_params: Source-specific parameters at the present
      timestep.
    geo: Tokamak geometry.
    unused_state: State argument not used in this function but is present to
      adhere to the source API.

  Returns:
    External current density profile along the face grid.
  """
  assert isinstance(dynamic_source_runtime_params, DynamicRuntimeParams)
  Iext = _calculate_Iext(
      dynamic_runtime_params_slice,
      dynamic_source_runtime_params,
  )
  # form of external current on face grid
  jextform_face = jnp.exp(
      -((geo.rho_face_norm - dynamic_source_runtime_params.rext) ** 2)
      / (2 * dynamic_source_runtime_params.wext**2)
  )

  Cext = Iext * 1e6 / _trapz(jextform_face * geo.spr_face, geo.rho_face_norm)

  jext_face = Cext * jextform_face  # external current profile
  return jext_face


# pytype bug: does not treat 'source_models.SourceModels' as a forward reference
def _calculate_jext_hires(  # pytype: disable=name-error
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    dynamic_source_runtime_params: runtime_params_lib.DynamicRuntimeParams,
    geo: geometry.Geometry,
    unused_state: state.CoreProfiles | None = None,
    unused_source_models: Optional['source_models.SourceModels'] = None,
) -> jax.Array:
  """Calculates the external current density profile along the hires grid.

  Args:
    dynamic_runtime_params_slice: Parameter configuration at present timestep.
    dynamic_source_runtime_params: Source-specific parameters at the present
      timestep.
    geo: Tokamak geometry.
    unused_state: State argument not used in this function but is present to
      adhere to the source API.

  Returns:
    External current density profile along the hires cell grid.
  """
  assert isinstance(dynamic_source_runtime_params, DynamicRuntimeParams)
  Iext = _calculate_Iext(
      dynamic_runtime_params_slice,
      dynamic_source_runtime_params,
  )
  # calculate "External" current profile (e.g. ECCD)
  # form of external current on cell grid
  jextform_hires = jnp.exp(
      -((geo.rho_hires_norm - dynamic_source_runtime_params.rext) ** 2)
      / (2 * dynamic_source_runtime_params.wext**2)
  )
  Cext_hires = (
      Iext * 1e6 / _trapz(jextform_hires * geo.spr_hires, geo.rho_hires_norm)
  )
  # External current profile on cell grid
  jext_hires = Cext_hires * jextform_hires
  return jext_hires


def _calculate_Iext(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    dynamic_source_runtime_params: DynamicRuntimeParams,
) -> chex.Numeric:
  """Calculates the total value of external current."""
  return jnp.where(
      dynamic_source_runtime_params.use_absolute_jext,
      dynamic_source_runtime_params.Iext,
      (
          dynamic_runtime_params_slice.profile_conditions.Ip
          * dynamic_source_runtime_params.fext
      ),
  )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ExternalCurrentSource(source.Source):
  """External current density source profile."""

  # The ExternalCurrentSource returns a profile on the face grid, unlike most
  # other sources.
  output_shape_getter: source.SourceOutputShapeFunction = (
      source.ProfileType.FACE.get_profile_shape
  )

  supported_types: tuple[runtime_params_lib.Mode, ...] = (
      runtime_params_lib.Mode.ZERO,
      runtime_params_lib.Mode.FORMULA_BASED,
  )

  # Don't include affected_core_profiles in the __init__ arguments.
  # "Freeze" this param.
  affected_core_profiles: tuple[source.AffectedCoreProfile, ...] = (
      dataclasses.field(
          init=False,
          default_factory=lambda: (source.AffectedCoreProfile.PSI,),
      )
  )

  formula: source.SourceProfileFunction = _calculate_jext_face
  hires_formula: source.SourceProfileFunction = _calculate_jext_hires

  def jext_hires(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      dynamic_source_runtime_params: runtime_params_lib.DynamicRuntimeParams,
      geo: geometry.Geometry,
  ) -> jax.Array:
    """Return the external current density profile along the hires cell grid."""
    assert isinstance(dynamic_source_runtime_params, DynamicRuntimeParams)
    self.check_mode(dynamic_source_runtime_params.mode)
    return source.get_source_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        dynamic_source_runtime_params=dynamic_source_runtime_params,
        geo=geo,
        core_profiles=None,
        # There is no model for this source.
        model_func=(
            lambda _0, _1, _2, _3, _4: jnp.zeros_like(geo.rho_hires_norm)
        ),
        formula=self.hires_formula,
        output_shape=geo.rho_hires_norm.shape,
        source_models=getattr(self, 'source_models', None),
    )


ExternalCurrentSourceBuilder = source.make_source_builder(
    ExternalCurrentSource, runtime_params_type=RuntimeParams
)

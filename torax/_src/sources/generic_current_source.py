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

import dataclasses
import logging
from typing import Annotated, ClassVar, Literal, Self

import chex
import jax
from jax import numpy as jnp
import pydantic
from torax._src import array_typing
from torax._src import math_utils
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.physics import psi_calculations
from torax._src.sources import base as source_base
from torax._src.sources import runtime_params as sources_runtime_params_lib
from torax._src.sources import source
from torax._src.sources import source_profiles
from torax._src.sources import source_state as source_state_lib
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(sources_runtime_params_lib.RuntimeParams):
  """Runtime parameters for the external current source."""

  I_generic: array_typing.FloatScalar
  fraction_of_total_current: array_typing.FloatScalar
  gaussian_width: array_typing.FloatScalar
  gaussian_location: array_typing.FloatScalar
  use_absolute_current: bool


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class GenericCurrentSource(source.Source):
  """A generic current density source profile."""

  SOURCE_ID: ClassVar[str] = 'generic_current'
  AFFECTED_CORE_PROFILES: ClassVar[tuple[source.AffectedCoreProfile, ...]] = (
      source.AffectedCoreProfile.PSI,
  )

  def _get_model_value(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      calculated_source_profiles: source_profiles.SourceProfiles | None,
      conductivity: conductivity_base.Conductivity | None,
      source_state: source_state_lib.SourceState,
  ) -> tuple[source_profiles.SourceOutput, source_state_lib.SourceState]:
    """Calculates external parallel current density profile on the cell grid.

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
      A tuple of (SourceOutput(profiles=(psi_profile,)), source_state).
    """
    del core_profiles, calculated_source_profiles, conductivity
    source_params = runtime_params.sources[self.SOURCE_ID]
    assert isinstance(source_params, RuntimeParams)

    I_generic = jnp.where(
        source_params.use_absolute_current,
        source_params.I_generic,
        (
            runtime_params.profile_conditions.I_p
            * source_params.fraction_of_total_current
        ),
    )
    # form of external current on cell grid
    generic_current_form = jnp.exp(
        -((geo.rho_norm - source_params.gaussian_location) ** 2)
        / (2 * source_params.gaussian_width**2)
    )

    Cext = I_generic / math_utils.area_integration(generic_current_form, geo)
    j_tor = Cext * generic_current_form

    j_parallel = psi_calculations.j_toroidal_to_j_parallel(
        j_tor, geo, runtime_params.numerics.min_rho_norm
    )
    return source_profiles.SourceOutput(profiles=(j_parallel,)), source_state


class GenericCurrentSourceConfig(source_base.SourceConfigBase):
  """Configuration for the GenericCurrentSource.

  Attributes:
    I_generic: total "external" current in A. Required when mode is
      'model_based' and use_absolute_current=True.
    fraction_of_total_current: total "external" current fraction. Required when
      mode is 'model_based' and use_absolute_current=False.
    gaussian_width: width of "external" Gaussian current profile. Required when
      mode is 'model_based'.
    gaussian_location: normalized radius of "external" Gaussian current profile.
      Required when mode is 'model_based'.
    use_absolute_current: Toggles if external current is provided absolutely or
      as a fraction of I_p.
  """

  model_name: Annotated[Literal['gaussian'], torax_pydantic.JAX_STATIC] = (
      'gaussian'
  )
  I_generic: torax_pydantic.TimeVaryingScalar | None = None
  fraction_of_total_current: (
      torax_pydantic.UnitIntervalTimeVaryingScalar | None
  ) = None
  gaussian_width: torax_pydantic.PositiveTimeVaryingScalar | None = None
  gaussian_location: torax_pydantic.UnitIntervalTimeVaryingScalar | None = None
  use_absolute_current: bool = False

  @pydantic.model_validator(mode='after')
  def _check_fields(self) -> Self:
    if self.mode == sources_runtime_params_lib.Mode.MODEL_BASED:
      if self.use_absolute_current:
        if self.I_generic is None:
          raise ValueError(
              "I_generic must be provided when mode is 'model_based' and"
              ' use_absolute_current is True.'
          )
        if self.fraction_of_total_current is not None:
          logging.warning(
              'fraction_of_total_current is set but will be ignored because'
              ' use_absolute_current is True.'
          )
      else:
        if self.fraction_of_total_current is None:
          raise ValueError(
              'fraction_of_total_current must be provided when mode is'
              " 'model_based' and use_absolute_current is False."
          )
        if self.I_generic is not None:
          logging.warning(
              'I_generic is set but will be ignored because'
              ' use_absolute_current is False.'
          )
      if self.gaussian_width is None:
        raise ValueError(
            "gaussian_width must be provided when mode is 'model_based'."
        )
      if self.gaussian_location is None:
        raise ValueError(
            "gaussian_location must be provided when mode is 'model_based'."
        )
    return self

  def build_runtime_params(
      self,
      t: chex.Numeric,
  ) -> RuntimeParams:
    base_params = super().build_runtime_params(t)
    return RuntimeParams(
        **vars(base_params),
        I_generic=0.0
        if self.I_generic is None
        else self.I_generic.get_value(t),
        fraction_of_total_current=(
            0.0
            if self.fraction_of_total_current is None
            else self.fraction_of_total_current.get_value(t)
        ),
        gaussian_width=(
            1.0
            if self.gaussian_width is None
            else self.gaussian_width.get_value(t)
        ),
        gaussian_location=(
            0.0
            if self.gaussian_location is None
            else self.gaussian_location.get_value(t)
        ),
        use_absolute_current=self.use_absolute_current,
    )

  def build_source(self) -> GenericCurrentSource:
    return GenericCurrentSource()

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

"""Module for a single source/sink term.

This module contains all the base classes for defining source terms. Other files
in this folder use these classes to define specific types of sources/sinks.

See Source class docstring for more details on what a TORAX source is and how to
use it.
"""
import abc
import dataclasses
import enum
import typing
from typing import ClassVar, Protocol

from jax import numpy as jnp
from torax._src import array_typing
from torax._src import state
from torax._src import static_dataclass
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.sources import runtime_params as runtime_params_lib
from torax._src.sources import source_profiles


@typing.runtime_checkable
class SourceProfileFunction(Protocol):
  """Sources implement these functions to be able to provide source profiles."""

  def __call__(
      self,
      runtime_params: runtime_params_slice.RuntimeParams,
      geo: geometry.Geometry,
      source_name: str,
      core_profiles: state.CoreProfiles,
      calculated_source_profiles: source_profiles.SourceProfiles | None,
      unused_conductivity: conductivity_base.Conductivity | None,
  ) -> tuple[array_typing.FloatVectorCell, ...]:
    ...


@enum.unique
class AffectedCoreProfile(enum.IntEnum):
  """Defines which part of the core profiles the source helps evolve.

  The profiles of each source/sink are terms included in equations evolving
  different core profiles. This enum maps a source to those equations.
  """

  # Current density equation.
  PSI = 1
  # Electron density equation.
  NE = 2
  # Ion temperature equation.
  TEMP_ION = 3
  # Electron temperature equation.
  TEMP_EL = 4


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class Source(static_dataclass.StaticDataclass, abc.ABC):
  """Base class for a single source/sink term.

  Sources are used to compute source profiles (see source_profiles.py), which
  are in turn used to compute coeffs in sim.py.

  Attributes:
    SOURCE_NAME: The name of the source.
    runtime_params: Input dataclass containing all the source-specific runtime
      parameters. At runtime, the parameters here are interpolated to a specific
      time t and then passed to the model_func, depending on the mode this
      source is running in.
    affected_core_profiles: Core profiles affected by this source's profile(s).
      This attribute defines which equations the source profiles are terms for.
      By default, the number of affected core profiles should equal the rank of
      the output shape returned by `output_shape`.
    model_func: The function used when the runtime type is set to "MODEL_BASED".
      If not provided, then it defaults to returning zeros.
    affected_core_profiles_ints: Derived property from the
      affected_core_profiles. Integer values of those enums.
  """

  SOURCE_NAME: ClassVar[str] = 'source'
  model_func: SourceProfileFunction | None = dataclasses.field(
      default=None, metadata={'hash_by_id': True}
  )

  @property
  @abc.abstractmethod
  def source_name(self) -> str:
    """Returns the name of the source."""

  @property
  @abc.abstractmethod
  def affected_core_profiles(self) -> tuple[AffectedCoreProfile, ...]:
    """Returns the core profiles affected by this source."""

  def get_value(
      self,
      runtime_params: runtime_params_slice.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      calculated_source_profiles: source_profiles.SourceProfiles | None,
      conductivity: conductivity_base.Conductivity | None,
  ) -> tuple[array_typing.FloatVectorCell, ...]:
    """Returns the cell grid profile for this source during one time step.

    Args:
      runtime_params: Slice of the general TORAX config that can be used as
        input for this time step.
      geo: Geometry of the torus.
      core_profiles: Core plasma profiles. May be the profiles at the start of
        the time step or a "live" set of core profiles being actively updated
        depending on whether this source is explicit or implicit. Explicit
        sources get the core profiles at the start of the time step, implicit
        sources get the "live" profiles that is updated through the course of
        the time step as the solver converges.
      calculated_source_profiles: The source profiles which have already been
        calculated for this time step if they exist. This is used to avoid
        recalculating profiles that are used as inputs to other sources. These
        profiles will only exist for Source instances that are implicit. i.e.
        explicit sources cannot depend on other calculated source profiles. In
        addition, different source types will have different availability of
        specific calculated_source_profiles since the calculation order matters.
        See source_profile_builders.py for more details.
      conductivity: Conductivity profile if it exists. It is only provided for
        implicit sources.

    Returns:
      A tuple of arrays of shape (cell grid length,) with one array per affected
      core profile.
    """
    source_params = runtime_params.sources[self.source_name]

    mode = source_params.mode
    match mode:
      case runtime_params_lib.Mode.MODEL_BASED:
        if self.model_func is None:
          raise ValueError(
              'Source is in MODEL_BASED mode but has no model function.'
          )
        return self.model_func(
            runtime_params,
            geo,
            self.source_name,
            core_profiles,
            calculated_source_profiles,
            conductivity,
        )
      case runtime_params_lib.Mode.PRESCRIBED:
        if len(self.affected_core_profiles) != len(
            source_params.prescribed_values
        ):
          raise ValueError(
              'When using PRESCRIBED mode, the number of prescribed values must'
              ' match the number of affected core profiles. Was: '
              f'{len(source_params.prescribed_values)} '
              f' Expected: {len(self.affected_core_profiles)}.'
          )
        return source_params.prescribed_values
      case runtime_params_lib.Mode.ZERO:
        zeros = jnp.zeros(geo.rho_norm.shape)
        return (zeros,) * len(self.affected_core_profiles)
      case _:
        raise ValueError(f'Unknown mode: {mode}')

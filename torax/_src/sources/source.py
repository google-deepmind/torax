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
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.physics import fast_ions as fast_ions_lib
from torax._src.sources import runtime_params as sources_runtime_params_lib
from torax._src.sources import source_profiles

SourceProfileElement = (
    array_typing.FloatVectorCell | tuple[fast_ions_lib.FastIon, ...]
)


@typing.runtime_checkable
class SourceProfileFunction(Protocol):
  """Sources implement these functions to be able to provide source profiles."""

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      source_name: str,
      core_profiles: state.CoreProfiles,
      calculated_source_profiles: source_profiles.SourceProfiles | None,
      unused_conductivity: conductivity_base.Conductivity | None,
  ) -> tuple[SourceProfileElement, ...]:
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
  # Fast ions.
  FAST_IONS = 5


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

  def zero_fast_ions(
      self,
      geo: geometry.Geometry,
  ) -> tuple[fast_ions_lib.FastIon, ...]:
    """Returns a tuple of zero fast ion profiles."""
    del geo  # Unused in the default case.
    if AffectedCoreProfile.FAST_IONS in self.affected_core_profiles:
      raise NotImplementedError(
          f'{type(self).__name__} affects FAST_IONS but does not override'
          ' zero_fast_ions.'
      )
    return ()

  def _validate_fast_ions(
      self,
      fast_ions: SourceProfileElement,
      geo: geometry.Geometry,
  ):
    """Validates the fast ion profiles."""
    if not isinstance(fast_ions, tuple):
      # PRESCRIBED mode might incorrectly supply a single array instead
      # of a tuple of FastIons if not configured correctly.
      raise TypeError(
          'FAST_IONS profile must be a tuple of FastIon, but got'
          f' {type(fast_ions)}.'
      )
    fast_ions = typing.cast(tuple[fast_ions_lib.FastIon, ...], fast_ions)
    zero_fast_ions = self.zero_fast_ions(geo)
    if len(fast_ions) != len(zero_fast_ions):
      raise ValueError(
          'Fast ion profiles must have the same length as zero_fast_ions. Was:'
          f' {len(fast_ions)}. Expected:'
          f' {len(zero_fast_ions)}.'
      )
    expected_species_order = [fast_ion.species for fast_ion in zero_fast_ions]
    actual_species_order = [fast_ion.species for fast_ion in fast_ions]
    if any(
        actual_species != expected_species
        for actual_species, expected_species in zip(
            actual_species_order, expected_species_order
        )
    ):
      raise ValueError(
          'Fast ion profiles must have the same species in the same order as'
          f' zero_fast_ions. Was: {actual_species_order}. Expected:'
          f' {expected_species_order}.'
      )

  def get_value(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      calculated_source_profiles: source_profiles.SourceProfiles | None,
      conductivity: conductivity_base.Conductivity | None,
  ) -> tuple[SourceProfileElement, ...]:
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
      A tuple with one element per affected core profile. Each element is either
      a FloatVectorCell array or, for FAST_IONS, a tuple of FastIon.
    """
    source_params = runtime_params.sources[self.source_name]

    mode = source_params.mode
    match mode:
      case sources_runtime_params_lib.Mode.MODEL_BASED:
        if self.model_func is None:
          raise ValueError(
              'Source is in MODEL_BASED mode but has no model function.'
          )
        res = self.model_func(
            runtime_params,
            geo,
            self.source_name,
            core_profiles,
            calculated_source_profiles,
            conductivity,
        )
      case sources_runtime_params_lib.Mode.PRESCRIBED:
        if len(self.affected_core_profiles) != len(
            source_params.prescribed_values
        ):
          raise ValueError(
              'When using PRESCRIBED mode, the number of prescribed values must'
              ' match the number of affected core profiles. Was: '
              f'{len(source_params.prescribed_values)} '
              f' Expected: {len(self.affected_core_profiles)}.'
          )
        res = source_params.prescribed_values
      case sources_runtime_params_lib.Mode.ZERO:
        zeros = jnp.zeros(geo.rho_norm.shape)
        res = tuple(
            self.zero_fast_ions(geo)
            if acp == AffectedCoreProfile.FAST_IONS
            else zeros
            for acp in self.affected_core_profiles
        )
      case _:
        raise ValueError(f'Unknown mode: {mode}')

    if AffectedCoreProfile.FAST_IONS in self.affected_core_profiles:
      fast_ions_idx = self.affected_core_profiles.index(
          AffectedCoreProfile.FAST_IONS
      )
      self._validate_fast_ions(
          res[fast_ions_idx],
          geo,
      )

    return res

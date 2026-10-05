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
from torax._src.physics import fast_ion as fast_ion_lib
from torax._src.sources import runtime_params as sources_runtime_params_lib
from torax._src.sources import source_profiles

SourceProfileElement = (
    array_typing.FloatVectorCell | tuple[fast_ion_lib.FastIon, ...]
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
    SOURCE_ID: Identifier for the source type (e.g. `'ecrh'`, `'icrh'`,
      `'fusion'`). This corresponds to the field name on the `Sources` Pydantic
      config and the key in `runtime_params.sources` and
      `SourceModels.standard_sources`.
    AFFECTED_CORE_PROFILES: Core profiles affected by this source's profile(s).
      This attribute defines which equations the source profiles are terms for.
      By default, the number of affected core profiles should equal the rank of
      the output shape returned by `output_shape`.
    runtime_params: Input dataclass containing all the source-specific runtime
      parameters. At runtime, the parameters here are interpolated to a specific
      time t and then passed to the model_func, depending on the mode this
      source is running in.
    model_func: The function used when the runtime type is set to "MODEL_BASED".
      If not provided, then it defaults to returning zeros.
  """

  SOURCE_ID: ClassVar[str]
  AFFECTED_CORE_PROFILES: ClassVar[tuple[AffectedCoreProfile, ...]]
  model_func: SourceProfileFunction | None = dataclasses.field(
      default=None, metadata={'hash_by_id': True}
  )

  def __post_init__(self):
    if not hasattr(self, 'SOURCE_ID'):
      raise ValueError('Source ID must be set.')
    if not getattr(self, 'AFFECTED_CORE_PROFILES', None):
      raise ValueError('Affected core profiles must be set.')

  def zero_fast_ions(
      self,
      geo: geometry.Geometry,
  ) -> tuple[fast_ion_lib.FastIon, ...]:
    """Returns a tuple of zero fast ion profiles."""
    del geo  # Unused in the default case.
    if AffectedCoreProfile.FAST_IONS in self.AFFECTED_CORE_PROFILES:
      raise NotImplementedError(
          f'{type(self).__name__} affects FAST_IONS but does not override'
          ' zero_fast_ions.'
      )
    return ()

  def _populate_and_validate_fast_ions(
      self,
      fast_ions: SourceProfileElement,
      geo: geometry.Geometry,
  ) -> tuple[fast_ion_lib.FastIon, ...]:
    """Validates fast ion profiles and fills uncomputed species with zeros."""
    if not isinstance(fast_ions, (tuple, list)):
      # PRESCRIBED mode might incorrectly supply a single array instead
      # of a tuple of FastIons if not configured correctly.
      raise TypeError(
          'FAST_IONS profile must be a tuple or list of FastIon, but got'
          f' {type(fast_ions)}.'
      )
    zero_fast_ions = self.zero_fast_ions(geo)
    expected_species = {fast_ion.species for fast_ion in zero_fast_ions}
    computed: dict[str, fast_ion_lib.FastIon] = {}
    for fast_ion in fast_ions:
      if not isinstance(fast_ion, fast_ion_lib.FastIon):
        raise TypeError(
            'Each element of FAST_IONS profile must be a FastIon, but got'
            f' {type(fast_ion)}.'
        )
      if fast_ion.species not in expected_species:
        raise ValueError(
            f'Unsupported FastIon species {fast_ion.species!r}. Expected one'
            f' of {[fi.species for fi in zero_fast_ions]}.'
        )
      if fast_ion.species in computed:
        raise ValueError(
            f'Duplicate FastIon species {fast_ion.species!r} in FAST_IONS'
            ' profile.'
        )
      computed[fast_ion.species] = fast_ion
    return tuple(
        computed.get(zero_fi.species, zero_fi) for zero_fi in zero_fast_ions
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
    source_params = runtime_params.sources[self.SOURCE_ID]

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
            self.SOURCE_ID,
            core_profiles,
            calculated_source_profiles,
            conductivity,
        )
      case sources_runtime_params_lib.Mode.PRESCRIBED:
        expected_len = len(self.AFFECTED_CORE_PROFILES)
        prescribed_len = len(source_params.prescribed_values)
        if (
            AffectedCoreProfile.FAST_IONS in self.AFFECTED_CORE_PROFILES
            and not runtime_params.numerics.enable_fast_ions
            and prescribed_len == expected_len - 1
        ):
          fast_ions_idx = self.AFFECTED_CORE_PROFILES.index(
              AffectedCoreProfile.FAST_IONS
          )
          res_list: list[SourceProfileElement] = list(
              source_params.prescribed_values
          )
          res_list.insert(fast_ions_idx, ())
          res = tuple(res_list)
        elif prescribed_len != expected_len:
          raise ValueError(
              'When using PRESCRIBED mode, the number of prescribed values must'
              ' match the number of affected core profiles. Was: '
              f'{len(source_params.prescribed_values)} '
              f' Expected: {len(self.AFFECTED_CORE_PROFILES)}.'
          )
        else:
          res = source_params.prescribed_values
      case sources_runtime_params_lib.Mode.ZERO:
        zeros = jnp.zeros(geo.rho_norm.shape)
        res_list = []
        for affected_core_profile in self.AFFECTED_CORE_PROFILES:
          if affected_core_profile == AffectedCoreProfile.FAST_IONS:
            if runtime_params.numerics.enable_fast_ions:
              res_list.append(self.zero_fast_ions(geo))
            else:
              res_list.append(())
          else:
            res_list.append(zeros)
        res = tuple(res_list)
      case _:
        raise ValueError(f'Unknown mode: {mode}')

    if AffectedCoreProfile.FAST_IONS in self.AFFECTED_CORE_PROFILES:
      fast_ions_idx = self.AFFECTED_CORE_PROFILES.index(
          AffectedCoreProfile.FAST_IONS
      )
      res_list = list(res)
      if runtime_params.numerics.enable_fast_ions:
        res_list[fast_ions_idx] = self._populate_and_validate_fast_ions(
            res[fast_ions_idx],
            geo,
        )
      else:
        res_list[fast_ions_idx] = ()
      res = tuple(res_list)

    return res

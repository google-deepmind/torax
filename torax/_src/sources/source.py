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

r"""Module for a single source/sink term.

This module contains the base class hierarchy for defining TORAX source and sink
terms. TORAX structures sources into three tiers based on their evaluation stage
and inputs:

1. :class:`Source`: Standard sources depending only on algebraic
   :class:`~torax.SolverProfiles` (kinetic temperatures and densities, ion
   composition, flux). Evaluated during basic source evaluation (pass 1
   explicit, pass 2 implicit before neoclassical/transport). May affect any
   channel (including current density / ``PSI``), can be configured as explicit
   or implicit, and can produce fast ion populations.

2. :class:`SourceUsingCoreProfiles`: Sources that require full
   :class:`~torax.CoreProfiles`, including quantities computed during the
   intra-step physics evaluation such as neoclassical conductivity and loop
   voltage (:math:`\\dot{\\psi}`). Evaluated after neoclassical and transport
   models. Cannot affect ``PSI``, cannot produce fast ions, and must be
   implicit.

3. :class:`FinalSource`: Source models evaluated after all other sources have
   been computed, with access to both full :class:`~torax.CoreProfiles` and the
   aggregated :class:`~torax.sources.SourceProfiles`. Used for models such as
   constant-fraction radiation sinks that depend on the total heating from all
   other sources. At most one ``FinalSource`` may be registered in a simulation.
"""

import abc
from collections.abc import Callable
import dataclasses
from typing import ClassVar, TypeAlias

from jax import numpy as jnp
from torax._src import static_dataclass
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import profiles
from torax._src.core_profiles import solver_profiles
from torax._src.geometry import geometry
from torax._src.sources import runtime_params as sources_runtime_params_lib
from torax._src.sources import source_profiles
from torax._src.sources import source_state as source_state_lib


AffectedCoreProfile = source_profiles.AffectedCoreProfile


_ModelFn: TypeAlias = Callable[
    [],
    tuple[source_profiles.SourceOutput, source_state_lib.SourceState],
]


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class SourceBase(static_dataclass.StaticDataclass, abc.ABC):
  """Shared abstract base class for all TORAX source and sink terms.

  Every source model in TORAX subclasses :class:`SourceBase` (typically via
  one of :class:`Source`, :class:`SourceUsingCoreProfiles`, or
  :class:`FinalSource`).

  Attributes:
    SOURCE_ID: String identifier for the source type (e.g. ``'ecrh'``,
      ``'icrh'``, ``'fusion'``). This corresponds to the field name on the
      :class:`~torax.sources.SourcesConfig` Pydantic configuration and the key
      in ``runtime_params.sources`` and ``SourceModels.standard_sources``.
    AFFECTED_CORE_PROFILES: Tuple of :class:`AffectedCoreProfile` enum members
      defining which PDE channel equations this source's output profiles
      contribute to. The order in this tuple must match the profile ordering in
      the returned :class:`~torax.sources.SourceOutput.profiles`.
  """

  SOURCE_ID: ClassVar[str]
  AFFECTED_CORE_PROFILES: ClassVar[tuple[AffectedCoreProfile, ...]]

  def __post_init__(self):
    super().__post_init__()
    if not hasattr(self, 'SOURCE_ID'):
      raise ValueError('Source ID must be set.')
    if not getattr(self, 'AFFECTED_CORE_PROFILES', None):
      raise ValueError('Affected core profiles must be set.')

  def initial_state(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
  ) -> source_state_lib.SourceState:
    """Returns the initial dynamic model state for this source.

    Sources that maintain time-dependent internal state across simulation steps
    should override this method to provide an initialized subclass of
    :class:`~torax.sources.SourceState`.

    Args:
      runtime_params: Full simulation runtime parameters for parameter-dependent
        initialization.

    Returns:
      An initialized :class:`~torax.sources.SourceState` (defaults to an empty
      base state).
    """
    del runtime_params
    return source_state_lib.SourceState()

  def _evaluate_with_mode(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      source_state: source_state_lib.SourceState,
      model_fn: _ModelFn,
  ) -> tuple[source_profiles.SourceOutput, source_state_lib.SourceState]:
    """Evaluates the source according to its configured runtime mode.

    Handles dispatching between:
    - ``MODEL_BASED``: executes ``model_fn()`` to compute physics values.
    - ``PRESCRIBED``: uses prescribed profile arrays from runtime parameters.
    - ``ZERO``: returns zero profiles on the cell grid.

    Args:
      runtime_params: Full simulation runtime parameters.
      geo: Toroidal geometry of the tokamak.
      source_state: Current dynamic model state.
      model_fn: Zero-argument callable executing the model evaluation when in
        ``MODEL_BASED`` mode.

    Returns:
      A tuple of ``(output, updated_source_state)``.

    Raises:
      ValueError: If in ``PRESCRIBED`` mode and the length of prescribed values
        does not match the number of affected core profiles, or if an unknown
        mode is encountered.
    """
    source_params = runtime_params.sources[self.SOURCE_ID]

    mode = source_params.mode
    match mode:
      case sources_runtime_params_lib.Mode.MODEL_BASED:
        output, source_state = model_fn()
      case sources_runtime_params_lib.Mode.PRESCRIBED:
        if len(source_params.prescribed_values) != len(
            self.AFFECTED_CORE_PROFILES
        ):
          raise ValueError(
              'When using PRESCRIBED mode, the number of prescribed values must'
              ' match the number of affected core profiles. Was: '
              f'{len(source_params.prescribed_values)} '
              f' Expected: {len(self.AFFECTED_CORE_PROFILES)}.'
          )
        output = source_profiles.SourceOutput(
            profiles=source_params.prescribed_values
        )
      case sources_runtime_params_lib.Mode.ZERO:
        zeros = jnp.zeros(geo.rho_norm.shape)
        output = source_profiles.SourceOutput(
            profiles=(zeros,) * len(self.AFFECTED_CORE_PROFILES)
        )
      case _:
        raise ValueError(f'Unknown mode: {mode}')

    return output, source_state


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class Source(SourceBase):
  r"""Standard source model depending on SolverProfiles.

  Subclass :class:`Source` when your physics model depends only on algebraic
  plasma state available on :class:`~torax.SolverProfiles` (temperatures,
  densities, ion composition, magnetic flux, safety factor) and does not require
  neoclassical conductivity, loop voltage (:math:`\\dot{\\psi}`), or aggregate
  heating profiles from other sources.

  Capabilities and constraints:
  - May affect any core profile channel, including
  :attr:`AffectedCoreProfile.PSI`.
  - May be configured as explicit (evaluated once before the solver step) or
    implicit (evaluated iteratively during each solver sub-step).
  - May generate fast ion populations by returning :class:`~torax.FastIon`
    objects in :attr:`SourceOutput.fast_ions`.
  """

  @abc.abstractmethod
  def _get_model_value(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: solver_profiles.SolverProfiles,
      source_state: source_state_lib.SourceState,
  ) -> tuple[source_profiles.SourceOutput, source_state_lib.SourceState]:
    """Computes model-based source profile terms and updated dynamic state.

    Subclasses must implement this method to perform the physics calculation.

    Args:
      runtime_params: Full simulation runtime parameters.
      geo: Magnetic geometry of the tokamak.
      core_profiles: Algebraic core profiles derived directly from SolverState.
      source_state: Dynamic model state carried from the previous timestep.

    Returns:
      A tuple of ``(output, updated_source_state)``, where ``output`` is a
      :class:`~torax.sources.SourceOutput` containing profile arrays on the
      cell grid (one per entry in :attr:`AFFECTED_CORE_PROFILES`) and any
      generated fast ion populations.
    """

  def get_value(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: solver_profiles.SolverProfiles,
      source_state: source_state_lib.SourceState,
  ) -> tuple[source_profiles.SourceOutput, source_state_lib.SourceState]:
    """Evaluates the source, applying mode overrides.

    Supported modes are MODEL_BASED, PRESCRIBED, and ZERO.

    Args:
      runtime_params: Full simulation runtime parameters.
      geo: Magnetic geometry of the tokamak.
      core_profiles: Algebraic core profiles derived directly from SolverState.
      source_state: Current dynamic model state.

    Returns:
      A tuple of ``(output, updated_source_state)`` for this evaluation.
    """
    return self._evaluate_with_mode(
        runtime_params,
        geo,
        source_state,
        lambda: self._get_model_value(
            runtime_params, geo, core_profiles, source_state
        ),
    )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class SourceUsingCoreProfiles(SourceBase):
  r"""Source requiring full CoreProfiles (such as conductivity and psidot).

  Subclass :class:`SourceUsingCoreProfiles` when your model requires profiles
  computed during the intra-step physics evaluation, such as neoclassical
  parallel conductivity (:attr:`CoreProfiles.conductivity`) or loop voltage /
  time derivative of poloidal flux (:attr:`CoreProfiles.psidot`).

  Capabilities and constraints:
  - Cannot affect :attr:`AffectedCoreProfile.PSI` because :math:`\\dot{\\psi}`
    depends on the total current from all current-driving sources.
  - Cannot produce fast ions (:attr:`SourceOutput.fast_ions` must be empty)
    because fast ions must be known before neoclassical and transport
    calculations.
  - Cannot be configured with ``is_explicit=True`` because full
    :class:`~torax.CoreProfiles` are only available during implicit intra-step
    evaluation.
  """

  def __post_init__(self):
    super().__post_init__()
    if AffectedCoreProfile.PSI in self.AFFECTED_CORE_PROFILES:
      raise ValueError(
          f'{type(self).__name__} cannot affect PSI, but got'
          f' AFFECTED_CORE_PROFILES={self.AFFECTED_CORE_PROFILES}.'
      )

  @abc.abstractmethod
  def _get_model_value(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: profiles.CoreProfiles,
      source_state: source_state_lib.SourceState,
  ) -> tuple[source_profiles.SourceOutput, source_state_lib.SourceState]:
    """Computes model-based source profile terms from full CoreProfiles.

    Subclasses must implement this method to perform the physics calculation.

    Args:
      runtime_params: Full simulation runtime parameters.
      geo: Magnetic geometry of the tokamak.
      core_profiles: Full core plasma profiles including neoclassical
        conductivity and psidot.
      source_state: Dynamic model state carried from the previous timestep.

    Returns:
      A tuple of ``(output, updated_source_state)``, where ``output`` is a
      :class:`~torax.sources.SourceOutput` containing profile arrays on the
      cell grid (one per entry in :attr:`AFFECTED_CORE_PROFILES`). Must not
      contain fast ions.
    """

  def get_value(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: profiles.CoreProfiles,
      source_state: source_state_lib.SourceState,
  ) -> tuple[source_profiles.SourceOutput, source_state_lib.SourceState]:
    """Evaluates the source with runtime validation and mode dispatch.

    Validates that the source is not configured with ``is_explicit=True`` and
    that the model output does not produce fast ions.

    Args:
      runtime_params: Full simulation runtime parameters.
      geo: Magnetic geometry of the tokamak.
      core_profiles: Full core plasma profiles including neoclassical
        conductivity and psidot.
      source_state: Current dynamic model state.

    Returns:
      A tuple of ``(output, updated_source_state)`` for this evaluation.

    Raises:
      ValueError: If configured with ``is_explicit=True`` or if the model
        produces fast ions.
    """
    source_params = runtime_params.sources[self.SOURCE_ID]
    if source_params.is_explicit:
      raise ValueError(
          f'{type(self).__name__} requires implicit CoreProfiles (such as'
          ' conductivity and psidot) and cannot be configured with'
          ' is_explicit=True.'
      )
    output, source_state = self._evaluate_with_mode(
        runtime_params,
        geo,
        source_state,
        lambda: self._get_model_value(
            runtime_params, geo, core_profiles, source_state
        ),
    )
    if output.fast_ions:
      raise ValueError(
          f'{type(self).__name__} consumes CoreProfiles and cannot produce'
          ' fast_ions.'
      )
    return output, source_state


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class FinalSource(SourceBase):
  """Final source model evaluated after all other sources are computed.

  Subclass :class:`FinalSource` when your model aggregates or scales against
  already-computed source profiles (for example, constant-fraction impurity
  radiation that radiates a prescribed fraction of the total input power).

  Capabilities and constraints:
  - At most one :class:`FinalSource` may be registered in
    :class:`~torax.sources.SourceModels` so evaluation order remains
    deterministic and unambiguous.
  - Cannot affect :attr:`AffectedCoreProfile.PSI`.
  - Cannot produce fast ions (:attr:`SourceOutput.fast_ions` must be empty).
  - Cannot be configured with ``is_explicit=True`` because aggregate source
    profiles are only assembled after all standard and core-profile sources
    have been evaluated.
  """

  def __post_init__(self):
    super().__post_init__()
    if AffectedCoreProfile.PSI in self.AFFECTED_CORE_PROFILES:
      raise ValueError(
          f'{type(self).__name__} cannot affect PSI, but got'
          f' AFFECTED_CORE_PROFILES={self.AFFECTED_CORE_PROFILES}.'
      )

  @abc.abstractmethod
  def _get_model_value(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: profiles.CoreProfiles,
      calculated_source_profiles: source_profiles.SourceProfiles,
      source_state: source_state_lib.SourceState,
  ) -> tuple[source_profiles.SourceOutput, source_state_lib.SourceState]:
    """Computes model-based source profile terms using source profiles.

    Subclasses must implement this method to perform the physics calculation.

    Args:
      runtime_params: Full simulation runtime parameters.
      geo: Magnetic geometry of the tokamak.
      core_profiles: Full core plasma profiles including neoclassical
        conductivity and psidot.
      calculated_source_profiles: Aggregated source profiles from all previously
        evaluated sources in the current step.
      source_state: Dynamic model state carried from the previous timestep.

    Returns:
      A tuple of ``(output, updated_source_state)``, where ``output`` is a
      :class:`~torax.sources.SourceOutput` containing profile arrays on the
      cell grid (one per entry in :attr:`AFFECTED_CORE_PROFILES`). Must not
      contain fast ions.
    """

  def get_value(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: profiles.CoreProfiles,
      calculated_source_profiles: source_profiles.SourceProfiles,
      source_state: source_state_lib.SourceState,
  ) -> tuple[source_profiles.SourceOutput, source_state_lib.SourceState]:
    """Evaluates the final source with runtime validation and mode dispatch.

    Validates that the source is not configured with ``is_explicit=True`` and
    that the model output does not produce fast ions.

    Args:
      runtime_params: Full simulation runtime parameters.
      geo: Magnetic geometry of the tokamak.
      core_profiles: Full core plasma profiles including neoclassical
        conductivity and psidot.
      calculated_source_profiles: Aggregated source profiles from all previously
        evaluated sources in the current step.
      source_state: Current dynamic model state.

    Returns:
      A tuple of ``(output, updated_source_state)`` for this evaluation.

    Raises:
      ValueError: If configured with ``is_explicit=True`` or if the model
        produces fast ions.
    """
    source_params = runtime_params.sources[self.SOURCE_ID]
    if source_params.is_explicit:
      raise ValueError(
          f'{type(self).__name__} aggregates other source profiles and cannot'
          ' be configured with is_explicit=True.'
      )
    output, source_state = self._evaluate_with_mode(
        runtime_params,
        geo,
        source_state,
        lambda: self._get_model_value(
            runtime_params,
            geo,
            core_profiles,
            calculated_source_profiles,
            source_state,
        ),
    )
    if output.fast_ions:
      raise ValueError(
          f'{type(self).__name__} consumes CoreProfiles and cannot produce'
          ' fast_ions.'
      )
    return output, source_state

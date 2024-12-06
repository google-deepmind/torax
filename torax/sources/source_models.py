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

"""Functions for building source profiles in TORAX."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from torax import constants
from torax import geometry
from torax import jax_utils
from torax import state
from torax.config import runtime_params_slice
from torax.sources import bootstrap_current_source
from torax.sources import generic_current_source
from torax.sources import qei_source as qei_source_lib
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source as source_lib
from torax.sources import source_profiles


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'source_models',
        'static_runtime_params_slice',
    ],
)
def build_source_profiles(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: SourceModels,
    explicit: bool,
) -> source_profiles.SourceProfiles:
  """Builds explicit or implicit source profiles.

  Args:
    static_runtime_params_slice: Input config. Cannot change from time step to
      time step.
    dynamic_runtime_params_slice: Input config for this time step. Can change
      from time step to time step.
    geo: Geometry of the torus.
    core_profiles: Core plasma profiles, either at the start of the time step
      (if explicit) or the live profiles being evolved during the time step (if
      implicit).
    source_models: Functions computing profiles for all TORAX sources/sinks.
    explicit: If True, this function should return profiles for all explicit
      sources. All implicit sources should be set to 0. And same vice versa.

  Returns:
    SourceProfiles for either explicit or implicit sources (and all others set
    to zero).
  """
  # Bootstrap current is a special-case source with multiple outputs, so handle
  # it here.
  dynamic_bootstrap_runtime_params = dynamic_runtime_params_slice.sources[
      source_models.j_bootstrap_name
  ]
  static_bootstrap_runtime_params = static_runtime_params_slice.sources[
      source_models.j_bootstrap_name
  ]
  bootstrap_profiles = _build_bootstrap_profiles(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      dynamic_source_runtime_params=dynamic_bootstrap_runtime_params,
      static_runtime_params_slice=static_runtime_params_slice,
      static_source_runtime_params=static_bootstrap_runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      j_bootstrap_source=source_models.j_bootstrap,
      explicit=explicit,
  )
  other_profiles = _build_standard_source_profiles(
      static_runtime_params_slice,
      dynamic_runtime_params_slice,
      geo,
      core_profiles,
      source_models,
      explicit,
  )
  return source_profiles.SourceProfiles(
      profiles=other_profiles,
      j_bootstrap=bootstrap_profiles,
      # Qei is computed within calc_coeffs and will replace this value. This is
      # here as a placeholder with correct shapes.
      qei=source_profiles.QeiInfo.zeros(geo),
  )


def _build_bootstrap_profiles(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    static_source_runtime_params: runtime_params_lib.StaticRuntimeParams,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    dynamic_source_runtime_params: runtime_params_lib.DynamicRuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    j_bootstrap_source: bootstrap_current_source.BootstrapCurrentSource,
    explicit: bool = True,
    calculate_anyway: bool = False,
) -> source_profiles.BootstrapCurrentProfile:
  """Computes the bootstrap current profile.

  Args:
    static_runtime_params_slice: Input config. Cannot change from time step to
      time step.
    static_source_runtime_params: Input runtime parameters specific to the
      bootstrap current source that do not change from time step to time step.
    dynamic_runtime_params_slice: Input config for this time step. Can change
      from time step to time step.
    dynamic_source_runtime_params: Input runtime parameters for this time step,
      specific to the bootstrap current source.
    geo: Geometry of the torus.
    core_profiles: Core plasma profiles, either at the start of the time step
      (if explicit) or the live profiles being evolved during the time step (if
      implicit).
    j_bootstrap_source: Bootstrap current source used to compute the profile.
    explicit: If True, this function should return the profile for an explicit
      source. If explicit is True and the bootstrap current source is not
      explicit, then this should return all zeros. And same with implicit (if
      explicit=False and the source is set to be explicit, then this will return
      all zeros).
    calculate_anyway: If True, returns values regardless of explicit

  Returns:
    Bootstrap current profile.
  """
  bootstrap_profile = j_bootstrap_source.get_value(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      dynamic_source_runtime_params=dynamic_source_runtime_params,
      static_runtime_params_slice=static_runtime_params_slice,
      static_source_runtime_params=static_source_runtime_params,
      geo=geo,
      core_profiles=core_profiles,
  )
  sigma = jax_utils.select(
      jnp.logical_or(
          explicit == static_source_runtime_params.is_explicit,
          calculate_anyway,
      ),
      bootstrap_profile.sigma,
      jnp.zeros_like(bootstrap_profile.sigma),
  )
  sigma_face = jax_utils.select(
      jnp.logical_or(
          explicit == static_source_runtime_params.is_explicit,
          calculate_anyway,
      ),
      bootstrap_profile.sigma_face,
      jnp.zeros_like(bootstrap_profile.sigma_face),
  )
  j_bootstrap = jax_utils.select(
      jnp.logical_or(
          explicit == static_source_runtime_params.is_explicit,
          calculate_anyway,
      ),
      bootstrap_profile.j_bootstrap,
      jnp.zeros_like(bootstrap_profile.j_bootstrap),
  )
  j_bootstrap_face = jax_utils.select(
      jnp.logical_or(
          explicit == static_source_runtime_params.is_explicit,
          calculate_anyway,
      ),
      bootstrap_profile.j_bootstrap_face,
      jnp.zeros_like(bootstrap_profile.j_bootstrap_face),
  )
  I_bootstrap = jax_utils.select(  # pylint: disable=invalid-name
      jnp.logical_or(
          explicit == static_source_runtime_params.is_explicit,
          calculate_anyway,
      ),
      bootstrap_profile.I_bootstrap,
      jnp.zeros_like(bootstrap_profile.I_bootstrap),
  )
  return source_profiles.BootstrapCurrentProfile(
      sigma=sigma,
      sigma_face=sigma_face,
      j_bootstrap=j_bootstrap,
      j_bootstrap_face=j_bootstrap_face,
      I_bootstrap=I_bootstrap,
  )


def _build_standard_source_profiles(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: SourceModels,
    explicit: bool = True,
    calculate_anyway: bool = False,
    affected_core_profiles: tuple[source_lib.AffectedCoreProfile, ...] = (
        source_lib.AffectedCoreProfile.PSI,
        source_lib.AffectedCoreProfile.NE,
        source_lib.AffectedCoreProfile.TEMP_ION,
        source_lib.AffectedCoreProfile.TEMP_EL,
    ),
) -> dict[str, jax.Array]:
  """Computes sources and builds a kwargs dict for SourceProfiles.

  Args:
    static_runtime_params_slice: Input config. Cannot change from time step to
      time step.
    dynamic_runtime_params_slice: Input config for this time step. Can change
      from time step to time step.
    geo: Geometry of the torus.
    core_profiles: Core plasma profiles, either at the start of the time step
      (if explicit) or the live profiles being evolved during the time step (if
      implicit).
    source_models: Collection of all TORAX sources.
    explicit: If True, this function should return the profile for an explicit
      source. If explicit is True and a given source is not explicit, then this
      function will return zeros for that source. And same with implicit (if
      explicit=False and the source is set to be explicit, then this will return
      all zeros).
    calculate_anyway: If True, returns values regardless of explicit
    affected_core_profiles: Populate the output for sources that affect these
      core profiles.

  Returns:
    dict of source profiles excluding the two special-case sources (bootstrap
    and qei).
  """
  computed_source_profiles = {}
  affected_core_profiles_set = set(affected_core_profiles)
  for source_name, source in source_models.standard_sources.items():
    if affected_core_profiles_set.intersection(source.affected_core_profiles):
      dynamic_source_runtime_params = dynamic_runtime_params_slice.sources[
          source_name
      ]
      static_source_runtime_params = static_runtime_params_slice.sources[
          source_name
      ]
      computed_source_profiles[source_name] = jax_utils.select(
          jnp.logical_or(
              explicit == static_source_runtime_params.is_explicit,
              calculate_anyway,
          ),
          source.get_value(
              static_runtime_params_slice,
              static_source_runtime_params,
              dynamic_runtime_params_slice,
              dynamic_source_runtime_params,
              geo,
              core_profiles,
          ),
          jnp.zeros(source.output_shape_getter(geo)),
      )
  return computed_source_profiles


def sum_sources_psi(
    geo: geometry.Geometry,
    source_profile: source_profiles.SourceProfiles,
    source_models: SourceModels,
) -> jax.Array:
  """Computes psi source values for sim.calc_coeffs."""
  total = source_profile.j_bootstrap.j_bootstrap
  for source_name, source in source_models.psi_sources.items():
    total += source.get_source_profile_for_affected_core_profile(
        profile=source_profile.profiles[source_name],
        affected_core_profile=source_lib.AffectedCoreProfile.PSI.value,
        geo=geo,
    )
  mu0 = constants.CONSTANTS.mu0
  prefactor = 8 * geo.vpr * jnp.pi**2 * geo.B0 * mu0 * geo.Phib / geo.F**2
  scale_source = lambda src: -src * prefactor
  return scale_source(total)


def sum_sources_ne(
    geo: geometry.Geometry,
    source_profile: source_profiles.SourceProfiles,
    source_models: SourceModels,
) -> jax.Array:
  """Computes ne source values for sim.calc_coeffs."""
  total = jnp.zeros_like(geo.rho)
  for source_name, source in source_models.ne_sources.items():
    total += source.get_source_profile_for_affected_core_profile(
        profile=source_profile.profiles[source_name],
        affected_core_profile=source_lib.AffectedCoreProfile.NE.value,
        geo=geo,
    )
  return total * geo.vpr


def sum_sources_temp_ion(
    geo: geometry.Geometry,
    source_profile: source_profiles.SourceProfiles,
    source_models: SourceModels,
) -> jax.Array:
  """Computes temp_ion source values for sim.calc_coeffs."""
  total = jnp.zeros_like(geo.rho)
  for source_name, source in source_models.temp_ion_sources.items():
    total += source.get_source_profile_for_affected_core_profile(
        profile=source_profile.profiles[source_name],
        affected_core_profile=(source_lib.AffectedCoreProfile.TEMP_ION.value),
        geo=geo,
    )
  return total * geo.vpr


def sum_sources_temp_el(
    geo: geometry.Geometry,
    source_profile: source_profiles.SourceProfiles,
    source_models: SourceModels,
) -> jax.Array:
  """Computes temp_el source values for sim.calc_coeffs."""
  total = jnp.zeros_like(geo.rho)
  for source_name, source in source_models.temp_el_sources.items():
    total += source.get_source_profile_for_affected_core_profile(
        profile=source_profile.profiles[source_name],
        affected_core_profile=(source_lib.AffectedCoreProfile.TEMP_EL.value),
        geo=geo,
    )
  return total * geo.vpr


def calc_and_sum_sources_psi(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: SourceModels,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Computes sum of psi sources for psi_dot calculation."""

  # TODO(b/335597108): Revisit how to calculate this once we enable more
  # expensive source functions that might not jittable (like file-based or
  # RPC-based sources).
  psi_profiles = _build_standard_source_profiles(
      static_runtime_params_slice,
      dynamic_runtime_params_slice,
      geo,
      core_profiles,
      source_models,
      calculate_anyway=True,
      affected_core_profiles=(source_lib.AffectedCoreProfile.PSI,),
  )
  total = 0
  for source_name, source in source_models.psi_sources.items():
    total += source.get_source_profile_for_affected_core_profile(
        profile=psi_profiles[source_name],
        affected_core_profile=source_lib.AffectedCoreProfile.PSI.value,
        geo=geo,
    )
  dynamic_bootstrap_runtime_params = dynamic_runtime_params_slice.sources[
      source_models.j_bootstrap_name
  ]
  static_bootstrap_runtime_params = static_runtime_params_slice.sources[
      source_models.j_bootstrap_name
  ]
  j_bootstrap_profiles = _build_bootstrap_profiles(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      dynamic_source_runtime_params=dynamic_bootstrap_runtime_params,
      static_runtime_params_slice=static_runtime_params_slice,
      static_source_runtime_params=static_bootstrap_runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      j_bootstrap_source=source_models.j_bootstrap,
      calculate_anyway=True,
  )
  total += j_bootstrap_profiles.j_bootstrap

  mu0 = constants.CONSTANTS.mu0
  prefactor = 8 * geo.vpr * jnp.pi**2 * geo.B0 * mu0 * geo.Phib / geo.F**2
  scale_source = lambda src: -src * prefactor

  return (
      scale_source(total),
      j_bootstrap_profiles.sigma,
      j_bootstrap_profiles.sigma_face,
  )


class SourceModels:
  """Source/sink models for the different equations being evolved in Torax.

  Each source/sink (all called sources as the difference is only a sign change)
  can be explicit or implicit and signal to our solver on how to handle these
  terms. Their values are provided via model, file, prescribed function, etc.
  The specific approach used depends on how the source is initialized and what
  runtime configuration inputs are provided.

  You can both override the default set of sources in TORAX as well as define
  new custom sources inline when constructing this object. The example below
  shows how to define a new custom electron-density source.

  .. code-block:: python

    # Define an electron-density source with a time-dependent Gaussian profile.
    my_custom_source = source.SingleProfileSource(
        supported_modes=(
            runtime_params_lib.Mode.ZERO,
            runtime_params_lib.Mode.FORMULA_BASED,
        ),
        affected_core_profiles=source.AffectedCoreProfile.NE,
        formula=formulas.Gaussian(),
        # Define (possibly) time-dependent parameters to feed to the formula.
        runtime_params=runtime_params_lib.RuntimeParams(
            formula=formula_config.Gaussian(
                total={0.0: 1.0, 5.0: 2.0, 10.0: 1.0},  # time-dependent.
                c1=2.0,
                c2=3.0,
            ),
        ),
    )
    # Define the collection of sources here, which in this example only includes
    # one source.
    all_torax_sources = SourceModels(
        sources={'my_custom_source': my_custom_source}
    )

  See runtime_params.py for more details on how to configure all the source/sink
  terms.
  """

  def __init__(
      self,
      source_builders: (
          dict[str, source_lib.SourceBuilderProtocol] | None
      ) = None,
  ):
    """Constructs a collection of sources.

    The constructor should only be called by SourceModelsBuilder.

    This class defines which sources are available in a TORAX simulation run.
    Users can configure whether each source is actually on and what kind of
    profile it produces by changing its runtime configuration (see
    runtime_params_lib.py).

    Args:
      source_builders: Mapping of source model names to builders of the Source
        objects. The names (i.e. the keys of this dictionary) also define the
        keys in the output SourceProfiles which are computed from this
        SourceModels object.

    NOTE - Some sources are "special-case": bootstrap_current, generic_current,
    and Qei. SourceModels will always instantiate default objects for these
    types of sources unless they are provided by this `sources` argument.

    Raises:
      ValueError if there is a naming collision with the reserved names as
      described above.
    """

    source_builders = source_builders or {}

    # Begin initial construction with sources that don't link back to the
    # SourceModels
    sources = {
        name: builder()
        for name, builder in source_builders.items()
        if not builder.links_back
    }

    # Some sources are accessed for specific use cases, so we extract those
    # ones and expose them directly.
    self._j_bootstrap = None
    self._generic_current = None
    self._qei_source = None
    # The rest of the sources are "standard".
    self._standard_sources = {}

    # Divide up the sources based on which core profiles they affect.
    self._psi_sources: dict[str, source_lib.Source] = {}
    self._ne_sources: dict[str, source_lib.Source] = {}
    self._temp_ion_sources: dict[str, source_lib.Source] = {}
    self._temp_el_sources: dict[str, source_lib.Source] = {}

    # First set the "special" sources.
    for source in sources.values():
      if isinstance(source, bootstrap_current_source.BootstrapCurrentSource):
        self._j_bootstrap = source
      elif isinstance(source, generic_current_source.GenericCurrentSource):
        self._generic_current = source
      elif isinstance(source, qei_source_lib.QeiSource):
        self._qei_source = source

    # Make sure defaults are set for the "special-case" sources.
    if self._j_bootstrap is None:
      self._j_bootstrap = bootstrap_current_source.BootstrapCurrentSource()
    if self._qei_source is None:
      self._qei_source = qei_source_lib.QeiSource()
    # If the generic current source wasn't provided, create a default one and
    # add to standard sources.
    if self._generic_current is None:
      self._generic_current = generic_current_source.GenericCurrentSource()
      self._add_standard_source(
          generic_current_source.GenericCurrentSource.SOURCE_NAME,
          self._generic_current,
      )

    # Then add all the "standard" sources.
    for source_name, source in sources.items():
      if isinstance(
          source, bootstrap_current_source.BootstrapCurrentSource
      ) or isinstance(source, qei_source_lib.QeiSource):
        continue
      else:
        self._add_standard_source(source_name, source)

    # Now add the sources that link back
    for name, builder in source_builders.items():
      if builder.links_back:
        self._add_standard_source(name, builder(self))

    # The instance is constructed, now freeze it
    self._frozen = True

  def __setattr__(self, attr, value):
    # pylint: disable=g-doc-args
    # pylint: disable=g-doc-return-or-yield
    """Override __setattr__ to make the class (sort of) immutable.

    Note that you can still do obj.field.subfield = x, so it is not true
    immutability, but this to helps to avoid some careless errors.
    """
    if getattr(self, '_frozen', False):
      raise AttributeError('SourceModels is immutable.')
    return super().__setattr__(attr, value)

  def _add_standard_source(
      self,
      source_name: str,
      source: source_lib.Source,
  ) -> None:
    """Adds a source to the collection of sources.

    Do NOT directly add new sources to `SourceModels.standard_sources`. Users
    should call this function instead. Cannot add additional bootstrap current,
    external current, or Qei sources - those must be defined in the __init__.

    Args:
      source_name: Name of the new source being added. This will be the key
        under which the source's output profile will be found in the output
        SourceProfiles object.
      source: The new standard source being added.

    Raises:
      ValueError if a "special-case" source is provided.
    """
    if (
        isinstance(source, bootstrap_current_source.BootstrapCurrentSource)
        or isinstance(source, qei_source_lib.QeiSource)
    ):
      raise ValueError(
          'Cannot add a source with the following types: '
          'bootstrap_current_source.BootstrapCurrentSource,'
          ' external_current_source.ExternalCurrentSource, or'
          ' qei_source_lib.QeiSource. These must be added at init time.'
      )
    if source_name in self.sources.keys():
      raise ValueError(
          f'Trying to add another source with the same name: {source_name}.'
      )
    self._standard_sources[source_name] = source
    if source_lib.AffectedCoreProfile.PSI in source.affected_core_profiles:
      self._psi_sources[source_name] = source
    if source_lib.AffectedCoreProfile.NE in source.affected_core_profiles:
      self._ne_sources[source_name] = source
    if source_lib.AffectedCoreProfile.TEMP_ION in source.affected_core_profiles:
      self._temp_ion_sources[source_name] = source
    if source_lib.AffectedCoreProfile.TEMP_EL in source.affected_core_profiles:
      self._temp_el_sources[source_name] = source

  # Some sources require direct access, so this class defines properties for
  # those sources.

  @property
  def j_bootstrap(self) -> bootstrap_current_source.BootstrapCurrentSource:
    if self._j_bootstrap is None:
      raise ValueError('j_bootstrap is not initialized.')
    return self._j_bootstrap

  @property
  def j_bootstrap_name(self) -> str:
    return bootstrap_current_source.BootstrapCurrentSource.SOURCE_NAME

  @property
  def generic_current_source(
      self,
  ) -> generic_current_source.GenericCurrentSource:
    # TODO(b/336995925): Modify to be a sum over all current sources.
    if self._generic_current is None:
      raise ValueError('generic_current is not initialized.')
    return self._generic_current

  @property
  def generic_current_source_name(self) -> str:
    return generic_current_source.GenericCurrentSource.SOURCE_NAME

  @property
  def qei_source(self) -> qei_source_lib.QeiSource:
    if self._qei_source is None:
      raise ValueError('qei_source is not initialized.')
    return self._qei_source

  @property
  def qei_source_name(self) -> str:
    return qei_source_lib.QeiSource.SOURCE_NAME

  @property
  def psi_sources(self) -> dict[str, source_lib.Source]:
    return self._psi_sources

  @property
  def ne_sources(self) -> dict[str, source_lib.Source]:
    return self._ne_sources

  @property
  def temp_ion_sources(self) -> dict[str, source_lib.Source]:
    return self._temp_ion_sources

  @property
  def temp_el_sources(self) -> dict[str, source_lib.Source]:
    return self._temp_el_sources

  @property
  def ion_el_sources(self) -> dict[str, source_lib.Source]:
    """Returns all source models which output both ion and el temp profiles."""
    return {
        name: source
        for name, source in self._standard_sources.items()
        if source.affected_core_profiles
        == (
            source_lib.AffectedCoreProfile.TEMP_ION,
            source_lib.AffectedCoreProfile.TEMP_EL,
        )
    }

  @property
  def standard_sources(self) -> dict[str, source_lib.Source]:
    """Returns all sources that are not used in special cases.

    Practically, this means this includes all sources other than j_bootstrap and
    qei_source.
    """
    return self._standard_sources

  @property
  def sources(self) -> dict[str, source_lib.Source]:
    return self._standard_sources | {
        self.j_bootstrap_name: self.j_bootstrap,
        self.qei_source_name: self.qei_source,
    }


class SourceModelsBuilder:
  """Builds a SourceModels and also holds its runtime_params.

  The SourceModels is a collection of many smaller Source models.

  Attributes:
    source_builders: Dict mapping the name of each Source to its builder.
  """

  def __init__(
      self,
      source_builders: (
          dict[str, source_lib.SourceBuilderProtocol] | None
      ) = None,
  ):

    # Note: this runtime type checking isn't needed just because of the
    # dynamically created builder classes, pytype seems to have a bug that
    # prevents it from checking this arg in general.
    # In an alternate version of SourceModelsBuilder it had a dict of dict of
    # configs and pytype failed to enforce that the values of the outer dict
    # were dicts.
    if source_builders:
      for name, builder in source_builders.items():
        if not source_lib.is_source_builder(builder):
          raise TypeError(
              f'Expected source builder, got {type(builder)}for "{name}"'
          )

    source_builders = source_builders or {}

    # Validate that these sources are found
    bootstrap_found = qei_found = generic_current_found = False
    if (
        bootstrap_current_source.BootstrapCurrentSource.SOURCE_NAME
        in source_builders
    ):
      bootstrap_found = True
    if qei_source_lib.QeiSource.SOURCE_NAME in source_builders:
      qei_found = True
    if (
        generic_current_source.GenericCurrentSource.SOURCE_NAME
        in source_builders
    ):
      generic_current_found = True
    # These are special sources that must be present for every TORAX run.
    # If these sources are missing, we need to include builders for them.
    # We also ZERO out these sources if they are not explicitly provided.
    # The SourceModels would also build them, but then there'd be no
    # user-editable runtime params for them.
    if not bootstrap_found:
      source_builders[
          bootstrap_current_source.BootstrapCurrentSource.SOURCE_NAME
      ] = source_lib.make_source_builder(
          bootstrap_current_source.BootstrapCurrentSource,
          runtime_params_type=bootstrap_current_source.RuntimeParams,
      )()
      source_builders[
          bootstrap_current_source.BootstrapCurrentSource.SOURCE_NAME
      ].runtime_params.mode = runtime_params_lib.Mode.ZERO
    if not qei_found:
      source_builders[qei_source_lib.QeiSource.SOURCE_NAME] = (
          source_lib.make_source_builder(
              qei_source_lib.QeiSource,
              runtime_params_type=qei_source_lib.RuntimeParams,
          )()
      )
      source_builders[
          qei_source_lib.QeiSource.SOURCE_NAME
      ].runtime_params.mode = runtime_params_lib.Mode.ZERO
    if not generic_current_found:
      source_builders[
          generic_current_source.GenericCurrentSource.SOURCE_NAME
      ] = source_lib.make_source_builder(
          generic_current_source.GenericCurrentSource,
          runtime_params_type=generic_current_source.RuntimeParams,
      )()
      source_builders[
          generic_current_source.GenericCurrentSource.SOURCE_NAME
      ].runtime_params.mode = runtime_params_lib.Mode.ZERO

    self.source_builders = source_builders

  def __call__(self) -> SourceModels:

    return SourceModels(self.source_builders)

  @property
  def runtime_params(self) -> dict[str, runtime_params_lib.RuntimeParams]:
    """Returns all the runtime params for all sources."""
    return {
        source_name: builder.runtime_params
        for source_name, builder in self.source_builders.items()
    }


def build_all_zero_profiles(
    geo: geometry.Geometry,
    source_models: SourceModels,
) -> source_profiles.SourceProfiles:
  """Returns a SourceProfiles object with all zero profiles."""
  profiles = {
      source_name: jnp.zeros(source_model.output_shape_getter(geo))
      for source_name, source_model in source_models.standard_sources.items()
  }
  return source_profiles.SourceProfiles(
      profiles=profiles,
      j_bootstrap=source_profiles.BootstrapCurrentProfile.zero_profile(geo),
      qei=source_profiles.QeiInfo.zeros(geo),
  )

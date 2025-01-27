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

from collections.abc import Mapping

import jax.numpy as jnp
from torax import array_typing
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.sources import bootstrap_current_source
from torax.sources import generic_current_source
from torax.sources import qei_source as qei_source_lib
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source as source_lib


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
    gas_puff_source = register_source.get_registered_source('gas_puff_source')
    gas_puff_source_builder = source_lib.make_source_builder(
        gas_puff_source.source_class,
        runtime_params_type=gas_puff_source.model_functions['calc_puff_source'].runtime_params_class,
        model_func=gas_puff_source.model_functions['calc_puff_source'].source_profile_function,
    )
    # Define the collection of sources here, which in this example only includes
    # one source.
    all_torax_sources = SourceModels(
        sources={'gas_puff_source': gas_puff_source_builder}
    )

  See runtime_params.py for more details on how to configure all the source/sink
  terms.
  """

  def __init__(
      self,
      source_builders: Mapping[str, source_lib.SourceBuilderProtocol]
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
    if isinstance(
        source, bootstrap_current_source.BootstrapCurrentSource
    ) or isinstance(source, qei_source_lib.QeiSource):
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

  def external_current_source(
      self,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
  ) -> array_typing.ArrayFloat:
    """Returns contributions to external current from all psi sources."""
    total = jnp.zeros_like(geo.rho)
    for source in self.psi_sources.values():
      source_value = source.get_value(
          dynamic_runtime_params_slice=dynamic_runtime_params_slice,
          static_runtime_params_slice=static_runtime_params_slice,
          geo=geo,
          core_profiles=core_profiles,
      )
      total += source.get_source_profile_for_affected_core_profile(
          source_value, source_lib.AffectedCoreProfile.PSI, geo,
      )

    return total

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
          model_func=generic_current_source.calculate_generic_current,
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

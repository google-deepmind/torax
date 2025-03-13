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

from torax.sources import base
from torax.sources import bootstrap_current_source
from torax.sources import generic_current_source
from torax.sources import qei_source as qei_source_lib
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
      sources: Mapping[str, base.SourceModelBase],
  ):
    """Constructs a collection of sources.

    The constructor should only be called by SourceModelsBuilder.

    This class defines which sources are available in a TORAX simulation run.
    Users can configure whether each source is actually on and what kind of
    profile it produces by changing its runtime configuration (see
    runtime_params_lib.py).

    Args:
      sources: Source models config.

    NOTE - Some sources are "special-case": bootstrap_current, generic_current,
    and Qei. SourceModels will always instantiate default objects for these
    types of sources unless they are provided by this `sources` argument.

    Raises:
      ValueError if there is a naming collision with the reserved names as
      described above.
    """
    sources = {
        name: source_config.build_source()
        for name, source_config in sources.items()
    }

    # Some sources are accessed for specific use cases, so we extract those
    # ones and expose them directly.
    self._j_bootstrap = None
    generic_current = None
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
        generic_current = source
      elif isinstance(source, qei_source_lib.QeiSource):
        self._qei_source = source

    # Make sure defaults are set for the "special-case" sources.
    if self._j_bootstrap is None:
      self._j_bootstrap = bootstrap_current_source.BootstrapCurrentSource()
    if self._qei_source is None:
      self._qei_source = qei_source_lib.QeiSource()
    # If the generic current source wasn't provided, create a default one and
    # add to standard sources.
    if generic_current is None:
      generic_current = generic_current_source.GenericCurrentSource()
      self._add_standard_source(
          generic_current_source.GenericCurrentSource.SOURCE_NAME,
          generic_current,
      )

    # Then add all the "standard" sources.
    for source_name, source in sources.items():
      if isinstance(
          source, bootstrap_current_source.BootstrapCurrentSource
      ) or isinstance(source, qei_source_lib.QeiSource):
        continue
      else:
        self._add_standard_source(source_name, source)

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

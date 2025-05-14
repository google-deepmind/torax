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
from torax.neoclassical import pydantic_model as neoclassical_pydantic_model
from torax.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax.neoclassical.conductivity import base as conductivity_base
from torax.sources import bootstrap_current_source
from torax.sources import pydantic_model
from torax.sources import qei_source as qei_source_lib
from torax.sources import source as source_lib


class SourceModels:
  """Source models for the different equations being evolved in Torax."""

  def __init__(
      self,
      sources: pydantic_model.Sources,
      neoclassical: neoclassical_pydantic_model.Neoclassical,
  ):
    """Constructs a collection of sources and neoclassical models.

    This class defines which sources are available in a TORAX simulation run.
    Users can configure whether each source is actually on and what kind of
    profile it produces by changing its runtime configuration (see
    sources.pydantic_model.py).

    It also contains the neoclassical bootstrap current and conductivity models.

    Args:
      sources: Source models config.
      neoclassical: Neoclassical models config.
    """
    self._bootstrap_current = neoclassical.bootstrap_current.build_model()
    self._conductivity = neoclassical.conductivity.build_model()
    self._j_bootstrap = sources.j_bootstrap.build_source()
    self._qei_source = sources.ei_exchange.build_source()
    self._standard_sources = {}
    self._psi_sources = {}

    for k, v in dict(sources).items():
      # skip these as they are handled above
      if k == 'j_bootstrap'  or k == 'ei_exchange':
        continue
      else:
        if v is not None:
          source = v.build_source()
          if k in self._standard_sources.keys():
            raise ValueError(
                f'Trying to add another source with the same name: {k}.'
            )
          self._standard_sources[k] = source
          if (
              source_lib.AffectedCoreProfile.PSI
              in source.affected_core_profiles
          ):
            self._psi_sources[k] = source

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

  # Some sources require direct access, so this class defines properties for
  # those sources.
  @property
  def j_bootstrap(self) -> bootstrap_current_source.BootstrapCurrentSource:
    return self._j_bootstrap

  @property
  def j_bootstrap_name(self) -> str:
    return bootstrap_current_source.BootstrapCurrentSource.SOURCE_NAME

  @property
  def conductivity(self) -> conductivity_base.ConductivityModel:
    return self._conductivity

  @property
  def bootstrap_current(self) -> bootstrap_current_base.BootstrapCurrentModel:
    return self._bootstrap_current

  @property
  def qei_source(self) -> qei_source_lib.QeiSource:
    return self._qei_source

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

  def __hash__(self) -> int:
    hashes = [hash(source) for source in self.standard_sources.values()]
    hashes.append(hash(self.j_bootstrap))
    hashes.append(hash(self.qei_source))
    return hash(tuple(hashes))

  def __eq__(self, other) -> bool:
    if set(self.standard_sources.keys()) == set(other.standard_sources.keys()):
      return (
          all(
              self.standard_sources[name] == other.standard_sources[name]
              for name in self.standard_sources.keys()
          )
          and self.j_bootstrap == other.j_bootstrap
          and self.qei_source == other.qei_source
      )
    return False

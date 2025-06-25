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

"""Source/sink profiles for all the sources in TORAX."""
import dataclasses
from typing import Literal

import chex
import jax
import jax.numpy as jnp
import typing_extensions

from torax._src import constants
from torax._src.geometry import geometry
from torax._src.neoclassical.bootstrap_current import \
    base as bootstrap_current_base

# pylint: disable=invalid-name


@chex.dataclass(frozen=True)
class QeiInfo:
  """Represents the source values coming from a QeiSource."""

  qei_coef: jax.Array
  implicit_ii: jax.Array
  explicit_i: jax.Array
  implicit_ee: jax.Array
  explicit_e: jax.Array
  implicit_ie: jax.Array
  implicit_ei: jax.Array

  @classmethod
  def zeros(cls, geo: geometry.Geometry) -> typing_extensions.Self:
    return QeiInfo(
        qei_coef=jnp.zeros_like(geo.rho),
        implicit_ii=jnp.zeros_like(geo.rho),
        explicit_i=jnp.zeros_like(geo.rho),
        implicit_ee=jnp.zeros_like(geo.rho),
        explicit_e=jnp.zeros_like(geo.rho),
        implicit_ie=jnp.zeros_like(geo.rho),
        implicit_ei=jnp.zeros_like(geo.rho),
    )


@chex.dataclass(frozen=True)
class SourceProfiles:
  """Collection of profiles for all sources in TORAX.

  Most profiles are stored in the attributes relating to the core profile they
  affect, but special-case profiles `j_bootstrap` and `qei` are pulled out into
  their own attributes as these sources need to be treated differently (though
  they could still be set to zero using appropriate runtime params).

  This dataclass is inspired by the IMAS `core_sources` IDS. It is not a 1:1
  mapping to that schema, but it contains similar profiles as you'd expect in
  that IDS.
  """

  # Special-case profiles.
  bootstrap_current: bootstrap_current_base.BootstrapCurrent
  qei: QeiInfo
  # Other profiles organised by the affected core profile. These are the
  # profiles that are used to compute the core profile equations.
  # The form is a dict of jax.Arrays, keyed by the name of the source. The array
  # is the profile on the cell grid from that source for that core profile.
  # For sources that affect multiple core profiles, they will have an entry for
  # each core profile they affect.
  T_e: dict[str, jax.Array] = dataclasses.field(default_factory=dict)
  T_i: dict[str, jax.Array] = dataclasses.field(default_factory=dict)
  n_e: dict[str, jax.Array] = dataclasses.field(default_factory=dict)
  psi: dict[str, jax.Array] = dataclasses.field(default_factory=dict)

  # This function can be jitted if source_models is a static argument. However,
  # in our tests, jitting this function actually slightly slows down runs, so
  # this is left as pure python.
  @classmethod
  def merge(
      cls,
      explicit_source_profiles: typing_extensions.Self,
      implicit_source_profiles: typing_extensions.Self,
  ) -> typing_extensions.Self:
    """Returns a SourceProfiles that merges the input profiles.

    Sources can either be explicit or implicit. The explicit_source_profiles
    contain the profiles for all source models that are set to explicit, and it
    contains profiles with all zeros for any implicit source. The opposite holds
    for the implicit_source_profiles.

    This function adds the two dictionaries of profiles and returns a single
    SourceProfiles that includes both.

    Args:
      explicit_source_profiles: Profiles from explicit source models. This
        SourceProfiles dict will include keys for both the explicit and implicit
        sources, but only the explicit sources will have non-zero profiles. See
        source.py and runtime_params.py for more info on explicit vs. implicit.
      implicit_source_profiles: Profiles from implicit source models. This
        SourceProfiles dict will include keys for both the explicit and implicit
        sources, but only the implicit sources will have non-zero profiles. See
        source.py and runtime_params.py for more info on explicit vs. implicit.

    Returns:
      A SourceProfiles with non-zero profiles for all sources, both explicit and
      implicit (assuming the source model outputted a non-zero profile).

    """
    sum_profiles = lambda a, b: a + b
    return jax.tree_util.tree_map(
        sum_profiles, explicit_source_profiles, implicit_source_profiles
    )

  def total_psi_sources(self, geo: geometry.Geometry) -> jax.Array:
    total = self.bootstrap_current.j_bootstrap
    total += sum(self.psi.values())
    mu0 = constants.CONSTANTS.mu0
    prefactor = 8 * geo.vpr * jnp.pi**2 * geo.B_0 * mu0 * geo.Phi_b / geo.F**2
    return -total * prefactor

  def total_sources(
      self,
      source_type: Literal['n_e', 'T_i', 'T_e'],
      geo: geometry.Geometry,
  ) -> jax.Array:
    source: dict[str, jax.Array] = getattr(self, source_type)
    total = sum(source.values())
    return total * geo.vpr

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

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
from torax.geometry import geometry


@chex.dataclass(frozen=True)
class SourceProfiles:
  """Collection of profiles for all sources in TORAX.

  Most profiles are stored in the `profiles` attribute, but special-case
  profiles `j_bootstrap` and `qei` are pulled out into their own attributes as
  these sources need to exist for a TORAX run (though they could still be set
  to zero using appropriate runtime params).

  The keys of profiles match the keys of the sources in the Sources object used
  to compute them.

  This dataclass is inspired by the IMAS `core_sources` IDS. It is not a 1:1
  mapping to that schema, but it contains similar profiles as you'd expect in
  that IDS.
  """

  profiles: dict[str, jax.Array]
  # Special-case profiles.
  j_bootstrap: BootstrapCurrentProfile
  qei: QeiInfo

  def get_profile(self, name: str) -> jax.Array:
    """Returns the profile, returning zeroes if profile name doesn't exist."""
    if name in self.profiles:
      return self.profiles[name]
    return jnp.zeros_like(self.j_bootstrap.j_bootstrap)

  # This function can be jitted if source_models is a static argument. However,
  # in our tests, jitting this function actually slightly slows down runs, so
  # this is left as pure python.
  @classmethod
  def merge(
      cls,
      explicit_source_profiles: SourceProfiles,
      implicit_source_profiles: SourceProfiles,
  ) -> SourceProfiles:
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
    summed_bootstrap_profile = jax.tree_util.tree_map(
        sum_profiles,
        explicit_source_profiles.j_bootstrap,
        implicit_source_profiles.j_bootstrap,
    )
    summed_qei_info = jax.tree_util.tree_map(
        sum_profiles, explicit_source_profiles.qei, implicit_source_profiles.qei
    )
    summed_other_profiles = jax.tree_util.tree_map(
        sum_profiles,
        explicit_source_profiles.profiles,
        implicit_source_profiles.profiles,
    )
    return cls(
        profiles=summed_other_profiles,
        j_bootstrap=summed_bootstrap_profile,
        qei=summed_qei_info,
    )


@chex.dataclass(frozen=True)
class BootstrapCurrentProfile:
  """Bootstrap current profile.

  Attributes:
    sigma: plasma conductivity with neoclassical corrections on cell grid.
    sigma_face: plasma conductivity with neoclassical corrections on face grid.
    j_bootstrap: Bootstrap current density (Amps / m^2)
    j_bootstrap_face: Bootstrap current density (Amps / m^2) on face grid
    I_bootstrap: Total bootstrap current. Used primarily for diagnostic
      purposes.
  """

  sigma: jax.Array
  sigma_face: jax.Array
  j_bootstrap: jax.Array
  j_bootstrap_face: jax.Array
  I_bootstrap: jax.Array  # pylint: disable=invalid-name

  @classmethod
  def zero_profile(cls, geo: geometry.Geometry) -> BootstrapCurrentProfile:
    return BootstrapCurrentProfile(
        sigma=jnp.zeros_like(geo.rho),
        sigma_face=jnp.zeros_like(geo.rho_face),
        j_bootstrap=jnp.zeros_like(geo.rho),
        j_bootstrap_face=jnp.zeros_like(geo.rho_face),
        I_bootstrap=jnp.zeros(()),
    )


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
  def zeros(cls, geo: geometry.Geometry) -> QeiInfo:
    return QeiInfo(
        qei_coef=jnp.zeros_like(geo.rho),
        implicit_ii=jnp.zeros_like(geo.rho),
        explicit_i=jnp.zeros_like(geo.rho),
        implicit_ee=jnp.zeros_like(geo.rho),
        explicit_e=jnp.zeros_like(geo.rho),
        implicit_ie=jnp.zeros_like(geo.rho),
        implicit_ei=jnp.zeros_like(geo.rho),
    )

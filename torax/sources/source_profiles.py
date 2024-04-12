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
import jax.numpy as jnp
from torax import geometry


@chex.dataclass(frozen=True)
class SourceProfiles:
  """Collection of profiles for all sources in TORAX.

  Most profiles are stored in the `profiles` attribute, but special-case
  profiles are pulled out into their own attributes.

  The keys of profiles match the keys of the sources in the Sources object used
  to compute them.

  This dataclass is inspired by the IMAS `core_sources` IDS. It is not a 1:1
  mapping to that schema, but it contains similar profiles as you'd expect in
  that IDS.
  """

  profiles: dict[str, jnp.ndarray]
  j_bootstrap: BootstrapCurrentProfile
  qei: QeiInfo


@chex.dataclass(frozen=True)
class BootstrapCurrentProfile:
  """Bootstrap current profile.

  Attributes:
    sigma: plasma conductivity with neoclassical corrections on cell grid.
    j_bootstrap: Bootstrap current density (Amps / m^2)
    j_bootstrap_face: Bootstrap current density (Amps / m^2) on face grid
    I_bootstrap: Total bootstrap current. Used primarily for diagnostic
      purposes.
  """

  sigma: jnp.ndarray
  j_bootstrap: jnp.ndarray
  j_bootstrap_face: jnp.ndarray
  I_bootstrap: jnp.ndarray  # pylint: disable=invalid-name

  @classmethod
  def zero_profile(cls, geo: geometry.Geometry) -> BootstrapCurrentProfile:
    return BootstrapCurrentProfile(
        sigma=jnp.zeros_like(geo.r),
        j_bootstrap=jnp.zeros_like(geo.r),
        j_bootstrap_face=jnp.zeros_like(geo.r_face),
        I_bootstrap=jnp.zeros(()),
    )


@chex.dataclass(frozen=True)
class QeiInfo:
  """Represents the source values coming from a QeiSource."""

  qei_coef: jnp.ndarray
  implicit_ii: jnp.ndarray
  explicit_i: jnp.ndarray
  implicit_ee: jnp.ndarray
  explicit_e: jnp.ndarray
  implicit_ie: jnp.ndarray
  implicit_ei: jnp.ndarray

  @classmethod
  def zeros(cls, geo: geometry.Geometry) -> QeiInfo:
    return QeiInfo(
        qei_coef=jnp.zeros_like(geo.r),
        implicit_ii=jnp.zeros_like(geo.r),
        explicit_i=jnp.zeros_like(geo.r),
        implicit_ee=jnp.zeros_like(geo.r),
        explicit_e=jnp.zeros_like(geo.r),
        implicit_ie=jnp.zeros_like(geo.r),
        implicit_ei=jnp.zeros_like(geo.r),
    )

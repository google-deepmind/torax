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

"""Sources for the ne equation."""

from __future__ import annotations

import dataclasses

from jax import numpy as jnp
from torax import geometry
from torax.sources import formulas
from torax.sources import source
from torax.sources import source_config


def calc_puff_source(
    geo: geometry.Geometry,
    puff_decay_length: float,
    S_puff_tot: float,  # pylint: disable=invalid-name
    nref: float,
) -> jnp.ndarray:
  """Calculates external source term for n from puffs."""
  return formulas.exponential_profile(
      c1=1.0,
      c2=puff_decay_length,
      total=S_puff_tot / nref,
      use_normalized_r=True,
      geo=geo,
  )


def calc_nbi_source(
    geo: geometry.Geometry,
    nbi_deposition_location: float,
    nbi_particle_width: float,
    S_nbi_tot: float,  # pylint: disable=invalid-name
    nref: float,
) -> jnp.ndarray:
  """Calculates external source term for n from SBI."""
  return formulas.gaussian_profile(
      c1=nbi_deposition_location,
      c2=nbi_particle_width,
      total=S_nbi_tot / nref,
      use_normalized_r=True,
      geo=geo,
  )


def calc_pellet_source(
    geo: geometry.Geometry,
    pellet_deposition_location: float,
    pellet_width: float,
    S_pellet_tot: float,  # pylint: disable=invalid-name
    nref: float,
) -> jnp.ndarray:
  """Calculates external source term for n from pellets."""
  return formulas.gaussian_profile(
      c1=pellet_deposition_location,
      c2=pellet_width,
      total=S_pellet_tot / nref,
      use_normalized_r=True,
      geo=geo,
  )


@dataclasses.dataclass(frozen=True, kw_only=True)
class GasPuffSource(source.SingleProfileNeSource):
  """Gas puff source for the ne equation."""

  name: str = 'gas_puff_source'

  formula: source_config.SourceProfileFunction = (
      lambda dcs, geo, unused_state: calc_puff_source(
          geo,
          puff_decay_length=dcs.puff_decay_length,
          S_puff_tot=dcs.S_puff_tot,
          nref=dcs.nref,
      )
  )


@dataclasses.dataclass(frozen=True, kw_only=True)
class NBIParticleSource(source.SingleProfileNeSource):
  """Neutral-beam injection source for the ne equation."""

  name: str = 'nbi_particle_source'

  formula: source_config.SourceProfileFunction = (
      lambda dcs, geo, unused_state: calc_nbi_source(
          geo,
          nbi_deposition_location=dcs.nbi_deposition_location,
          nbi_particle_width=dcs.nbi_particle_width,
          S_nbi_tot=dcs.S_nbi_tot,
          nref=dcs.nref,
      )
  )


@dataclasses.dataclass(frozen=True, kw_only=True)
class PelletSource(source.SingleProfileNeSource):
  """Pellet source for the ne equation."""

  name: str = 'pellet_source'

  formula: source_config.SourceProfileFunction = (
      lambda dcs, geo, unused_state: calc_pellet_source(
          geo,
          pellet_deposition_location=dcs.pellet_deposition_location,
          pellet_width=dcs.pellet_width,
          S_pellet_tot=dcs.S_pellet_tot,
          nref=dcs.nref,
      )
  )


# The sources below don't have any source-specific implementations, so their
# bodies are empty. You can refer to their base class to see the implementation.
# We define new classes here to:
#  a) support any future source-specific implementation.
#  b) better readability and human-friendly error messages when debugging.


@dataclasses.dataclass(frozen=True, kw_only=True)
class RecombinationDensitySink(source.SingleProfileNeSource):
  """Recombination sink for the electron density equation."""

  name: str = 'recombination_density_sink'

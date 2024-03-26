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

"""Generic heat source for both ion and electron heat."""

from __future__ import annotations

import dataclasses

import jax
from jax import numpy as jnp
from torax import config_slice
from torax import geometry
from torax import state
from torax.sources import source
from torax.sources import source_config


def calc_generic_heat_source(
    geo: geometry.Geometry,
    rsource: float,
    w: float,
    Ptot: float,
    el_heat_fraction: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Computes ion/electron heat source terms.

  Flexible prescribed heat source term.

  Args:
    geo: Geometry describing the torus.
    rsource: Source Gaussian central location
    w: Gaussian width
    Ptot: total heating
    el_heat_fraction: fraction of heating deposited on electrons

  Returns:
    source_ion: source term for ions.
    source_el: source term for electrons.
  """

  # Many variables throughout this function are capitalized based on physics
  # notational conventions rather than on Google Python style
  # pylint: disable=invalid-name

  # calculate heat profile (face grid)
  Q = jnp.exp(-((geo.r_norm - rsource) ** 2) / (2 * w**2))
  Q_face = jnp.exp(-((geo.r_face_norm - rsource) ** 2) / (2 * w**2))
  # calculate constant prefactor
  C = Ptot / jax.scipy.integrate.trapezoid(geo.vpr_face * Q_face, geo.r_face)

  source_ion = C * Q * (1 - el_heat_fraction)
  source_el = C * Q * el_heat_fraction

  return source_ion, source_el


def _default_formula(
    dynamic_config_slice: config_slice.DynamicConfigSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
) -> jnp.ndarray:
  """Returns the default formula-based ion/electron heat source profile."""
  del core_profiles  # Unused.
  ion, el = calc_generic_heat_source(
      geo,
      dynamic_config_slice.rsource,
      dynamic_config_slice.w,
      dynamic_config_slice.Ptot,
      dynamic_config_slice.el_heat_fraction,
  )
  return jnp.stack([ion, el])


@dataclasses.dataclass(frozen=True, kw_only=True)
class GenericIonElectronHeatSource(source.IonElectronSource):
  """Generic heat source for both ion and electron heat."""

  name: str = 'generic_ion_el_heat_source'

  formula: source_config.SourceProfileFunction = _default_formula

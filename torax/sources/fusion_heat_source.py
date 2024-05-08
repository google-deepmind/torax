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

"""Fusion heat source for both ion and electron heat equations."""

from __future__ import annotations

import dataclasses

import jax
from jax import numpy as jnp
from torax import constants
from torax import geometry
from torax import state
from torax.config import runtime_params_slice
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source


def calc_fusion(
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    nref: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Computes fusion power with the Bosch-Hale parameterization NF 1992.

  Assumes that core_profiles.ni is a 50-50% DT mix

  Args:
    geo: Magnetic geometry.
    core_profiles: Core plasma profiles.
    nref: Reference density.

  Returns:
    Ptot: fusion power in MW.
  """

  t_face = core_profiles.temp_ion.face_value()

  # P [W/m^3] = Efus *1/4 * n^2 * <sigma*v>.
  # <sigma*v> for DT calculated with the Bosch-Hale parameterization NF 1992.
  # T is in keV for the formula

  # Many variables throughout this function are capitalized based on physics
  # notational conventions rather than on Google Python style
  # pylint: disable=invalid-name
  Efus = 17.6 * 1e3 * constants.CONSTANTS.keV2J
  mrc2 = 1124656
  BG = 34.3827
  C1 = 1.17302e-9
  C2 = 1.51361e-2
  C3 = 7.51886e-2
  C4 = 4.60643e-3
  C5 = 1.35e-2
  C6 = -1.0675e-4
  C7 = 1.366e-5

  theta = t_face / (
      1.0
      - (t_face * (C2 + t_face * (C4 + t_face * C6)))
      / (1.0 + t_face * (C3 + t_face * (C5 + t_face * C7)))
  )
  xi = (BG**2 / (4 * theta)) ** (1 / 3)

  # sigmav = <cross section * velocity>, in m^3/s
  # Calculate in log space to avoid overflow/underflow in f32
  logsigmav = (
      jnp.log(C1 * theta)
      + 0.5 * jnp.log(xi / (mrc2 * t_face**3))
      - 3 * xi
      - jnp.log(1e6)
  )

  logPfus = (
      jnp.log(0.25 * Efus)
      + 2 * jnp.log(core_profiles.ni.face_value())
      + logsigmav
      + 2 * jnp.log(nref)
  )

  # [W/m^3]
  Pfus_face = jnp.exp(logPfus)
  Pfus_cell = 0.5 * (Pfus_face[:-1] + Pfus_face[1:])

  # [MW]
  Ptot = (
      jax.scipy.integrate.trapezoid(Pfus_face * geo.vpr_face, geo.r_face) / 1e6
  )

  alpha_fraction = 3.5 / 17.6  # fusion power fraction to alpha particles

  # Fractional fusion power ions/electrons.
  # From Mikkelsen Nucl. Tech. Fusion 237 4 1983
  D1 = 88.0 / core_profiles.temp_el.value
  D2 = jnp.sqrt(D1)
  frac_i = (
      2
      * (
          0.166667 * jnp.log((1.0 - D2 + D1) / (1.0 + 2.0 * D2 + D1))
          + 0.57735026
          * (jnp.arctan(0.57735026 * (2.0 * D2 - 1.0)) + 0.52359874)
      )
      / D1
  )
  frac_e = 1.0 - frac_i
  Pfus_i = Pfus_cell * frac_i * alpha_fraction
  Pfus_e = Pfus_cell * frac_e * alpha_fraction

  return Ptot, Pfus_i, Pfus_e


def fusion_heat_model_func(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    dynamic_source_runtime_params: runtime_params_lib.DynamicRuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
) -> jnp.ndarray:
  del dynamic_source_runtime_params  # Unused.
  # pylint: disable=invalid-name
  _, Pfus_i, Pfus_e = calc_fusion(
      geo, core_profiles, dynamic_runtime_params_slice.numerics.nref
  )
  return jnp.stack((Pfus_i, Pfus_e))
  # pylint: enable=invalid-name


@dataclasses.dataclass(kw_only=True)
class FusionHeatSource(source.IonElectronSource):
  """Fusion heat source for both ion and electron heat."""

  supported_modes: tuple[runtime_params_lib.Mode, ...] = (
      runtime_params_lib.Mode.ZERO,
      runtime_params_lib.Mode.MODEL_BASED,
  )

  model_func: source.SourceProfileFunction = fusion_heat_model_func

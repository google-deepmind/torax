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
import dataclasses
import typing
from typing import Annotated, ClassVar, Literal
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.physics import collisions
from torax._src.sources import base
from torax._src.sources import source
from torax._src.sources import source_profiles
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class FusionHeatSource(source.Source):
  """Fusion heat source for both ion and electron heat."""

  SOURCE_ID: ClassVar[str] = 'fusion'
  AFFECTED_CORE_PROFILES: ClassVar[tuple[source.AffectedCoreProfile, ...]] = (
      source.AffectedCoreProfile.TEMP_ION,
      source.AffectedCoreProfile.TEMP_EL,
  )

  def _get_model_value(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      calculated_source_profiles: source_profiles.SourceProfiles | None = None,
      conductivity: conductivity_base.Conductivity | None = None,
  ) -> tuple[source.SourceProfileElement, ...]:
    """Calculates DT fusion heating power with the Bosch-Hale parameterization NF 1992."""
    del calculated_source_profiles, conductivity
    # If both D and T not present in the main ion mixture, return zero fusion.
    # Otherwise, calculate the fusion power.
    main_ion_names = runtime_params.plasma_composition.main_ion_names
    if not {'D', 'T'}.issubset(main_ion_names):
      return (
          jnp.zeros_like(core_profiles.T_i.value),
          jnp.zeros_like(core_profiles.T_i.value),
      )
    else:
      product = 1.0
      for (
          symbol,
          fraction,
      ) in runtime_params.plasma_composition.main_ion.fractions.items():
        if symbol == 'D' or symbol == 'T':
          product *= fraction
      DT_fraction_product = product

    t_face = core_profiles.T_i.face_value()

    # P [W/m^3] = Efus *1/4 * n^2 * <sigma*v>.
    # <sigma*v> for DT calculated with the Bosch-Hale parameterization NF 1992.
    # T is in keV for the formula
    Efus = 17.6 * 1e3 * constants.CONSTANTS.keV_to_J
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
        jnp.log(DT_fraction_product * Efus)
        + 2 * jnp.log(core_profiles.n_i.face_value())
        + logsigmav
    )

    # [W/m^3]
    Pfus_face = jnp.exp(logPfus)
    Pfus_cell = 0.5 * (Pfus_face[:-1] + Pfus_face[1:])

    alpha_fraction = 3.5 / 17.6  # fusion power fraction to alpha particles

    # Fractional fusion power ions/electrons.
    birth_energy = 3520  # Birth energy of alpha particles is 3.52MeV.
    alpha_mass = 4.002602
    frac_i = collisions.fast_ion_fractional_heating_formula(
        birth_energy,
        typing.cast(array_typing.FloatVectorCell, core_profiles.T_e.value),
        alpha_mass,
    )
    frac_e = 1.0 - frac_i
    Pfus_i = Pfus_cell * frac_i * alpha_fraction
    Pfus_e = Pfus_cell * frac_e * alpha_fraction
    return (Pfus_i, Pfus_e)


class FusionHeatSourceConfig(base.SourceConfigBase):
  """Configuration for the FusionHeatSource."""

  model_name: Annotated[Literal['bosch_hale'], torax_pydantic.JAX_STATIC] = (
      'bosch_hale'
  )

  def build_source(self) -> FusionHeatSource:
    return FusionHeatSource()

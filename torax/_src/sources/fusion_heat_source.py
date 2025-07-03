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
from typing import ClassVar, Literal

import chex
import jax
from jax import numpy as jnp
from torax._src import constants
from torax._src import jax_utils
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.physics import collisions
from torax._src.sources import base
from torax._src.sources import runtime_params as runtime_params_lib
from torax._src.sources import source
from torax._src.sources import source_profiles

# Default value for the model function to be used for the fusion heat
# source. This is also used as an identifier for the model function in
# the default source config for Pydantic to "discriminate" against.
DEFAULT_MODEL_FUNCTION_NAME: str = 'bosch_hale'


def calc_fusion(
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Computes DT fusion power with the Bosch-Hale parameterization NF 1992.

  Args:
    geo: Magnetic geometry.
    core_profiles: Core plasma profiles.
    static_runtime_params_slice: Static runtime params, used to determine the
      existence of deuterium and tritium.
    dynamic_runtime_params_slice: Dynamic runtime params, used to extract the
      D and T densities.

  Returns:
    Tuple of P_total, Pfus_i, Pfus_e: total fusion power in MW, ion and electron
      fusion power densities in W/m^3.
  """

  # If both D and T not present in the main ion mixture, return zero fusion.
  # Otherwise, calculate the fusion power.
  if not {'D', 'T'}.issubset(static_runtime_params_slice.main_ion_names):
    return (
        jnp.array(0.0, dtype=jax_utils.get_dtype()),
        jnp.zeros_like(core_profiles.T_i.value),
        jnp.zeros_like(core_profiles.T_i.value),
    )
  else:
    product = 1.0
    for fraction, symbol in zip(
        dynamic_runtime_params_slice.plasma_composition.main_ion.fractions,
        static_runtime_params_slice.main_ion_names,
    ):
      if symbol == 'D' or symbol == 'T':
        product *= fraction
    DT_fraction_product = product  # pylint: disable=invalid-name

  t_face = core_profiles.T_i.face_value()

  # P [W/m^3] = Efus *1/4 * n^2 * <sigma*v>.
  # <sigma*v> for DT calculated with the Bosch-Hale parameterization NF 1992.
  # T is in keV for the formula

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
      jnp.log(DT_fraction_product * Efus)
      + 2 * jnp.log(core_profiles.n_i.face_value())
      + logsigmav
  )

  # [W/m^3]
  Pfus_face = jnp.exp(logPfus)
  Pfus_cell = 0.5 * (Pfus_face[:-1] + Pfus_face[1:])

  # [MW]
  P_total = (
      jax.scipy.integrate.trapezoid(Pfus_face * geo.vpr_face, geo.rho_face_norm)
      / 1e6
  )

  alpha_fraction = 3.5 / 17.6  # fusion power fraction to alpha particles

  # Fractional fusion power ions/electrons.
  birth_energy = 3520  # Birth energy of alpha particles is 3.52MeV.
  alpha_mass = 4.002602
  frac_i = collisions.fast_ion_fractional_heating_formula(
      birth_energy,
      core_profiles.T_e.value,
      alpha_mass,
  )
  frac_e = 1.0 - frac_i
  Pfus_i = Pfus_cell * frac_i * alpha_fraction
  Pfus_e = Pfus_cell * frac_e * alpha_fraction
  return P_total, Pfus_i, Pfus_e


def fusion_heat_model_func(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    unused_source_name: str,
    core_profiles: state.CoreProfiles,
    unused_calculated_source_profiles: source_profiles.SourceProfiles | None,
    unused_conductivity: conductivity_base.Conductivity | None,
) -> tuple[chex.Array, ...]:
  """Model function for fusion heating."""
  # pylint: disable=invalid-name
  _, Pfus_i, Pfus_e = calc_fusion(
      geo,
      core_profiles,
      static_runtime_params_slice,
      dynamic_runtime_params_slice,
  )
  return (Pfus_i, Pfus_e)
  # pylint: enable=invalid-name


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class FusionHeatSource(source.Source):
  """Fusion heat source for both ion and electron heat."""

  SOURCE_NAME: ClassVar[str] = 'fusion'
  model_func: source.SourceProfileFunction = fusion_heat_model_func

  @property
  def source_name(self) -> str:
    return self.SOURCE_NAME

  @property
  def affected_core_profiles(self) -> tuple[source.AffectedCoreProfile, ...]:
    return (
        source.AffectedCoreProfile.TEMP_ION,
        source.AffectedCoreProfile.TEMP_EL,
    )


class FusionHeatSourceConfig(base.SourceModelBase):
  """Configuration for the FusionHeatSource."""

  model_name: Literal['bosch_hale'] = 'bosch_hale'
  mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED

  @property
  def model_func(self) -> source.SourceProfileFunction:
    return fusion_heat_model_func

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> runtime_params_lib.DynamicRuntimeParams:
    return runtime_params_lib.DynamicRuntimeParams(
        prescribed_values=tuple(
            [v.get_value(t) for v in self.prescribed_values]
        ),
    )

  def build_source(self) -> FusionHeatSource:
    return FusionHeatSource(model_func=self.model_func)

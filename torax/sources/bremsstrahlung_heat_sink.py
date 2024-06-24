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

# Many variables throughout this function are capitalized based on physics
# notational conventions rather than on Google Python style
# pylint: disable=invalid-name

"""Bremsstrahlung heat sink for electron heat equation.."""

import dataclasses

import chex
import jax
from jax import numpy as jnp
from torax import geometry
from torax import state
from torax.config import config_args
from torax.config import runtime_params_slice
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  use_relativistic_correction: bool


@dataclasses.dataclass(kw_only=True)
class RuntimeParams(runtime_params_lib.RuntimeParams):
  use_relativistic_correction: bool = False

  def build_dynamic_params(self, t: chex.Numeric) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(
        **config_args.get_init_kwargs(
            input_config=self,
            output_type=DynamicRuntimeParams,
            t=t,
        )
    )


def calc_bremsstrahlung(
    core_profiles: state.CoreProfiles,
    geo: geometry.Geometry,
    Zeff: float,
    nref: float,
    use_relativistic_correction: bool = False,
) -> tuple[chex.Array, chex.Array]:
  """Calculate the Bremsstrahlung radiation power profile.

  Uses the model from Wesson, John, and David J. Campbell. Tokamaks. Vol. 149.
  An optional correction for relativistic effects from Stott PPCF 2005 can be
  enabled with the flag "use_relativistic_correction".

  Args:
      core_profiles (state.CoreProfiles): core plasma profiles.
      geo (geometry.Geometry): geometry object.
      Zeff (float): effective charge number.
      nref (float): reference density.
      use_relativistic_correction (bool, optional): Set to true to include the
        relativistic correction from Stott. Defaults to False.

  Returns:
      jax.Array: total bremsstrahlung radiation power [MW]
      jax.Array: bremsstrahlung radiation power profile [W/m^3]
  """
  ne20 = (nref / 1e20) * core_profiles.ne.face_value()

  Te_kev = core_profiles.temp_el.face_value()

  P_brem_profile_face: jax.Array = (
      5.35e-3 * Zeff * ne20**2 * jnp.sqrt(Te_kev)
  )  # MW/m^3

  def calc_relativistic_correction() -> jax.Array:
    # Apply the Stott relativistic correction.
    Tm = 511.0  # m_e * c**2 in keV
    correction = (1.0 + 2.0 * Te_kev / Tm) * (
        1.0 + (2.0 / Zeff) * (1.0 - 1.0 / (1.0 + Te_kev / Tm))
    )
    return correction

  # In MW/m^3
  P_brem_profile_face = jnp.where(
      use_relativistic_correction,
      P_brem_profile_face * calc_relativistic_correction(),
      P_brem_profile_face,
  )

  # In W/m^3
  P_brem_profile_cell = geometry.face_to_cell(P_brem_profile_face)*1e6

  # In MW
  P_brem_total = jax.scipy.integrate.trapezoid(
      P_brem_profile_face * geo.vpr_face, geo.r_face
  )
  return P_brem_total, P_brem_profile_cell


def bremsstrahlung_model_func(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    dynamic_source_runtime_params: runtime_params_lib.DynamicRuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
) -> jax.Array:
  """Model function for the Bremsstrahlung heat sink."""
  assert isinstance(dynamic_source_runtime_params, DynamicRuntimeParams)
  _, P_brem_profile = calc_bremsstrahlung(
      core_profiles,
      geo,
      dynamic_runtime_params_slice.plasma_composition.Zeff,
      dynamic_runtime_params_slice.numerics.nref,
      use_relativistic_correction=dynamic_source_runtime_params.use_relativistic_correction,
  )
  # As a sink, the power is negative.
  return -1.0 * P_brem_profile


@dataclasses.dataclass(kw_only=True)
class BremsstrahlungHeatSink(source.SingleProfileTempElSource):
  """Fusion heat source for both ion and electron heat."""

  supported_modes: tuple[runtime_params_lib.Mode, ...] = (
      runtime_params_lib.Mode.ZERO,
      runtime_params_lib.Mode.MODEL_BASED,
  )

  model_func: source.SourceProfileFunction = bremsstrahlung_model_func


BremsstrahlungHeatSinkBuilder = source.make_source_builder(
    BremsstrahlungHeatSink, runtime_params_type=RuntimeParams
)

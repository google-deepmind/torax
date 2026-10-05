# Copyright 2026 DeepMind Technologies Limited
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
"""Scaled-profile ICRH model with magnetic-field-dependent resonance shift."""

import dataclasses
from typing import Annotated, Literal

import chex
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import math_utils
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.sources import runtime_params as source_runtime_params_lib
from torax._src.sources import source as source_lib
from torax._src.sources import source_profiles
from torax._src.sources.ion_cyclotron_source import base
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(source_runtime_params_lib.RuntimeParams):
  """Runtime parameters for the scaled-profile ICRH model."""

  P_total: array_typing.FloatScalar
  absorption_fraction: array_typing.FloatScalar
  heat_profile_ion: array_typing.FloatVector
  heat_profile_electron: array_typing.FloatVector
  reference_B0: array_typing.FloatScalar


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class ScaledProfileIonCyclotronSource(base.IonCyclotronSource):
  """ICRH source with prescribed, B-field-shiftable profiles."""

  def _get_model_value(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      calculated_source_profiles: source_profiles.SourceProfiles | None = None,
      conductivity: conductivity_base.Conductivity | None = None,
  ) -> tuple[source_lib.SourceProfileElement, ...]:
    """Compute ICRH heating from prescribed profiles with B-field shift.

    The model performs two operations on the reference profiles:
    1. **Radial shift**: The ICRH resonance location in major radius scales as
       R_res ∝ B₀. When B₀ differs from the reference field, the resonance
       moves, and the heating profile shifts accordingly in normalised radius.
    2. **Power normalisation**: The shifted profiles are rescaled so the
       volume-integrated total heating equals ``P_total * absorption_fraction``.

    Args:
      runtime_params: Full simulation runtime parameters.
      geo: Magnetic geometry.
      core_profiles: Core plasma profiles (unused by this model).
      calculated_source_profiles: Unused.
      conductivity: Unused.

    Returns:
      Tuple of (ion_heating, electron_heating, fast_ions). Fast_ions are
      all zeros for now. Scaled fast ion profiles will be supported in a future
      version.
    """
    del core_profiles, calculated_source_profiles, conductivity
    source_params = runtime_params.sources[self.SOURCE_ID]
    assert isinstance(source_params, RuntimeParams)

    ref_ion = source_params.heat_profile_ion
    ref_el = source_params.heat_profile_electron

    # --- 1. Compute resonance shift ---
    # The ICRH resonance occurs where ω = n·ω_ci(R) and since B_t ∝ 1/R,
    # the resonance major radius scales linearly with B₀.
    # B_ratio > 1 means stronger field → resonance moves outward in R.
    B_ratio = geo.B_0 / source_params.reference_B0

    # Outboard midplane major radius on the cell grid:
    # R_out(ρ) = R_major + r(ρ).
    # This is monotonically increasing with ρ, unlike the flux-surface-averaged
    # R_major_profile which can be constant (e.g. circular geometry).
    R_out = geo.R_out
    rho = geo.torax_mesh.cell_centers

    # Shifted major radius for each grid point.
    R_shifted = R_out / B_ratio

    # Map back to normalised radius: for each shifted R, find the
    # corresponding ρ on the original R_out(ρ) curve.
    rho_shifted = jnp.interp(R_shifted, R_out, rho)

    # Evaluate reference profiles at the shifted ρ positions.
    shifted_ion = jnp.interp(rho_shifted, rho, ref_ion)
    shifted_el = jnp.interp(rho_shifted, rho, ref_el)

    # --- 2. Normalise to target power ---
    absorbed_power = source_params.P_total * source_params.absorption_fraction
    total_shape = shifted_ion + shifted_el
    integrated = math_utils.volume_integration(total_shape, geo)
    # Guard against zero integrated power (e.g. if profiles are all zero).
    # Use safe denominator to avoid NaN gradients in dead jnp.where branches.
    safe_integrated = jnp.where(integrated > 0, integrated, 1.0)
    scale = jnp.where(integrated > 0, absorbed_power / safe_integrated, 0.0)

    source_ion = shifted_ion * scale
    source_el = shifted_el * scale

    # --- 3. Default zero fast ions in build_fast_ions ---
    # TODO(b/508118026): extend to support scaled prescribed fast ion profiles.
    fast_ions = base.build_fast_ions(source_name=self.SOURCE_ID, geo=geo)

    return (source_ion, source_el, fast_ions)


class ScaledProfileIonCyclotronSourceConfig(base.IonCyclotronSourceConfig):
  """Configuration for ICRH with prescribed, B-field-shiftable profiles.

  This model takes reference ion and electron heating profiles and:
  1. Shifts them radially based on the ratio of the actual vacuum toroidal
     magnetic field to a reference field (``B₀ / reference_B0``).
  2. Rescales the amplitude so that the volume-integrated total heating
     equals ``P_total * absorption_fraction``.

  This is useful when computed reference heating profiles are available at a
  specific magnetic field, and you need to approximate them to different
  operating points without re-running the full RF solver.

  Attributes:
    model_name: Discriminator literal for Pydantic.
    heat_profile_ion: Reference ion heating power density shape [W/m³],
      provided on a normalised toroidal coordinate grid as a TimeVaryingArray.
    heat_profile_electron: Reference electron heating power density shape
      [W/m³], provided on a normalised toroidal coordinate grid as a
      TimeVaryingArray.
    reference_B0: Vacuum toroidal magnetic field at which the reference
      profiles were computed [T].
  """

  model_name: Annotated[
      Literal['scaled_profile'], torax_pydantic.JAX_STATIC
  ] = 'scaled_profile'
  heat_profile_ion: torax_pydantic.TimeVaryingArray = (
      torax_pydantic.ValidatedDefault({0: {0: 0, 1: 0}})
  )
  heat_profile_electron: torax_pydantic.TimeVaryingArray = (
      torax_pydantic.ValidatedDefault({0: {0: 0, 1: 0}})
  )
  reference_B0: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(12.2)
  )

  def build_runtime_params(
      self,
      t: chex.Numeric,
  ) -> RuntimeParams:
    return RuntimeParams(
        **dataclasses.asdict(super().build_runtime_params(t)),
        P_total=self.P_total.get_value(t),
        absorption_fraction=self.absorption_fraction.get_value(t),
        heat_profile_ion=self.heat_profile_ion.get_value(t),
        heat_profile_electron=self.heat_profile_electron.get_value(t),
        reference_B0=self.reference_B0.get_value(t),
    )

  def build_source(self) -> ScaledProfileIonCyclotronSource:
    return ScaledProfileIonCyclotronSource()

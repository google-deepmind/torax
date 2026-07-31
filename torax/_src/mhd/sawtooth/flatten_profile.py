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

"""Routines for flattening profiles in redistribution models."""

import dataclasses

from jax import numpy as jnp
from torax._src import array_typing
from torax._src import math_utils
from torax._src.core_profiles import initialization
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry


def flatten_density_profile(
    rho_norm_q1: array_typing.FloatScalar,
    rho_norm_mixing: array_typing.FloatScalar,
    redistribution_mask: array_typing.BoolVector,
    flattening_factor: array_typing.FloatScalar,
    original_density_profile: cell_variable.CellVariable,
    geo: geometry.Geometry,
) -> cell_variable.CellVariable:
  """Redistributes a density profile while preserving total particle number.

  This function redistributes a profile due to a sawtooth crash by modifying
  the profile from the magnetic axis up to the mixing radius. The profile is
  (roughly) flattened between the magnetic axis and the q=1 surface. Between
  the q=1 surface to the mixing radius, the profile is linearly redistributed.
  The original profile value is maintained at the mixing radius.

  The unknown quantity is the value of the redistributed profile at the q=1
  surface. This is calculated by ensuring that volume integrals are conserved,
  e.g. for conservation of particles, energy, currents.

  Args:
    rho_norm_q1: The normalized radius of the q=1 surface.
    rho_norm_mixing: The normalized radius of the mixing surface.
    redistribution_mask: boolean mask for the redistribution zone inside the
      mixing radius.
    flattening_factor: The factor by which the profile is flattened.
    original_density_profile: The original density profile to be redistributed.
    geo: The geometry of the simulation at this time slice.

  Returns:
    The redistributed density profile.
  """

  original_density = original_density_profile.value
  ones = jnp.ones_like(original_density)

  new_profile = _redistribute_profile(
      rho_norm_q1=rho_norm_q1,
      rho_norm_mixing=rho_norm_mixing,
      redistribution_mask=redistribution_mask,
      flattening_factor=flattening_factor,
      original_profile=original_density,
      geo=geo,
      pre_crash_weight=ones,
      post_crash_weight=ones,
  )

  return dataclasses.replace(
      original_density_profile,
      value=new_profile,
  )


def flatten_temperature_profile(
    rho_norm_q1: array_typing.FloatScalar,
    rho_norm_mixing: array_typing.FloatScalar,
    redistribution_mask: array_typing.BoolVector,
    flattening_factor: array_typing.FloatScalar,
    original_temperature_profile: cell_variable.CellVariable,
    original_density_profile: cell_variable.CellVariable,
    flattened_density_profile: cell_variable.CellVariable,
    geo: geometry.Geometry,
) -> cell_variable.CellVariable:
  """Redistributes a temperature profile while preserving total energy.

  The integral of density*temperature is proportional to total energy.

  This function redistributes a profile due to a sawtooth crash by modifying
  the profile from the magnetic axis up to the mixing radius. The profile is
  (roughly) flattened between the magnetic axis and the q=1 surface. Between
  the q=1 surface to the mixing radius, the profile is linearly redistributed.
  The original profile value is maintained at the mixing radius.

  The unknown quantity is the value of the redistributed profile at the q=1
  surface. This is calculated by ensuring that volume integrals are conserved,
  e.g. for conservation of particles, energy, currents.

  Args:
    rho_norm_q1: The normalized radius of the q=1 surface.
    rho_norm_mixing: The normalized radius of the mixing surface.
    redistribution_mask: boolean mask for the redistribution zone inside the
      mixing radius.
    flattening_factor: The factor by which the profile is flattened.
    original_temperature_profile: The original temperature profile to be
      redistributed.
    original_density_profile: The original density profile.
    flattened_density_profile: The already redistributed density profile.
    geo: The geometry of the simulation at this time slice.

  Returns:
    The redistributed temperature profile.
  """

  original_temperature = original_temperature_profile.value
  original_density = original_density_profile.value
  flattened_density = flattened_density_profile.value

  # Weight by densities for energy conservation.
  new_temperature = _redistribute_profile(
      rho_norm_q1=rho_norm_q1,
      rho_norm_mixing=rho_norm_mixing,
      redistribution_mask=redistribution_mask,
      flattening_factor=flattening_factor,
      original_profile=original_temperature,
      geo=geo,
      pre_crash_weight=original_density,
      post_crash_weight=flattened_density,
  )

  return dataclasses.replace(
      original_temperature_profile,
      value=new_temperature,
  )


# pylint: disable=invalid-name
def flatten_current_profile(
    rho_norm_q1: array_typing.FloatScalar,
    rho_norm_mixing: array_typing.FloatScalar,
    redistribution_mask: array_typing.BoolVector,
    flattening_factor: array_typing.FloatScalar,
    original_psi_profile: cell_variable.CellVariable,
    original_j_total_profile: array_typing.FloatVector,
    Ip_total: array_typing.FloatScalar,
    geo: geometry.Geometry,
) -> cell_variable.CellVariable:
  """Redistributes a poloidal flux profile while preserving total current.

  This function redistributes a profile due to a sawtooth crash by modifying
  the profile from the magnetic axis up to the mixing radius. The profile is
  (roughly) flattened between the magnetic axis and the q=1 surface. Between
  the q=1 surface to the mixing radius, the profile is linearly redistributed.
  The original profile value is maintained at the mixing radius.

  The unknown quantity is the value of the redistributed profile at the q=1
  surface. This is calculated by ensuring that volume integrals are conserved,
  e.g. for conservation of particles, energy, currents.

  Args:
    rho_norm_q1: The normalized radius of the q=1 surface.
    rho_norm_mixing: The normalized radius of the mixing surface.
    redistribution_mask: boolean mask for the redistribution zone inside the
      mixing radius.
    flattening_factor: The factor by which the profile is flattened.
    original_psi_profile: The original poloidal flux profile.
    original_j_total_profile: The original j_total profile already precalculated
      and consistent with the psi profile.
    Ip_total: The total plasma current [A].
    geo: The geometry of the simulation at this time slice.

  Returns:
    The redistributed poloidal flux profile.
  """

  ones = jnp.ones_like(original_j_total_profile)
  new_j_total = _redistribute_profile(
      rho_norm_q1=rho_norm_q1,
      rho_norm_mixing=rho_norm_mixing,
      redistribution_mask=redistribution_mask,
      flattening_factor=flattening_factor,
      original_profile=original_j_total_profile,
      geo=geo,
      pre_crash_weight=ones,
      post_crash_weight=ones,
  )

  # Construct a new psi profile using the new j_total profile.
  # Since we will need to use a hires j_total profile, we expect a minor
  # deviation from the conserved current.
  # TODO(b/317360481). Come up with a better way to conserve current through
  # the j-->psi conversion.

  new_j_total_hires = jnp.interp(geo.rho_hires_norm, geo.rho_norm, new_j_total)

  new_psi = initialization.update_psi_from_j(Ip_total, geo, new_j_total_hires)

  # Shift the new psi profile to match the original psi profile at the
  # face boundary.
  new_psi = (
      new_psi.value
      - new_psi.face_value()[-1]  # pyrefly: ignore[bad-index]
      + original_psi_profile.face_value()[-1]  # pyrefly: ignore[bad-index]
  )

  return dataclasses.replace(
      original_psi_profile,
      value=new_psi,
  )


def _redistribute_profile(
    rho_norm_q1: array_typing.FloatScalar,
    rho_norm_mixing: array_typing.FloatScalar,
    redistribution_mask: array_typing.BoolVector,
    flattening_factor: array_typing.FloatScalar,
    original_profile: array_typing.FloatVector,
    geo: geometry.Geometry,
    pre_crash_weight: array_typing.FloatVector,
    post_crash_weight: array_typing.FloatVector,
) -> array_typing.FloatVector:
  """Redistributes a profile inside the mixing radius due to a sawtooth crash.

  Example redistributed profiles are plasma current, density, or temperature.

  The resulting redistributed profile looks as follows:
  1. Between the magnetic axis and the q=1 radius (`rho_norm_q1`), the
     profile is either completely flat (when `flattening_factor == 1.0`) or
     has a linear gradient (when `flattening_factor > 1.0`).
  2. Between the q=1 radius and the mixing radius (`rho_norm_mixing`), the
     profile transitions linearly to match the original profile value at the
     mixing radius.
  3. Outside the mixing radius, the profile is unchanged.

  The overall value of the profile at the q=1 surface is scaled so that the
  integrated quantity (e.g. total particle number, thermal energy, or enclosed
  current) is conserved. The weighting profiles (`pre_crash_weight` and
  `post_crash_weight`) can be used to conserve a weighted integral.
  For example, for energy conservation when the temperature profile is
  flattened, the density profile should be used as a weighting profile.

  Derivation:
    Let y_orig(rho) be the original profile before the crash, and
    y_mix = y_orig(rho_mix) be the value of the profile at the mixing radius.
    Let w_pre(rho) and w_post(rho) be pre- and post-crash weighting profiles.

    The conserved original integrated quantity inside the mixing radius is:
      I_orig = int_{rho <= rho_mix} y_orig(rho) * w_pre(rho) dV

    Let C be the redistributed scale value at the q=1 surface (rho_norm_q1).
    Inside q=1 (rho <= rho_q1), we define:
      r_hat(rho) = rho / rho_q1
      flat_zone_shape(rho) = 1 + (flattening_factor - 1) * (1 - r_hat(rho))
      y_new(rho) = C * flat_zone_shape(rho)

    In the mixing zone (rho_q1 < rho <= rho_mix), we define normalized
    coordinate xi(rho) in [0, 1] from rho_q1 to rho_mix:
      xi(rho) = (rho - rho_q1) / (rho_mix - rho_q1)
      y_new(rho) = C * (1 - xi(rho)) + y_mix * xi(rho)

    The redistributed integrated quantity inside the mixing radius is:
      I_new = int_{rho <= rho_mix} y_new(rho) * w_post(rho) dV
            = C * (I_flat + I_mix) + y_mix * I_edge

    where I_core, I_mix, and I_edge are volume integrals of the
    weighted shape functions:
      I_flat    = int_{rho <= rho_q1} flat_zone_shape(rho) * w_post(rho) dV
      I_mix     = int_{rho_q1 < rho <= rho_mix} (1 - xi(rho)) * w_post(rho) dV
      I_edge    = int_{rho_q1 < rho <= rho_mix} xi(rho) * w_post(rho) dV

    Setting I_new = I_orig, we obtain a linear analytical solution for C:
      C = (I_orig - y_mix * I_edge) / (I_flat + I_mix)

  Args:
    rho_norm_q1: The normalized radius of the q=1 surface.
    rho_norm_mixing: The normalized radius of the mixing surface.
    redistribution_mask: boolean mask for cells inside rho_norm_mixing.
    flattening_factor: Factor controlling core gradient (1.0 for flat core).
    original_profile: The original 1D profile array to be redistributed.
    geo: The geometry of the simulation at this time slice.
    pre_crash_weight: Weighting profile for the original profile integral (e.g.
      original density when flattening temperature).
    post_crash_weight: Weighting profile for redistributed shape integrals (e.g.
      redistributed density when flattening temperature).

  Returns:
    The redistributed 1D profile array.
  """

  rho_norm = geo.rho_norm
  value_at_mixing_radius = jnp.interp(
      rho_norm_mixing, rho_norm, original_profile
  )

  # Zone masks.
  flat_zone_mask = rho_norm <= rho_norm_q1
  mixing_zone_mask = redistribution_mask & ~flat_zone_mask

  # Normalized coordinate xi in [0, 1] from rho_q1 to rho_mix.
  mixing_zone_coord_norm = jnp.where(
      mixing_zone_mask,
      (rho_norm - rho_norm_q1) / (rho_norm_mixing - rho_norm_q1),
      0.0,
  )

  # Flat zone shape: ranges from flattening_factor on-axis to 1.0 at rho_q1.
  flat_zone_shape = jnp.where(
      flat_zone_mask,
      1.0 + (flattening_factor - 1.0) * (1.0 - rho_norm / rho_norm_q1),
      0.0,
  )

  # Weight contributed by C (unknown value at q=1) in flat and mixing zones:
  core_scale_weight = flat_zone_shape + jnp.where(
      mixing_zone_mask, 1.0 - mixing_zone_coord_norm, 0.0
  )
  # Weight contributed by value_at_mixing_radius: only in mixing zone.
  edge_scale_weight = mixing_zone_coord_norm

  # Integrals for analytical solution of C (profile value at q=1).
  core_shape_integral = math_utils.volume_integration(
      core_scale_weight * post_crash_weight, geo
  )
  edge_shape_integral = math_utils.volume_integration(
      edge_scale_weight * post_crash_weight, geo
  )
  original_integrated_quantity = math_utils.volume_integration(
      jnp.where(redistribution_mask, original_profile * pre_crash_weight, 0.0),
      geo,
  )

  # Analytical solution for C (profile value at q=1).
  redistributed_value_q1 = (
      original_integrated_quantity
      - value_at_mixing_radius * edge_shape_integral
  ) / core_shape_integral

  # Reconstruct profile using the same shape functions.
  new_profile = (
      redistributed_value_q1 * core_scale_weight
      + value_at_mixing_radius * edge_scale_weight
  )
  return jnp.where(redistribution_mask, new_profile, original_profile)

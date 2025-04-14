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
from torax import array_typing
from torax import math_utils
from torax.core_profiles import initialization
from torax.fvm import cell_variable
from torax.geometry import geometry


def flatten_density_profile(
    rho_norm_q1: array_typing.ScalarFloat,
    rho_norm_mixing: array_typing.ScalarFloat,
    redistribution_mask: array_typing.ArrayBool,
    flattening_factor: array_typing.ScalarFloat,
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
    rho_norm_q1: The normalised radius of the q=1 surface.
    rho_norm_mixing: The normalised radius of the mixing surface.
    redistribution_mask: boolean mask for the redistribution region inside
      the mixing radius.
    flattening_factor: The factor by which the profile is flattened.
    original_density_profile: The original density profile to be redistributed.
    geo: The geometry of the simulation at this time slice.

  Returns:
    The redistributed density profile.
  """

  original_density = original_density_profile.value
  rho_norm = geo.rho_norm

  # Get a trial profile shape for the redistributed value and a boolean mask
  # for the redistribution region.
  trial_density = _get_trial_profile(
      rho_norm_q1,
      rho_norm_mixing,
      flattening_factor,
      original_density,
      geo,
  )

  density_at_mixing_edge = jnp.interp(
      rho_norm_mixing, rho_norm, original_density
  )
  scaling = _get_scaling_factor(
      value_at_mixing_edge=density_at_mixing_edge,
      original_profile=original_density,
      trial_profile=trial_density,
      redistribution_mask=redistribution_mask,
      geo=geo,
  )

  # Build the new value using masks and the scaling factor
  new_profile = jnp.where(
      redistribution_mask,
      (trial_density - density_at_mixing_edge) * scaling
      + density_at_mixing_edge,
      original_density,
  )

  return dataclasses.replace(
      original_density_profile,
      value=new_profile,
  )


def flatten_temperature_profile(
    rho_norm_q1: array_typing.ScalarFloat,
    rho_norm_mixing: array_typing.ScalarFloat,
    redistribution_mask: array_typing.ArrayBool,
    flattening_factor: array_typing.ScalarFloat,
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
    rho_norm_q1: The normalised radius of the q=1 surface.
    rho_norm_mixing: The normalised radius of the mixing surface.
    redistribution_mask: boolean mask for the redistribution region inside
      the mixing radius.
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
  rho_norm = geo.rho_norm

  # Get a trial profile shape for the redistributed value and a boolean mask
  # for the redistribution region.
  trial_temperature = _get_trial_profile(
      rho_norm_q1,
      rho_norm_mixing,
      flattening_factor,
      original_temperature,
      geo,
  )

  original_pressure = original_temperature * original_density

  pressure_at_mixing_edge = jnp.interp(
      rho_norm_mixing,
      rho_norm,
      original_pressure,
  )

  trial_pressure = trial_temperature * flattened_density

  scaling = _get_scaling_factor(
      value_at_mixing_edge=pressure_at_mixing_edge,
      original_profile=original_pressure,
      trial_profile=trial_pressure,
      redistribution_mask=redistribution_mask,
      geo=geo,
  )

  new_pressure = (
      trial_pressure - pressure_at_mixing_edge
  ) * scaling + pressure_at_mixing_edge

  # Build the new value using masks and the scaling factor
  new_temperature = jnp.where(
      redistribution_mask,
      new_pressure / flattened_density,
      original_temperature,
  )

  return dataclasses.replace(
      original_temperature_profile,
      value=new_temperature,
  )


# pylint: disable=invalid-name
def flatten_current_profile(
    rho_norm_q1: array_typing.ScalarFloat,
    rho_norm_mixing: array_typing.ScalarFloat,
    redistribution_mask: array_typing.ArrayBool,
    flattening_factor: array_typing.ScalarFloat,
    original_psi_profile: cell_variable.CellVariable,
    original_jtot_profile: array_typing.ArrayFloat,
    Ip_total: array_typing.ScalarFloat,
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
    rho_norm_q1: The normalised radius of the q=1 surface.
    rho_norm_mixing: The normalised radius of the mixing surface.
    redistribution_mask: boolean mask for the redistribution region inside
      the mixing radius.
    flattening_factor: The factor by which the profile is flattened.
    original_psi_profile: The original poloidal flux profile.
    original_jtot_profile: The original jtot profile already precalculated and
      consistent with the psi profile.
    Ip_total: The total plasma current.
    geo: The geometry of the simulation at this time slice.

  Returns:
    The redistributed temperature profile.
  """

  rho_norm = geo.rho_norm

  # Get a trial profile shape for the redistributed value and a boolean mask
  # for the redistribution region.
  trial_jtot = _get_trial_profile(
      rho_norm_q1,
      rho_norm_mixing,
      flattening_factor,
      original_jtot_profile,
      geo,
  )

  jtot_at_mixing_edge = jnp.interp(
      rho_norm_mixing,
      rho_norm,
      original_jtot_profile,
  )

  scaling = _get_scaling_factor(
      value_at_mixing_edge=jtot_at_mixing_edge,
      original_profile=original_jtot_profile,
      trial_profile=trial_jtot,
      redistribution_mask=redistribution_mask,
      geo=geo,
  )

  new_jtot = (trial_jtot - jtot_at_mixing_edge) * scaling + jtot_at_mixing_edge

  # Build the new value using masks and the scaling factor
  new_jtot = jnp.where(
      redistribution_mask,
      new_jtot,
      original_jtot_profile,
  )

  # Construct a new psi profile using the new jtot profile.
  # Since we will need to use a hires jtot profile, we expect a minor deviation
  # from the conserved current.
  # TODO(b/317360481). Come up with a better way to conserve current through
  # the j-->psi conversion.

  new_jtot_hires = jnp.interp(geo.rho_hires_norm, geo.rho_norm, new_jtot)

  new_psi = initialization.update_psi_from_j(Ip_total, geo, new_jtot_hires)

  # Shift the new psi profile to match the original psi profile at the
  # cell boundary.
  new_psi = new_psi.value - new_psi.value[-1] + original_psi_profile.value[-1]

  return dataclasses.replace(
      original_psi_profile,
      value=new_psi,
  )


def _get_trial_profile(
    rho_norm_q1: array_typing.ScalarFloat,
    rho_norm_mixing: array_typing.ScalarFloat,
    flattening_factor: array_typing.ScalarFloat,
    original_profile: array_typing.ArrayFloat,
    geo: geometry.Geometry,
) -> array_typing.ArrayFloat:
  """Returns a trial new value using two linear interpolations."""

  rho_norm = geo.rho_norm

  # Construct a trial new value using two linear interpolations in
  # the redistribution region.
  value_at_q1 = jnp.interp(rho_norm_q1, rho_norm, original_profile)
  value_at_axis = flattening_factor * value_at_q1
  value_at_mixing_edge = jnp.interp(rho_norm_mixing, rho_norm, original_profile)

  # Define the key points for interpolation for the trial new value
  interp_rhos = jnp.array([0.0, rho_norm_q1, rho_norm_mixing])
  interp_vals = jnp.array([value_at_axis, value_at_q1, value_at_mixing_edge])

  # Create the trial value shape.
  # This will do constant extrapolation for rho_norm > rho_norm_mixing , but
  # it doesn't matter since the redistribution mask will be false for these
  # points. It is important for trial_profile to have the same shape as
  # original_profile to avoid unnecessary slicing operations.
  trial_profile = jnp.interp(rho_norm, interp_rhos, interp_vals)

  return trial_profile


def _get_scaling_factor(
    original_profile: array_typing.ArrayFloat,
    trial_profile: array_typing.ArrayFloat,
    value_at_mixing_edge: array_typing.ScalarFloat,
    redistribution_mask: array_typing.ArrayBool,
    geo: geometry.Geometry,
) -> array_typing.ScalarFloat:
  """Returns a profile scaling factor based on integral conservation."""

  # Use where mask for integration regions. Shift by mixing value to
  # calculate correct scaling factor
  shifted_original_integrand = jnp.where(
      redistribution_mask, original_profile - value_at_mixing_edge, 0.0
  )
  shifted_trial_integrand = jnp.where(
      redistribution_mask, trial_profile - value_at_mixing_edge, 0.0
  )
  original_integral = math_utils.volume_integration(
      shifted_original_integrand, geo
  )
  trial_integral = math_utils.volume_integration(shifted_trial_integrand, geo)

  return original_integral / trial_integral

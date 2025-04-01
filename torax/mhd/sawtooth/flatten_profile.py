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
from torax.fvm import cell_variable
from torax.geometry import geometry


def flatten_profile(
    rho_norm_q1: array_typing.ScalarFloat,
    rho_norm_mixing: array_typing.ScalarFloat,
    flattening_factor: array_typing.ScalarFloat,
    original_profile: cell_variable.CellVariable,
    geo: geometry.Geometry,
) -> cell_variable.CellVariable:
  """Helper to redistribute profiles and conserve integral quantities.

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
    flattening_factor: The factor by which the profile is flattened.
    original_profile: The original profile to be redistributed.
    geo: The geometry of the simulation at this time slice.

  Returns:
    The redistributed profile.
  """

  original_value = original_profile.value
  rho_norm = geo.rho_norm
  idx_mixing = jnp.searchsorted(rho_norm, rho_norm_mixing, side='left')

  # Construct masks for different profile domains.
  # The redistribution mask is for all cells up to the mixing radius, since
  # those are the only locations where the modified values contribute to the
  # volume integral.
  indices = jnp.arange(rho_norm.shape[0])
  redistribution_mask = indices < idx_mixing

  # Construct a trial new value using two linear interpolations in
  # the redistribution region.
  value_at_q1 = jnp.interp(rho_norm_q1, rho_norm, original_value)
  value_at_axis = flattening_factor * value_at_q1
  value_at_mixing_edge = jnp.interp(rho_norm_mixing, rho_norm, original_value)

  # Define the key points for interpolation for the trial new value
  interp_rhos = jnp.array([0.0, rho_norm_q1, rho_norm_mixing])
  interp_vals = jnp.array([value_at_axis, value_at_q1, value_at_mixing_edge])

  # Create the trial value shape.
  # This will do constant extrapolation for rho_norm > rho_norm_mixing , but
  # it doesn't matter since the redistribution mask will be false for these
  # points. It is important for trial_value to have the same shape as
  # original_value to avoid unnecessary slicing operations.
  trial_value = jnp.interp(rho_norm, interp_rhos, interp_vals)

  # Use where mask for integration regions. Shift by mixing value to
  # calculate correct scaling factor
  shifted_original_integrand = jnp.where(
      redistribution_mask, original_value - value_at_mixing_edge, 0.0
  )
  shifted_trial_integrand = jnp.where(
      redistribution_mask, trial_value - value_at_mixing_edge, 0.0
  )
  original_integral = math_utils.volume_integration(
      shifted_original_integrand, geo
  )
  trial_integral = math_utils.volume_integration(shifted_trial_integrand, geo)

  scaling = original_integral / trial_integral

  # Build the new value using masks and the scaling factor
  new_value = jnp.where(
      redistribution_mask,
      shifted_trial_integrand * scaling + value_at_mixing_edge,
      original_value,
  )

  return dataclasses.replace(
      original_profile,
      value=new_value,
  )

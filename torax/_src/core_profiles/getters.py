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

"""Functions for getting updated CellVariable objects for CoreProfiles."""

import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import math_utils
from torax._src.core_profiles import runtime_params as core_profile_runtime_params
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.physics import psi_calculations

_trapz = jax.scipy.integrate.trapezoid

# pylint: disable=invalid-name


def get_updated_ion_temperature(
    profile_conditions_params: core_profile_runtime_params.RuntimeParams,
    geo: geometry.Geometry,
    only_boundary_condition: bool = False,
    original_T_i_value: cell_variable.CellVariable | None = None,
) -> cell_variable.CellVariable:
  """Gets initial and/or prescribed ion temperature profiles."""
  if only_boundary_condition:
    if original_T_i_value is None:
      raise ValueError(
          'original_T_i_value must be provided when only updating the boundary'
          ' condition.'
      )
    value = original_T_i_value.value
  else:
    value = profile_conditions_params.T_i
  T_i = cell_variable.CellVariable(
      value=value,
      face_centers=geo.rho_face_norm,
      left_face_grad_constraint=jnp.zeros(()),
      right_face_grad_constraint=None,
      right_face_constraint=profile_conditions_params.T_i_right_bc,
  )
  return T_i


def get_updated_electron_temperature(
    profile_conditions_params: core_profile_runtime_params.RuntimeParams,
    geo: geometry.Geometry,
    only_boundary_condition: bool = False,
    original_T_e_value: cell_variable.CellVariable | None = None,
) -> cell_variable.CellVariable:
  """Gets initial and/or prescribed electron temperature profiles."""
  if only_boundary_condition:
    if original_T_e_value is None:
      raise ValueError(
          'original_T_e_value must be provided when only updating the boundary'
          ' condition.'
      )
    value = original_T_e_value.value
  else:
    value = profile_conditions_params.T_e

  T_e = cell_variable.CellVariable(
      value=value,
      face_centers=geo.rho_face_norm,
      left_face_grad_constraint=jnp.zeros(()),
      right_face_grad_constraint=None,
      right_face_constraint=profile_conditions_params.T_e_right_bc,
  )
  return T_e


def get_updated_electron_density(
    profile_conditions_params: core_profile_runtime_params.RuntimeParams,
    geo: geometry.Geometry,
    only_boundary_condition: bool = False,
    original_n_e_value: cell_variable.CellVariable | None = None,
) -> cell_variable.CellVariable:
  """Gets initial and/or prescribed electron density profiles."""
  # Greenwald density in m^-3.
  # I_p in MA. a_minor in m.
  nGW = (
      profile_conditions_params.I_p
      / 1e6  # Convert to MA.
      / (jnp.pi * geo.a_minor**2)
      * 1e20
  )

  # Calculate n_e_right_bc.
  n_e_right_bc = jnp.where(
      profile_conditions_params.n_e_right_bc_is_fGW,
      profile_conditions_params.n_e_right_bc * nGW,
      profile_conditions_params.n_e_right_bc,
  )

  n_e_value = jnp.where(
      profile_conditions_params.n_e_nbar_is_fGW,
      profile_conditions_params.n_e * nGW,
      profile_conditions_params.n_e,
  )

  if profile_conditions_params.normalize_n_e_to_nbar:
    face_left = n_e_value[0]  # Zero gradient boundary condition at left face.
    face_right = n_e_right_bc
    face_inner = math_utils.inner_face_values_from_cell_values(
        cell_values=n_e_value,
        face_centers=geo.rho_face_norm,
        cell_centers=geo.rho_norm,
    )
    n_e_face = jnp.concatenate(
        [face_left[None], face_inner, face_right[None]],
    )
    # Find normalization factor such that desired line-averaged n_e is set.
    # Line-averaged electron density (nbar) is poorly defined. In general, the
    # definition is machine-dependent and even shot-dependent since it depends
    # on the usage of a specific interferometry chord. Furthermore, even if we
    # knew the specific chord used, its calculation would depend on magnetic
    # geometry information beyond what is available in StandardGeometry.
    # In lieu of a better solution, we use line-averaged electron density
    # defined on the outer midplane.
    a_minor_out = geo.R_out_face[-1] - geo.R_out_face[0]
    # find target nbar in absolute units
    target_nbar = jnp.where(
        profile_conditions_params.n_e_nbar_is_fGW,
        profile_conditions_params.nbar * nGW,
        profile_conditions_params.nbar,
    )
    if not profile_conditions_params.n_e_right_bc_is_absolute:
      # In this case, n_e_right_bc is taken from n_e and we also normalize it.
      C = target_nbar / (_trapz(n_e_face, geo.R_out_face) / a_minor_out)
      n_e_right_bc = C * n_e_right_bc
    else:
      # If n_e_right_bc is absolute, subtract off contribution from outer
      # face to get C we need to multiply the inner values with.
      nbar_from_n_e_face_inner = (
          _trapz(n_e_face[:-1], geo.R_out_face[:-1]) / a_minor_out
      )

      dr_edge = geo.R_out_face[-1] - geo.R_out_face[-2]

      C = (target_nbar - 0.5 * n_e_face[-1] * dr_edge / a_minor_out) / (
          nbar_from_n_e_face_inner + 0.5 * n_e_face[-2] * dr_edge / a_minor_out
      )
  else:
    C = 1

  n_e_value = C * n_e_value

  if only_boundary_condition:
    if original_n_e_value is None:
      raise ValueError(
          'original_n_e_value must be provided when only updating the boundary'
          ' condition.'
      )
    value = original_n_e_value.value
  else:
    value = n_e_value

  n_e = cell_variable.CellVariable(
      value=value,
      face_centers=geo.rho_face_norm,
      right_face_grad_constraint=None,
      right_face_constraint=n_e_right_bc,
  )
  return n_e


def get_updated_psi(
    profile_conditions_params: core_profile_runtime_params.RuntimeParams,
    geo: geometry.Geometry,
    dt: array_typing.FloatScalar,
    theta: array_typing.FloatScalar,
    only_boundary_condition: bool,
    original_psi: cell_variable.CellVariable,
    current_diffusion_time_reduction_factor: array_typing.FloatScalar = 1.0,
) -> cell_variable.CellVariable:
  """Gets psi boundary conditions and maybe the prescribed profile."""
  if only_boundary_condition or profile_conditions_params.psi is None:
    value = original_psi.value
  else:
    value = profile_conditions_params.psi

  right_face_grad_constraint = (
      psi_calculations.calculate_psi_grad_constraint_from_I_p(  # pylint: disable=g-long-ternary
          I_p=profile_conditions_params.I_p,
          geo=geo,
      )
      if not profile_conditions_params.use_v_loop_lcfs_boundary_condition
      else None
  )
  right_face_constraint = (
      psi_calculations.calculate_psi_value_constraint_from_v_loop(  # pylint: disable=g-long-ternary
          dt=dt,
          v_loop_lcfs_t=profile_conditions_params.v_loop_lcfs,
          v_loop_lcfs_t_plus_dt=profile_conditions_params.v_loop_lcfs,
          psi_lcfs_t=original_psi.right_face_constraint,  # pyrefly: ignore[bad-argument-type]
          theta=theta,
          current_diffusion_time_reduction_factor=(
              current_diffusion_time_reduction_factor
          ),
      )
      if profile_conditions_params.use_v_loop_lcfs_boundary_condition
      else None
  )
  return cell_variable.CellVariable(
      value=value,
      face_centers=geo.rho_face_norm,
      right_face_grad_constraint=right_face_grad_constraint,
      right_face_constraint=right_face_constraint,
  )


def get_updated_toroidal_angular_velocity(
    profile_conditions_params: core_profile_runtime_params.RuntimeParams,
    geo: geometry.Geometry,
) -> cell_variable.CellVariable:
  """Gets initial and/or prescribed toroidal velocity profiles."""
  if profile_conditions_params.toroidal_angular_velocity is None:
    value = jnp.zeros_like(geo.rho)
  else:
    value = profile_conditions_params.toroidal_angular_velocity
  toroidal_angular_velocity = cell_variable.CellVariable(
      value=value,
      face_centers=geo.rho_face_norm,
      right_face_grad_constraint=None,
      right_face_constraint=profile_conditions_params.toroidal_angular_velocity_right_bc,
  )
  return toroidal_angular_velocity


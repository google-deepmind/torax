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
from torax._src.geometry import standard_geometry
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
      profile_conditions_params.n_e_right_bc_is_f_GW,
      profile_conditions_params.n_e_right_bc * nGW,
      profile_conditions_params.n_e_right_bc,
  )

  n_e_value = jnp.where(
      profile_conditions_params.n_e_bar_is_f_GW,
      profile_conditions_params.n_e * nGW,
      profile_conditions_params.n_e,
  )

  if profile_conditions_params.normalize_n_e:
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
    # Line-averaged electron density (n_e_bar) is poorly defined. In general,
    # the definition is machine-dependent and even shot-dependent since it
    # depends on the usage of a specific interferometry chord. Furthermore, even
    # if we knew the specific chord used, its calculation would depend on
    # magnetic geometry information beyond what is available in
    # StandardGeometry.
    # In lieu of a better solution, we use line-averaged electron density
    # defined on the outer midplane.
    a_minor_out = geo.R_out_face[-1] - geo.R_out_face[0]
    # find target n_e_bar in absolute units
    target_n_e_bar = jnp.where(
        profile_conditions_params.n_e_bar_is_f_GW,
        profile_conditions_params.n_e_bar * nGW,
        profile_conditions_params.n_e_bar,
    )
    if not profile_conditions_params.n_e_right_bc_is_absolute:
      # In this case, n_e_right_bc is taken from n_e and we also normalize it.
      C = target_n_e_bar / (_trapz(n_e_face, geo.R_out_face) / a_minor_out)
      n_e_right_bc = C * n_e_right_bc
    else:
      # If n_e_right_bc is absolute, subtract off contribution from outer
      # face to get C we need to multiply the inner values with.
      n_e_bar_from_n_e_face_inner = (
          _trapz(n_e_face[:-1], geo.R_out_face[:-1]) / a_minor_out
      )

      dr_edge = geo.R_out_face[-1] - geo.R_out_face[-2]

      C = (target_n_e_bar - 0.5 * n_e_face[-1] * dr_edge / a_minor_out) / (
          n_e_bar_from_n_e_face_inner
          + 0.5 * n_e_face[-2] * dr_edge / a_minor_out
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


def build_psi_cell_variable(
    psi_value: array_typing.FloatVectorCell,
    I_p: array_typing.FloatScalar,
    geo: geometry.Geometry,
    use_v_loop_lcfs_boundary_condition: bool,
    psi_lcfs: array_typing.FloatScalar | None = None,
) -> cell_variable.CellVariable:
  """Builds a psi CellVariable with boundary conditions consistent with I_p.

  Args:
    psi_value: Poloidal flux values on the cell grid [Wb].
    I_p: Total plasma current [A].
    geo: Tokamak geometry.
    use_v_loop_lcfs_boundary_condition: If True, sets a Dirichlet right-face
      value constraint (`psi_lcfs` if provided, otherwise extrapolated from
      `psi_value[-1]` using the `I_p`-derived edge gradient) to initialize the
      LCFS flux for subsequent loop-voltage evolution. If False, sets a
      Neumann right-face gradient constraint from `I_p`.
    psi_lcfs: Optional LCFS poloidal flux [Wb] from a numerical equilibrium.

  Returns:
    Poloidal flux `CellVariable`.
  """
  if use_v_loop_lcfs_boundary_condition:
    right_face_grad_constraint = None
    if psi_lcfs is not None:
      right_face_constraint = psi_lcfs
    else:
      dpsi_drhonorm_edge = (
          psi_calculations.calculate_psi_grad_constraint_from_I_p(I_p, geo)
      )
      right_face_constraint = (
          psi_value[-1] + dpsi_drhonorm_edge * geo.drho_norm[-1] / 2
      )
  else:
    right_face_grad_constraint = (
        psi_calculations.calculate_psi_grad_constraint_from_I_p(I_p, geo)
    )
    right_face_constraint = None

  return cell_variable.CellVariable(
      value=psi_value,
      face_centers=geo.rho_face_norm,
      right_face_grad_constraint=right_face_grad_constraint,
      right_face_constraint=right_face_constraint,
  )


def get_prescribed_psi(
    profile_conditions_params: core_profile_runtime_params.RuntimeParams,
    geo: geometry.Geometry,
) -> cell_variable.CellVariable:
  """Gets prescribed poloidal flux (psi) profile and boundary conditions.

  Supports two prescribed modes selected by
  `profile_conditions_params.initial_psi_source`:
  - `PROFILE_CONDITIONS`: Uses the prescribed `profile_conditions_params.psi`
    profile.
  - `GEOMETRY`: Uses `geo.psi_from_I_p` and `geo.psi_from_I_p_face[-1]` from a
    `StandardGeometry` equilibrium.

  Args:
    profile_conditions_params: Runtime parameters for profile conditions.
    geo: Tokamak geometry.

  Returns:
    Poloidal flux `CellVariable` with boundary conditions consistent with
    `I_p`.

  Raises:
    ValueError: If `initial_psi_source` is not `PROFILE_CONDITIONS` or
      `GEOMETRY`, if `initial_psi_source` is `PROFILE_CONDITIONS` and `psi` is
      `None`, or if `initial_psi_source` is `GEOMETRY` and `geo` is not a
      `StandardGeometry`.
  """
  use_v_loop = (
      profile_conditions_params.psi_bc_source
      == core_profile_runtime_params.PsiBoundaryConditionSource.V_LOOP
  )
  match profile_conditions_params.initial_psi_source:
    case core_profile_runtime_params.InitialPsiSource.PROFILE_CONDITIONS:
      if profile_conditions_params.psi is None:
        raise ValueError(
            'psi is None, but initial_psi_source is PROFILE_CONDITIONS.'
        )
      return build_psi_cell_variable(
          psi_value=profile_conditions_params.psi,
          I_p=profile_conditions_params.I_p,
          geo=geo,
          use_v_loop_lcfs_boundary_condition=use_v_loop,
      )

    case core_profile_runtime_params.InitialPsiSource.GEOMETRY:
      if not isinstance(geo, standard_geometry.StandardGeometry):
        raise ValueError(
            'GEOMETRY initial_psi_source is only supported for standard'
            ' geometry.'
        )
      # geo.psi_from_I_p_face[-1] is already made consistent with I_p by
      # make_I_p_consistent.
      return build_psi_cell_variable(
          psi_value=geo.psi_from_I_p,
          I_p=profile_conditions_params.I_p,
          geo=geo,
          use_v_loop_lcfs_boundary_condition=use_v_loop,
          psi_lcfs=geo.psi_from_I_p_face[-1],
      )

    case _:
      raise ValueError(
          'Unsupported prescribed initial_psi_source:'
          f' {profile_conditions_params.initial_psi_source}'
      )


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

  match profile_conditions_params.psi_bc_source:
    case core_profile_runtime_params.PsiBoundaryConditionSource.I_P:
      right_face_grad_constraint = (
          psi_calculations.calculate_psi_grad_constraint_from_I_p(
              I_p=profile_conditions_params.I_p,
              geo=geo,
          )
      )
      right_face_constraint = None
    case core_profile_runtime_params.PsiBoundaryConditionSource.V_LOOP:
      assert original_psi.right_face_constraint is not None
      right_face_grad_constraint = None
      right_face_constraint = (
          psi_calculations.calculate_psi_value_constraint_from_v_loop(
              dt=dt,
              v_loop_lcfs_t=profile_conditions_params.v_loop_lcfs,
              v_loop_lcfs_t_plus_dt=profile_conditions_params.v_loop_lcfs,
              psi_lcfs_t=original_psi.right_face_constraint,
              theta=theta,
              current_diffusion_time_reduction_factor=(
                  current_diffusion_time_reduction_factor
              ),
          )
      )
    case _:
      raise ValueError(
          'Unknown psi_bc_source:'
          f' {profile_conditions_params.psi_bc_source}'
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


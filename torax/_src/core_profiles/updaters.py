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

"""Update routines for core_profiles.

Set of routines that relate to updating/creating core profiles from existing
core profiles.

Includes:
- get_prescribed_core_profile_values: Updates core profiles which are not being
  evolved by PDE.
- update_core_profiles_during_step: Intra-step updates of the evolving variables
  in core profiles during solver iterations.
- update_all_core_profiles_after_step: Updates all core_profiles after a step.
  Includes the evolved variables and derived variables like q_face, psidot, etc.
- compute_boundary_conditions_for_t_plus_dt: Computes boundary conditions for
  the next timestep and returns updates to State.
"""
import dataclasses
import functools
import jax
from jax import numpy as jnp
from torax import array_typing
from torax import jax_utils
from torax import state
from torax._src.config import runtime_params_slice
from torax._src.core_profiles import getters
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.physics import charge_states
from torax._src.physics import formulas
from torax._src.physics import psi_calculations
from torax._src.sources import source_profiles as source_profiles_lib

_trapz = jax.scipy.integrate.trapezoid

# pylint: disable=invalid-name


def _calculate_psi_value_constraint_from_vloop(
    dt: array_typing.ScalarFloat,
    theta: array_typing.ScalarFloat,
    vloop_lcfs_t: array_typing.ScalarFloat,
    vloop_lcfs_t_plus_dt: array_typing.ScalarFloat,
    psi_lcfs_t: array_typing.ScalarFloat,
) -> jax.Array:
  """Calculates the value constraint on the poloidal flux for the next time step from loop voltage."""
  theta_weighted_vloop_lcfs = (
      1 - theta
  ) * vloop_lcfs_t + theta * vloop_lcfs_t_plus_dt
  return psi_lcfs_t + theta_weighted_vloop_lcfs * dt


@functools.partial(
    jax_utils.jit,
    static_argnames=['static_runtime_params_slice'],
)
def get_prescribed_core_profile_values(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
) -> dict[str, array_typing.ArrayFloat]:
  """Updates core profiles which are not being evolved by PDE.

  Uses same functions as for profile initialization.

  Args:
    static_runtime_params_slice: Static simulation runtime parameters.
    dynamic_runtime_params_slice: Dynamic runtime parameters at t=t_initial.
    geo: Torus geometry.
    core_profiles: Core profiles dataclass to be updated

  Returns:
    Updated core profiles values on the cell grid.
  """
  # If profiles are not evolved, they can still potential be time-evolving,
  # depending on the runtime params. If so, they are updated below.
  if not static_runtime_params_slice.evolve_ion_heat:
    T_i = getters.get_updated_ion_temperature(
        dynamic_runtime_params_slice.profile_conditions, geo
    ).value
  else:
    T_i = core_profiles.T_i.value
  if not static_runtime_params_slice.evolve_electron_heat:
    T_e_cell_variable = getters.get_updated_electron_temperature(
        dynamic_runtime_params_slice.profile_conditions, geo
    )
    T_e = T_e_cell_variable.value
  else:
    T_e_cell_variable = core_profiles.T_e
    T_e = T_e_cell_variable.value
  if not static_runtime_params_slice.evolve_density:
    n_e_cell_variable = getters.get_updated_electron_density(
        static_runtime_params_slice,
        dynamic_runtime_params_slice.numerics,
        dynamic_runtime_params_slice.profile_conditions,
        geo,
    )
  else:
    n_e_cell_variable = core_profiles.n_e
  n_i, n_impurity, Z_i, Z_i_face, Z_impurity, Z_impurity_face = (
      getters.get_ion_density_and_charge_states(
          static_runtime_params_slice,
          dynamic_runtime_params_slice,
          geo,
          n_e_cell_variable,
          T_e_cell_variable,
      )
  )
  n_e = n_e_cell_variable.value
  n_i = n_i.value
  n_impurity = n_impurity.value

  return {
      'T_i': T_i,
      'T_e': T_e,
      'n_e': n_e,
      'n_i': n_i,
      'n_impurity': n_impurity,
      'Z_i': Z_i,
      'Z_i_face': Z_i_face,
      'Z_impurity': Z_impurity,
      'Z_impurity_face': Z_impurity_face,
  }


def _get_update(
    x_new: tuple[cell_variable.CellVariable, ...],
    evolving_names: tuple[str, ...],
    core_profiles: state.CoreProfiles,
    var: str,
):
  """If variable `var` is evolving, return its new value stored in x_new."""
  if var in evolving_names:
    return x_new[evolving_names.index(var)]
  # `var` is not evolving, so its new value is just its old value
  return getattr(core_profiles, var)


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'static_runtime_params_slice',
        'evolving_names',
    ],
)
def update_core_profiles_during_step(
    x_new: tuple[cell_variable.CellVariable, ...],
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    evolving_names: tuple[str, ...],
) -> state.CoreProfiles:
  """Returns the new core profiles after updating the evolving variables.

  Intended for use during iterative solves in the step function. Only updates
  the core profiles which are being evolved by the PDE and directly derivable
  quantities like q_face, s_face. core_profile calculations which require
  sources are not updated.

  Args:
    x_new: The new values of the evolving variables.
    static_runtime_params_slice: The static runtime params slice.
    dynamic_runtime_params_slice: The dynamic runtime params slice.
    geo: Magnetic geometry.
    core_profiles: The old set of core plasma profiles.
    evolving_names: The names of the evolving variables.
  """

  T_i = _get_update(x_new, evolving_names, core_profiles, 'T_i')
  T_e = _get_update(x_new, evolving_names, core_profiles, 'T_e')
  psi = _get_update(x_new, evolving_names, core_profiles, 'psi')
  n_e = _get_update(x_new, evolving_names, core_profiles, 'n_e')

  n_i, n_impurity, Z_i, Z_i_face, Z_impurity, Z_impurity_face = (
      getters.get_ion_density_and_charge_states(
          static_runtime_params_slice,
          dynamic_runtime_params_slice,
          geo,
          n_e,
          T_e,
      )
  )

  return dataclasses.replace(
      core_profiles,
      T_i=T_i,
      T_e=T_e,
      psi=psi,
      n_e=n_e,
      n_i=n_i,
      n_impurity=n_impurity,
      Z_i=Z_i,
      Z_i_face=Z_i_face,
      Z_impurity=Z_impurity,
      Z_impurity_face=Z_impurity_face,
      q_face=psi_calculations.calc_q_face(geo, psi),
      s_face=psi_calculations.calc_s_face(geo, psi),
  )


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'static_runtime_params_slice',
        'evolving_names',
    ],
)
def update_all_core_profiles_after_step(
    x_new: tuple[cell_variable.CellVariable, ...],
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice_t_plus_dt: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    source_profiles: source_profiles_lib.SourceProfiles,
    core_profiles_t: state.CoreProfiles,
    core_profiles_t_plus_dt: state.CoreProfiles,
    evolving_names: tuple[str, ...],
    dt: array_typing.ScalarFloat,
) -> state.CoreProfiles:
  """Returns a new core profiles after the stepper has finished.

  Updates the evolved variables and derived variables like q_face, psidot, etc.

  Args:
    x_new: The new values of the evolving variables.
    static_runtime_params_slice: The static runtime params slice.
    dynamic_runtime_params_slice_t_plus_dt: The dynamic runtime params slice.
    geo: Magnetic geometry.
    source_profiles: The source profiles from the step function output.
    core_profiles_t: The old set of core plasma profiles.
    core_profiles_t_plus_dt: The partially new set of core plasma profiles. On
      input into this function, all prescribed profiles and used boundary
      conditions are already set. But evolving values are not.
    evolving_names: The names of the evolving variables.
    dt: The size of the last timestep.
  """

  T_i = _get_update(
      x_new, evolving_names, core_profiles_t_plus_dt, 'T_i'
  )
  T_e = _get_update(
      x_new, evolving_names, core_profiles_t_plus_dt, 'T_e'
  )
  psi = _get_update(x_new, evolving_names, core_profiles_t_plus_dt, 'psi')
  n_e = _get_update(x_new, evolving_names, core_profiles_t_plus_dt, 'n_e')

  n_i, n_impurity, Z_i, Z_i_face, Z_impurity, Z_impurity_face = (
      getters.get_ion_density_and_charge_states(
          static_runtime_params_slice,
          dynamic_runtime_params_slice_t_plus_dt,
          geo,
          n_e,
          T_e,
      )
  )

  psi_sources = source_profiles.total_psi_sources(geo)

  vloop_lcfs = (
      dynamic_runtime_params_slice_t_plus_dt.profile_conditions.vloop_lcfs  # pylint: disable=g-long-ternary
      if static_runtime_params_slice.profile_conditions.use_vloop_lcfs_boundary_condition
      else _update_vloop_lcfs_from_psi(
          core_profiles_t.psi,
          psi,
          dt,
      )
  )

  psidot = dataclasses.replace(
      core_profiles_t_plus_dt.psidot,
      value=psi_calculations.calculate_psidot_from_psi_sources(
          psi_sources=psi_sources,
          sigma=core_profiles_t_plus_dt.sigma,
          sigma_face=core_profiles_t_plus_dt.sigma_face,
          resistivity_multiplier=dynamic_runtime_params_slice_t_plus_dt.numerics.resistivity_multiplier,
          psi=psi,
          geo=geo,
      ),
      right_face_constraint=vloop_lcfs,
      right_face_grad_constraint=None,
  )

  j_total, j_total_face, Ip_profile_face = psi_calculations.calc_j_total(
      geo, psi
  )

  return state.CoreProfiles(
      T_i=T_i,
      T_e=T_e,
      psi=psi,
      n_e=n_e,
      n_i=n_i,
      n_impurity=n_impurity,
      Z_i=Z_i,
      Z_i_face=Z_i_face,
      Z_impurity=Z_impurity,
      Z_impurity_face=Z_impurity_face,
      psidot=psidot,
      q_face=psi_calculations.calc_q_face(geo, psi),
      s_face=psi_calculations.calc_s_face(geo, psi),
      density_reference=core_profiles_t_plus_dt.density_reference,
      A_i=core_profiles_t_plus_dt.A_i,
      A_impurity=core_profiles_t_plus_dt.A_impurity,
      vloop_lcfs=vloop_lcfs,
      # These have already been updated in the solver.
      sigma=core_profiles_t_plus_dt.sigma,
      sigma_face=core_profiles_t_plus_dt.sigma_face,
      j_total=j_total,
      j_total_face=j_total_face,
      Ip_profile_face=Ip_profile_face,
  )


def compute_boundary_conditions_for_t_plus_dt(
    dt: array_typing.ScalarFloat,
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice_t: runtime_params_slice.DynamicRuntimeParamsSlice,
    dynamic_runtime_params_slice_t_plus_dt: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo_t_plus_dt: geometry.Geometry,
    core_profiles_t: state.CoreProfiles,
) -> dict[str, dict[str, jax.Array | None]]:
  """Computes boundary conditions for the next timestep and returns updates to State.

  Args:
    dt: Size of the next timestep
    static_runtime_params_slice: Static (concrete) runtime parameters
    dynamic_runtime_params_slice_t: Dynamic runtime parameters for the current
      timestep. Will not be used if
      dynamic_runtime_params_slice_t_plus_dt.profile_conditions.vloop_lcfs is
      None, i.e. if the dirichlet psi boundary condition based on Ip is
      used
    dynamic_runtime_params_slice_t_plus_dt: Dynamic runtime parameters for the
      next timestep
    geo_t_plus_dt: Geometry object for the next timestep
    core_profiles_t: Core profiles at the current timestep. Will not be used if
      dynamic_runtime_params_slice_t_plus_dt.profile_conditions.vloop_lcfs is
      None, i.e. if the dirichlet psi boundary condition based on Ip is
      used

  Returns:
    Mapping from State attribute names to dictionaries updating attributes of
    each CellVariable in the state. This dict can in theory recursively replace
    values in a State object.
  """
  profile_conditions_t_plus_dt = (
      dynamic_runtime_params_slice_t_plus_dt.profile_conditions
  )
  # TODO(b/390143606): Separate out the boundary condition calculation from the
  # core profile calculation.
  n_e = getters.get_updated_electron_density(
      static_runtime_params_slice,
      dynamic_runtime_params_slice_t_plus_dt.numerics,
      profile_conditions_t_plus_dt,
      geo_t_plus_dt,
  )
  n_e_right_bc = n_e.right_face_constraint

  Z_i_edge = charge_states.get_average_charge_state(
      static_runtime_params_slice.main_ion_names,
      ion_mixture=dynamic_runtime_params_slice_t_plus_dt.plasma_composition.main_ion,
      T_e=profile_conditions_t_plus_dt.T_e_right_bc,
  )
  Z_impurity_edge = charge_states.get_average_charge_state(
      static_runtime_params_slice.impurity_names,
      ion_mixture=dynamic_runtime_params_slice_t_plus_dt.plasma_composition.impurity,
      T_e=profile_conditions_t_plus_dt.T_e_right_bc,
  )

  dilution_factor_edge = formulas.calculate_main_ion_dilution_factor(
      Z_i_edge,
      Z_impurity_edge,
      dynamic_runtime_params_slice_t_plus_dt.plasma_composition.Z_eff_face[-1],
  )

  n_i_bound_right = n_e_right_bc * dilution_factor_edge
  n_impurity_bound_right = (
      n_e_right_bc - n_i_bound_right * Z_i_edge
  ) / Z_impurity_edge

  return {
      'T_i': dict(
          left_face_grad_constraint=jnp.zeros(()),
          right_face_grad_constraint=None,
          right_face_constraint=profile_conditions_t_plus_dt.T_i_right_bc,
      ),
      'T_e': dict(
          left_face_grad_constraint=jnp.zeros(()),
          right_face_grad_constraint=None,
          right_face_constraint=profile_conditions_t_plus_dt.T_e_right_bc,
      ),
      'n_e': dict(
          left_face_grad_constraint=jnp.zeros(()),
          right_face_grad_constraint=None,
          right_face_constraint=jnp.array(n_e_right_bc),
      ),
      'n_i': dict(
          left_face_grad_constraint=jnp.zeros(()),
          right_face_grad_constraint=None,
          right_face_constraint=jnp.array(n_i_bound_right),
      ),
      'n_impurity': dict(
          left_face_grad_constraint=jnp.zeros(()),
          right_face_grad_constraint=None,
          right_face_constraint=jnp.array(n_impurity_bound_right),
      ),
      'psi': dict(
          right_face_grad_constraint=(
              psi_calculations.calculate_psi_grad_constraint_from_Ip(  # pylint: disable=g-long-ternary
                  Ip=profile_conditions_t_plus_dt.Ip,
                  geo=geo_t_plus_dt,
              )
              if not static_runtime_params_slice.profile_conditions.use_vloop_lcfs_boundary_condition
              else None
          ),
          right_face_constraint=(
              _calculate_psi_value_constraint_from_vloop(  # pylint: disable=g-long-ternary
                  dt=dt,
                  vloop_lcfs_t=dynamic_runtime_params_slice_t.profile_conditions.vloop_lcfs,
                  vloop_lcfs_t_plus_dt=profile_conditions_t_plus_dt.vloop_lcfs,
                  psi_lcfs_t=core_profiles_t.psi.right_face_constraint,
                  theta=static_runtime_params_slice.solver.theta_implicit,
              )
              if static_runtime_params_slice.profile_conditions.use_vloop_lcfs_boundary_condition
              else None
          ),
      ),
      'Z_i_edge': Z_i_edge,
      'Z_impurity_edge': Z_impurity_edge,
  }


def provide_core_profiles_t_plus_dt(
    dt: jax.Array,
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice_t: runtime_params_slice.DynamicRuntimeParamsSlice,
    dynamic_runtime_params_slice_t_plus_dt: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo_t_plus_dt: geometry.Geometry,
    core_profiles_t: state.CoreProfiles,
) -> state.CoreProfiles:
  """Provides state at t_plus_dt with new boundary conditions and prescribed profiles."""
  updated_boundary_conditions = compute_boundary_conditions_for_t_plus_dt(
      dt=dt,
      static_runtime_params_slice=static_runtime_params_slice,
      dynamic_runtime_params_slice_t=dynamic_runtime_params_slice_t,
      dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
      geo_t_plus_dt=geo_t_plus_dt,
      core_profiles_t=core_profiles_t,
  )
  updated_values = get_prescribed_core_profile_values(
      static_runtime_params_slice=static_runtime_params_slice,
      dynamic_runtime_params_slice=dynamic_runtime_params_slice_t_plus_dt,
      geo=geo_t_plus_dt,
      core_profiles=core_profiles_t,
  )
  T_i = dataclasses.replace(
      core_profiles_t.T_i,
      value=updated_values['T_i'],
      **updated_boundary_conditions['T_i'],
  )
  T_e = dataclasses.replace(
      core_profiles_t.T_e,
      value=updated_values['T_e'],
      **updated_boundary_conditions['T_e'],
  )
  psi = dataclasses.replace(
      core_profiles_t.psi, **updated_boundary_conditions['psi']
  )
  n_e = dataclasses.replace(
      core_profiles_t.n_e,
      value=updated_values['n_e'],
      **updated_boundary_conditions['n_e'],
  )
  n_i = dataclasses.replace(
      core_profiles_t.n_i,
      value=updated_values['n_i'],
      **updated_boundary_conditions['n_i'],
  )
  n_impurity = dataclasses.replace(
      core_profiles_t.n_impurity,
      value=updated_values['n_impurity'],
      **updated_boundary_conditions['n_impurity'],
  )

  # pylint: disable=invalid-name
  # Update Z_face with boundary condition Z, needed for cases where T_e
  # is evolving and updated_prescribed_core_profiles is a no-op.
  Z_i_face = jnp.concatenate(
      [
          updated_values['Z_i_face'][:-1],
          jnp.array([updated_boundary_conditions['Z_i_edge']]),
      ],
  )
  Z_impurity_face = jnp.concatenate(
      [
          updated_values['Z_impurity_face'][:-1],
          jnp.array([updated_boundary_conditions['Z_impurity_edge']]),
      ],
  )
  # pylint: enable=invalid-name
  core_profiles_t_plus_dt = dataclasses.replace(
      core_profiles_t,
      T_i=T_i,
      T_e=T_e,
      psi=psi,
      n_e=n_e,
      n_i=n_i,
      n_impurity=n_impurity,
      Z_i=updated_values['Z_i'],
      Z_i_face=Z_i_face,
      Z_impurity=updated_values['Z_impurity'],
      Z_impurity_face=Z_impurity_face,
  )
  return core_profiles_t_plus_dt


# TODO(b/406173731): Find robust solution for underdetermination and solve this
# for general theta_implicit values.
def _update_vloop_lcfs_from_psi(
    psi_t: cell_variable.CellVariable,
    psi_t_plus_dt: cell_variable.CellVariable,
    dt: array_typing.ScalarFloat,
) -> jax.Array:
  """Updates the vloop_lcfs for the next timestep.

  For the Ip boundary condition case, the vloop_lcfs formula is in principle
  calculated from:

  (psi_lcfs_t_plus_dt - psi_lcfs_t) / dt =
    vloop_lcfs_t_plus_dt*theta_implicit - vloop_lcfs_t*(1-theta_implicit)

  However this set of equations is underdetermined. We thus restrict this
  calculation assuming a fully implicit system, i.e. theta_implicit=1, which is
  the TORAX default. Be cautious when interpreting the results of this function
  when theta_implicit != 1 (non-standard usage).

  Args:
    psi_t: The psi CellVariable at the beginning of the timestep interval.
    psi_t_plus_dt: The updated psi CellVariable for the next timestep.
    dt: The size of the last timestep.

  Returns:
    The updated vloop_lcfs for the next timestep.
  """
  psi_lcfs_t = psi_t.face_value()[-1]
  psi_lcfs_t_plus_dt = psi_t_plus_dt.face_value()[-1]
  vloop_lcfs_t_plus_dt = (psi_lcfs_t_plus_dt - psi_lcfs_t) / dt
  return vloop_lcfs_t_plus_dt

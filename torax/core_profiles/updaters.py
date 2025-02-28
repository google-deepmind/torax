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

Set of routines that updates time-dependent boundary conditions, and updates
time-dependent prescribed core_profiles that are not evolved by the PDE system.
"""
import dataclasses
import functools
import jax
from jax import numpy as jnp
from torax import array_typing
from torax import jax_utils
from torax import state
from torax.config import runtime_params_slice
from torax.core_profiles import formulas
from torax.fvm import cell_variable
from torax.geometry import geometry
from torax.physics import charge_states

_trapz = jax.scipy.integrate.trapezoid


# pylint: disable=invalid-name
def _get_charge_states(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    temp_el: cell_variable.CellVariable,
) -> tuple[
    array_typing.ArrayFloat,
    array_typing.ArrayFloat,
    array_typing.ArrayFloat,
    array_typing.ArrayFloat,
]:
  """Updated charge states based on IonMixtures and electron temperature."""
  Zi = charge_states.get_average_charge_state(
      ion_symbols=static_runtime_params_slice.main_ion_names,
      ion_mixture=dynamic_runtime_params_slice.plasma_composition.main_ion,
      Te=temp_el.value,
  )
  Zi_face = charge_states.get_average_charge_state(
      ion_symbols=static_runtime_params_slice.main_ion_names,
      ion_mixture=dynamic_runtime_params_slice.plasma_composition.main_ion,
      Te=temp_el.face_value(),
  )

  Zimp = charge_states.get_average_charge_state(
      ion_symbols=static_runtime_params_slice.impurity_names,
      ion_mixture=dynamic_runtime_params_slice.plasma_composition.impurity,
      Te=temp_el.value,
  )
  Zimp_face = charge_states.get_average_charge_state(
      ion_symbols=static_runtime_params_slice.impurity_names,
      ion_mixture=dynamic_runtime_params_slice.plasma_composition.impurity,
      Te=temp_el.face_value(),
  )

  return Zi, Zi_face, Zimp, Zimp_face


# jitted since also used outside the stepper
@functools.partial(
    jax_utils.jit, static_argnames=['static_runtime_params_slice']
)
def get_ion_density_and_charge_states(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    ne: cell_variable.CellVariable,
    temp_el: cell_variable.CellVariable,
) -> tuple[
    cell_variable.CellVariable,
    cell_variable.CellVariable,
    array_typing.ArrayFloat,
    array_typing.ArrayFloat,
    array_typing.ArrayFloat,
    array_typing.ArrayFloat,
]:
  """Updated ion densities based on state.

  Main ion and impurities are each treated as a single effective ion, but could
  be comparised of multiple species within an IonMixture. The main ion and
  impurity densities are calculated depending on the Zeff constraint,
  quasineutrality, and the average impurity charge state which may be
  temperature dependent.

  Zeff = (Zi**2 * ni + Zimp**2 * nimp)/ne  ;  nimp*Zimp + ni*Zi = ne

  Args:
    static_runtime_params_slice: Static runtime parameters.
    dynamic_runtime_params_slice: Dynamic runtime parameters.
    geo: Geometry of the tokamak.
    ne: Electron density profile [nref].
    temp_el: Electron temperature profile [keV].

  Returns:
    ni: Ion density profile [nref].
    nimp: Impurity density profile [nref].
    Zi: Average charge state of main ion on cell grid [amu].
      Typically just the average of the atomic numbers since these are normally
      low Z ions and can be assumed to be fully ionized.
    Zi_face: Average charge state of main ion on face grid [amu].
    Zimp: Average charge state of impurities on cell grid [amu].
    Zimp_face: Average charge state of impurities on face grid [amu].
  """

  Zi, Zi_face, Zimp, Zimp_face = _get_charge_states(
      static_runtime_params_slice,
      dynamic_runtime_params_slice,
      temp_el,
  )

  Zeff = dynamic_runtime_params_slice.plasma_composition.Zeff
  Zeff_face = dynamic_runtime_params_slice.plasma_composition.Zeff_face

  dilution_factor = formulas.get_main_ion_dilution_factor(Zi, Zimp, Zeff)
  dilution_factor_edge = formulas.get_main_ion_dilution_factor(
      Zi_face[-1], Zimp_face[-1], Zeff_face[-1]
  )

  ni = cell_variable.CellVariable(
      value=ne.value * dilution_factor,
      dr=geo.drho_norm,
      right_face_grad_constraint=None,
      right_face_constraint=jnp.array(
          ne.right_face_constraint * dilution_factor_edge
      ),
  )

  nimp = cell_variable.CellVariable(
      value=(ne.value - ni.value * Zi) / Zimp,
      dr=geo.drho_norm,
      right_face_grad_constraint=None,
      right_face_constraint=jnp.array(
          ne.right_face_constraint - ni.right_face_constraint * Zi[-1]
      )
      / Zimp_face[-1],
  )
  return ni, nimp, Zi, Zi_face, Zimp, Zimp_face


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
  if not static_runtime_params_slice.ion_heat_eq:
    temp_ion = formulas.get_updated_ion_temperature(
        dynamic_runtime_params_slice.profile_conditions, geo
    ).value
  else:
    temp_ion = core_profiles.temp_ion.value
  if not static_runtime_params_slice.el_heat_eq:
    temp_el_cell_variable = formulas.get_updated_electron_temperature(
        dynamic_runtime_params_slice.profile_conditions, geo
    )
    temp_el = temp_el_cell_variable.value
  else:
    temp_el_cell_variable = core_profiles.temp_el
    temp_el = temp_el_cell_variable.value
  if not static_runtime_params_slice.dens_eq:
    ne_cell_variable = formulas.get_ne(
        dynamic_runtime_params_slice.numerics,
        dynamic_runtime_params_slice.profile_conditions,
        geo,
    )
  else:
    ne_cell_variable = core_profiles.ne
  ni, nimp, Zi, Zi_face, Zimp, Zimp_face = get_ion_density_and_charge_states(
      static_runtime_params_slice,
      dynamic_runtime_params_slice,
      geo,
      ne_cell_variable,
      temp_el_cell_variable,
  )
  ne = ne_cell_variable.value
  ni = ni.value
  nimp = nimp.value

  return {
      'temp_ion': temp_ion,
      'temp_el': temp_el,
      'ne': ne,
      'ni': ni,
      'nimp': nimp,
      'Zi': Zi,
      'Zi_face': Zi_face,
      'Zimp': Zimp,
      'Zimp_face': Zimp_face,
  }


def update_evolving_core_profiles(
    x_new: tuple[cell_variable.CellVariable, ...],
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    evolving_names: tuple[str, ...],
) -> state.CoreProfiles:
  """Returns the new core profiles after updating the evolving variables.

  Args:
    x_new: The new values of the evolving variables.
    static_runtime_params_slice: The static runtime params slice.
    dynamic_runtime_params_slice: The dynamic runtime params slice.
    geo: Magnetic geometry.
    core_profiles: The old set of core plasma profiles.
    evolving_names: The names of the evolving variables.
  """

  def get_update(x_new, var):
    """Returns the new value of `var`."""
    if var in evolving_names:
      return x_new[evolving_names.index(var)]
    # `var` is not evolving, so its new value is just its old value
    return getattr(core_profiles, var)

  temp_ion = get_update(x_new, 'temp_ion')
  temp_el = get_update(x_new, 'temp_el')
  psi = get_update(x_new, 'psi')
  ne = get_update(x_new, 'ne')

  ni, nimp, Zi, Zi_face, Zimp, Zimp_face = get_ion_density_and_charge_states(
      static_runtime_params_slice,
      dynamic_runtime_params_slice,
      geo,
      ne,
      temp_el,
  )

  return dataclasses.replace(
      core_profiles,
      temp_ion=temp_ion,
      temp_el=temp_el,
      psi=psi,
      ne=ne,
      ni=ni,
      nimp=nimp,
      Zi=Zi,
      Zi_face=Zi_face,
      Zimp=Zimp,
      Zimp_face=Zimp_face,
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
      None, i.e. if the dirichlet psi boundary condition based on Ip_tot is used
    dynamic_runtime_params_slice_t_plus_dt: Dynamic runtime parameters for the
      next timestep
    geo_t_plus_dt: Geometry object for the next timestep
    core_profiles_t: Core profiles at the current timestep. Will not be used if
      dynamic_runtime_params_slice_t_plus_dt.profile_conditions.vloop_lcfs is
      None, i.e. if the dirichlet psi boundary condition based on Ip_tot is used

  Returns:
    Mapping from State attribute names to dictionaries updating attributes of
    each CellVariable in the state. This dict can in theory recursively replace
    values in a State object.
  """
  Ti_bound_right = jax_utils.error_if_not_positive(
      dynamic_runtime_params_slice_t_plus_dt.profile_conditions.Ti_bound_right,
      'Ti_bound_right',
  )

  Te_bound_right = jax_utils.error_if_not_positive(
      dynamic_runtime_params_slice_t_plus_dt.profile_conditions.Te_bound_right,
      'Te_bound_right',
  )
  # TODO(b/390143606): Separate out the boundary condition calculation from the
  # core profile calculation.
  ne = formulas.get_ne(
      dynamic_runtime_params_slice_t_plus_dt.numerics,
      dynamic_runtime_params_slice_t_plus_dt.profile_conditions,
      geo_t_plus_dt,
  )
  ne_bound_right = ne.right_face_constraint

  Zi_edge = charge_states.get_average_charge_state(
      static_runtime_params_slice.main_ion_names,
      ion_mixture=dynamic_runtime_params_slice_t_plus_dt.plasma_composition.main_ion,
      Te=Te_bound_right,
  )
  Zimp_edge = charge_states.get_average_charge_state(
      static_runtime_params_slice.impurity_names,
      ion_mixture=dynamic_runtime_params_slice_t_plus_dt.plasma_composition.impurity,
      Te=Te_bound_right,
  )

  dilution_factor_edge = formulas.get_main_ion_dilution_factor(
      Zi_edge,
      Zimp_edge,
      dynamic_runtime_params_slice_t_plus_dt.plasma_composition.Zeff_face[-1],
  )

  ni_bound_right = ne_bound_right * dilution_factor_edge
  nimp_bound_right = (ne_bound_right - ni_bound_right * Zi_edge) / Zimp_edge

  return {
      'temp_ion': dict(
          left_face_grad_constraint=jnp.zeros(()),
          right_face_grad_constraint=None,
          right_face_constraint=jnp.array(Ti_bound_right),
      ),
      'temp_el': dict(
          left_face_grad_constraint=jnp.zeros(()),
          right_face_grad_constraint=None,
          right_face_constraint=jnp.array(Te_bound_right),
      ),
      'ne': dict(
          left_face_grad_constraint=jnp.zeros(()),
          right_face_grad_constraint=None,
          right_face_constraint=jnp.array(ne_bound_right),
      ),
      'ni': dict(
          left_face_grad_constraint=jnp.zeros(()),
          right_face_grad_constraint=None,
          right_face_constraint=jnp.array(ni_bound_right),
      ),
      'nimp': dict(
          left_face_grad_constraint=jnp.zeros(()),
          right_face_grad_constraint=None,
          right_face_constraint=jnp.array(nimp_bound_right),
      ),
      'psi': dict(
          right_face_grad_constraint=(
              formulas.calculate_psi_grad_constraint_from_Ip_tot(  # pylint: disable=g-long-ternary
                  Ip_tot=dynamic_runtime_params_slice_t_plus_dt.profile_conditions.Ip_tot,
                  geo=geo_t_plus_dt,
              )
              if not dynamic_runtime_params_slice_t_plus_dt.profile_conditions.use_vloop_lcfs_boundary_condition
              else None
          ),
          right_face_constraint=(
              _calculate_psi_value_constraint_from_vloop(  # pylint: disable=g-long-ternary
                  dt=dt,
                  vloop_lcfs_t=dynamic_runtime_params_slice_t.profile_conditions.vloop_lcfs,
                  vloop_lcfs_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt.profile_conditions.vloop_lcfs,
                  psi_lcfs_t=core_profiles_t.psi.face_value()[-1],
                  theta=static_runtime_params_slice.stepper.theta_imp,
              )
              if dynamic_runtime_params_slice_t_plus_dt.profile_conditions.use_vloop_lcfs_boundary_condition
              else None
          ),
      ),
      'Zi_edge': Zi_edge,
      'Zimp_edge': Zimp_edge,
  }

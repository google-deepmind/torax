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

"""Initialization and update routines for core_profiles.

Set of routines that initializes core_profiles, updates time-dependent boundary
conditions, and updates time-dependent prescribed core_profiles that are not
evolved by the PDE system.
"""
import dataclasses
import functools
import jax
from jax import numpy as jnp
from torax import array_typing
from torax import charge_states
from torax import constants
from torax import jax_utils
from torax import math_utils
from torax import physics
from torax import state
from torax.config import numerics
from torax.config import profile_conditions
from torax.config import runtime_params_slice
from torax.fvm import cell_variable
from torax.geometry import geometry
from torax.sources import ohmic_heat_source
from torax.sources import source_models as source_models_lib
from torax.sources import source_profiles as source_profiles_lib

_trapz = jax.scipy.integrate.trapezoid

# Using capitalized variables for physics notational conventions rather than
# Python style.
# pylint: disable=invalid-name


def _updated_ion_temperature(
    dynamic_profile_conditions: profile_conditions.DynamicProfileConditions,
    geo: geometry.Geometry,
) -> cell_variable.CellVariable:
  """Updated ion temp. Used upon initialization and if temp_ion=False."""
  Ti_bound_right = jax_utils.error_if_not_positive(
      dynamic_profile_conditions.Ti_bound_right,
      'Ti_bound_right',
  )
  temp_ion = cell_variable.CellVariable(
      value=dynamic_profile_conditions.Ti,
      left_face_grad_constraint=jnp.zeros(()),
      right_face_grad_constraint=None,
      right_face_constraint=Ti_bound_right,
      dr=geo.drho_norm,
  )

  return temp_ion


def _updated_electron_temperature(
    dynamic_profile_conditions: profile_conditions.DynamicProfileConditions,
    geo: geometry.Geometry,
) -> cell_variable.CellVariable:
  """Updated electron temp. Used upon initialization and if temp_el=False."""
  Te_bound_right = jax_utils.error_if_not_positive(
      dynamic_profile_conditions.Te_bound_right,
      'Te_bound_right',
  )
  temp_el = cell_variable.CellVariable(
      value=dynamic_profile_conditions.Te,
      left_face_grad_constraint=jnp.zeros(()),
      right_face_grad_constraint=None,
      right_face_constraint=Te_bound_right,
      dr=geo.drho_norm,
  )
  return temp_el


def _get_ne(
    dynamic_numerics: numerics.DynamicNumerics,
    dynamic_profile_conditions: profile_conditions.DynamicProfileConditions,
    geo: geometry.Geometry,
) -> cell_variable.CellVariable:
  """Gets initial or prescribed electron density profile at current timestep."""

  nGW = (
      dynamic_profile_conditions.Ip_tot
      / (jnp.pi * geo.Rmin**2)
      * 1e20
      / dynamic_numerics.nref
  )
  ne_value = jnp.where(
      dynamic_profile_conditions.ne_is_fGW,
      dynamic_profile_conditions.ne * nGW,
      dynamic_profile_conditions.ne,
  )
  # Calculate ne_bound_right.
  ne_bound_right = jnp.where(
      dynamic_profile_conditions.ne_bound_right_is_fGW,
      dynamic_profile_conditions.ne_bound_right * nGW,
      dynamic_profile_conditions.ne_bound_right,
  )

  if dynamic_profile_conditions.normalize_to_nbar:
    face_left = ne_value[0]  # Zero gradient boundary condition at left face.
    face_right = ne_bound_right
    face_inner = (ne_value[..., :-1] + ne_value[..., 1:]) / 2.0
    ne_face = jnp.concatenate(
        [face_left[None], face_inner, face_right[None]],
    )
    # Find normalization factor such that desired line-averaged ne is set.
    # Line-averaged electron density (nbar) is poorly defined. In general, the
    # definition is machine-dependent and even shot-dependent since it depends
    # on the usage of a specific interferometry chord. Furthermore, even if we
    # knew the specific chord used, its calculation would depend on magnetic
    # geometry information beyond what is available in StandardGeometry.
    # In lieu of a better solution, we use line-averaged electron density
    # defined on the outer midplane.
    Rmin_out = geo.Rout_face[-1] - geo.Rout_face[0]
    # find target nbar in absolute units
    target_nbar = jnp.where(
        dynamic_profile_conditions.ne_is_fGW,
        dynamic_profile_conditions.nbar * nGW,
        dynamic_profile_conditions.nbar,
    )
    if not dynamic_profile_conditions.ne_bound_right_is_absolute:
      # In this case, ne_bound_right is taken from ne and we also normalize it.
      C = target_nbar / (_trapz(ne_face, geo.Rout_face) / Rmin_out)
      ne_bound_right = C * ne_bound_right
    else:
      # If ne_bound_right is absolute, subtract off contribution from outer
      # face to get C we need to multiply the inner values with.
      nbar_from_ne_face_inner = (
          _trapz(ne_face[:-1], geo.Rout_face[:-1]) / Rmin_out
      )

      dr_edge = geo.Rout_face[-1] - geo.Rout_face[-2]

      C = (target_nbar - 0.5 * ne_face[-1] * dr_edge / Rmin_out) / (
          nbar_from_ne_face_inner + 0.5 * ne_face[-2] * dr_edge / Rmin_out
      )
  else:
    C = 1

  ne_value = C * ne_value

  ne = cell_variable.CellVariable(
      value=ne_value,
      dr=geo.drho_norm,
      right_face_grad_constraint=None,
      right_face_constraint=jnp.array(ne_bound_right),
  )
  return ne


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

  dilution_factor = physics.get_main_ion_dilution_factor(Zi, Zimp, Zeff)
  dilution_factor_edge = physics.get_main_ion_dilution_factor(
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


def _prescribe_currents_no_bootstrap(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
) -> state.Currents:
  """Creates the initial Currents without the bootstrap current.

  Args:
    static_runtime_params_slice: Static runtime parameters.
    dynamic_runtime_params_slice: General runtime parameters at t_initial.
    geo: Geometry of the tokamak.
    core_profiles: Core profiles.
    source_models: All TORAX source/sink functions.

  Returns:
    currents: Initial Currents
  """
  # Many variables throughout this function are capitalized based on physics
  # notational conventions rather than on Google Python style

  # Calculate splitting of currents depending on input runtime params.
  Ip = dynamic_runtime_params_slice.profile_conditions.Ip_tot

  # Set zero bootstrap current
  bootstrap_profile = source_profiles_lib.BootstrapCurrentProfile.zero_profile(
      geo
  )

  # calculate "External" current profile (e.g. ECCD)
  external_current = source_models.external_current_source(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      static_runtime_params_slice=static_runtime_params_slice,
      geo=geo,
      core_profiles=core_profiles,
  )
  Iext = (
      math_utils.cell_integration(external_current * geo.spr_cell, geo) / 10**6
  )
  # Total Ohmic current.
  Iohm = Ip - Iext

  # construct prescribed current formula on grid.
  jformula = (
      1 - geo.rho_norm**2
  ) ** dynamic_runtime_params_slice.profile_conditions.nu
  # calculate total and Ohmic current profiles
  denom = _trapz(jformula * geo.spr_cell, geo.rho_norm)
  if dynamic_runtime_params_slice.profile_conditions.initial_j_is_total_current:
    Ctot = Ip * 1e6 / denom
    jtot = jformula * Ctot
    johm = jtot - external_current
  else:
    Cohm = Iohm * 1e6 / denom
    johm = jformula * Cohm
    jtot = johm + external_current

  jtot_hires = _get_jtot_hires(
      dynamic_runtime_params_slice,
      geo,
      bootstrap_profile,
      Iohm,
      external_current=external_current,
  )

  currents = state.Currents(
      jtot=jtot,
      jtot_face=math_utils.cell_to_face(
          jtot,
          geo,
          preserved_quantity=math_utils.IntegralPreservationQuantity.SURFACE,
      ),
      jtot_hires=jtot_hires,
      johm=johm,
      external_current_source=external_current,
      j_bootstrap=bootstrap_profile.j_bootstrap,
      j_bootstrap_face=bootstrap_profile.j_bootstrap_face,
      I_bootstrap=bootstrap_profile.I_bootstrap,
      Ip_profile_face=jnp.zeros(geo.rho_face.shape),  # psi not yet calculated
      sigma=bootstrap_profile.sigma,
  )

  return currents


def _prescribe_currents_with_bootstrap(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
) -> state.Currents:
  """Creates the initial Currents.

  Args:
    static_runtime_params_slice: Static runtime parameters.
    dynamic_runtime_params_slice: General runtime parameters at t_initial.
    geo: Geometry of the tokamak.
    core_profiles: Core profiles.
    source_models: All TORAX source/sink functions. If not provided, uses the
      default sources.

  Returns:
    currents: Plasma currents
  """

  # Many variables throughout this function are capitalized based on physics
  # notational conventions rather than on Google Python style
  Ip = dynamic_runtime_params_slice.profile_conditions.Ip_tot

  bootstrap_profile = source_models.j_bootstrap.get_value(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      static_runtime_params_slice=static_runtime_params_slice,
      geo=geo,
      core_profiles=core_profiles,
  )
  f_bootstrap = bootstrap_profile.I_bootstrap / (Ip * 1e6)

  # calculate "External" current profile (e.g. ECCD)
  external_current = source_models.external_current_source(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      static_runtime_params_slice=static_runtime_params_slice,
      geo=geo,
      core_profiles=core_profiles,
  )
  Iext = (
      math_utils.cell_integration(external_current * geo.spr_cell, geo) / 10**6
  )
  Iohm = Ip - Iext - f_bootstrap * Ip

  # construct prescribed current formula on grid.
  jformula = (
      1 - geo.rho_norm**2
  ) ** dynamic_runtime_params_slice.profile_conditions.nu
  denom = _trapz(jformula * geo.spr_cell, geo.rho_norm)
  # calculate total and Ohmic current profiles
  if dynamic_runtime_params_slice.profile_conditions.initial_j_is_total_current:
    Ctot = Ip * 1e6 / denom
    jtot = jformula * Ctot
    johm = jtot - external_current - bootstrap_profile.j_bootstrap
  else:
    Cohm = Iohm * 1e6 / denom
    johm = jformula * Cohm
    jtot = johm + external_current + bootstrap_profile.j_bootstrap

  jtot_hires = _get_jtot_hires(
      dynamic_runtime_params_slice,
      geo,
      bootstrap_profile,
      Iohm,
      external_current,
  )
  currents = state.Currents(
      jtot=jtot,
      jtot_face=math_utils.cell_to_face(
          jtot, geo, math_utils.IntegralPreservationQuantity.SURFACE
      ),
      jtot_hires=jtot_hires,
      johm=johm,
      external_current_source=external_current,
      j_bootstrap=bootstrap_profile.j_bootstrap,
      j_bootstrap_face=bootstrap_profile.j_bootstrap_face,
      I_bootstrap=bootstrap_profile.I_bootstrap,
      Ip_profile_face=jnp.zeros(geo.rho_face.shape),  # psi not yet calculated
      sigma=bootstrap_profile.sigma,
  )

  return currents


def _calculate_currents_from_psi(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
) -> state.Currents:
  """Creates the initial Currents using psi to calculate jtot.

  Args:
    static_runtime_params_slice: Static runtime parameters.
    dynamic_runtime_params_slice: General runtime parameters at t_initial.
    geo: Geometry of the tokamak.
    core_profiles: Core profiles.
    source_models: All TORAX source/sink functions. If not provided, uses the
      default sources.

  Returns:
    currents: Plasma currents
  """

  # Many variables throughout this function are capitalized based on physics
  # notational conventions rather than on Google Python style
  jtot, jtot_face, Ip_profile_face = physics.calc_jtot_from_psi(
      geo,
      core_profiles.psi,
  )

  bootstrap_profile = source_models.j_bootstrap.get_value(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      static_runtime_params_slice=static_runtime_params_slice,
      geo=geo,
      core_profiles=core_profiles,
  )

  # calculate "External" current profile (e.g. ECCD)
  # form of external current on face grid
  external_current = source_models.external_current_source(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      static_runtime_params_slice=static_runtime_params_slice,
      geo=geo,
      core_profiles=core_profiles,
  )
  johm = jtot - external_current - bootstrap_profile.j_bootstrap
  currents = state.Currents(
      jtot=jtot,
      jtot_face=jtot_face,
      jtot_hires=None,
      johm=johm,
      external_current_source=external_current,
      j_bootstrap=bootstrap_profile.j_bootstrap,
      j_bootstrap_face=bootstrap_profile.j_bootstrap_face,
      I_bootstrap=bootstrap_profile.I_bootstrap,
      Ip_profile_face=Ip_profile_face,
      sigma=bootstrap_profile.sigma,
  )

  return currents


def _update_psi_from_j(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    jtot_hires: jax.Array,
) -> cell_variable.CellVariable:
  """Calculates poloidal flux (psi) consistent with plasma current.

  For increased accuracy of psi, a hi-res grid is used, due to the double
    integration.

  Args:
    dynamic_runtime_params_slice: Dynamic runtime parameters.
    geo: Torus geometry.
    jtot_hires: High resolution version of jtot.

  Returns:
    psi: Poloidal flux cell variable.
  """
  y = jtot_hires * geo.spr_hires
  assert y.ndim == 1
  assert geo.rho_hires.ndim == 1
  Ip_profile = math_utils.cumulative_trapezoid(
      y=y, x=geo.rho_hires_norm, initial=0.0
  )
  scale = jnp.concatenate((
      jnp.zeros((1,)),
      (16 * jnp.pi**3 * constants.CONSTANTS.mu0 * geo.Phib)
      / (geo.F_hires[1:] * geo.g2g3_over_rhon_hires[1:]),
  ))
  # dpsi_dr on hires cell grid
  dpsi_drhon_hires = scale * Ip_profile

  # psi on hires cell grid
  psi_hires = math_utils.cumulative_trapezoid(
      y=dpsi_drhon_hires, x=geo.rho_hires_norm, initial=0.0
  )

  psi_value = jnp.interp(geo.rho_norm, geo.rho_hires_norm, psi_hires)

  # Set the BCs for psi to ensure the correct Ip_tot
  dpsi_drhonorm_edge = _calculate_psi_grad_constraint_from_Ip_tot(
      dynamic_runtime_params_slice,
      geo,
  )

  if (
      dynamic_runtime_params_slice.profile_conditions.use_vloop_lcfs_boundary_condition
  ):
    # For vloop_lcfs, we will prescribe a rate of change of psi at the LCFS
    # For the first timestep, we need an initial value for psi at the LCFS, so
    # we set it to match the desired plasma current.
    right_face_grad_constraint = None
    right_face_constraint = (
        psi_value[-1] + dpsi_drhonorm_edge * geo.drho_norm / 2
    )
  else:
    # Use the dpsi/drho calculated above as the right face gradient constraint
    right_face_grad_constraint = dpsi_drhonorm_edge
    right_face_constraint = None

  psi = cell_variable.CellVariable(
      value=psi_value,
      dr=geo.drho_norm,
      right_face_grad_constraint=right_face_grad_constraint,
      right_face_constraint=right_face_constraint,
  )

  return psi


def _calculate_psi_grad_constraint_from_Ip_tot(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
) -> jax.Array:
  """Calculates the gradient constraint on the poloidal flux (psi) from Ip."""
  return (
      dynamic_runtime_params_slice.profile_conditions.Ip_tot
      * 1e6
      * (16 * jnp.pi**3 * constants.CONSTANTS.mu0 * geo.Phib)
      / (geo.g2g3_over_rhon_face[-1] * geo.F_face[-1])
  )


def _calculate_psi_value_constraint_from_vloop(
    dt: array_typing.ScalarFloat,
    theta: array_typing.ScalarFloat,
    vloop_lcfs_t: array_typing.ScalarFloat,
    vloop_lcfs_t_plus_dt: array_typing.ScalarFloat,
    psi_lcfs_t: array_typing.ScalarFloat,
) -> jax.Array:
  """Calculates the value constraint on the poloidal flux for the next time step from loop voltage."""
  theta_weighted_vloop_lcfs = (
      theta * vloop_lcfs_t + (1 - theta) * vloop_lcfs_t_plus_dt
  )
  return psi_lcfs_t + theta_weighted_vloop_lcfs * dt


def _init_psi_and_current(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
) -> state.CoreProfiles:
  """Initialises psi and currents in core profiles.

  There are three modes of doing this that are supported:
    1. Retrieving psi from the profile conditions.
    2. Retrieving psi from the standard geometry input.
    3. Calculating j according to the nu formula and then calculating psi from
    that. As we are calculating j using a guess for psi, this method is iterated
    to converge to the true psi.

  Args:
    static_runtime_params_slice: Static runtime parameters.
    dynamic_runtime_params_slice: Dynamic runtime parameters.
    geo: Torus geometry.
    core_profiles: Core profiles.
    source_models: All TORAX source/sink functions. If not provided, uses the
      default sources.

  Returns:
    Refined core profiles.
  """
  use_vloop_as_boundary_condition = (
      dynamic_runtime_params_slice.profile_conditions.use_vloop_lcfs_boundary_condition
  )

  # Case 1: retrieving psi from the profile conditions, using the prescribed profile and Ip_tot
  if dynamic_runtime_params_slice.profile_conditions.psi is not None:
    # Calculate the dpsi/drho necessary to achieve the given Ip_tot
    dpsi_drhonorm_edge = _calculate_psi_grad_constraint_from_Ip_tot(
        dynamic_runtime_params_slice,
        geo,
    )

    # Set the BCs to ensure the correct Ip_tot
    if use_vloop_as_boundary_condition:
      # Extrapolate the value of psi at the LCFS from the dpsi/drho constraint
      # to achieve the desired Ip_tot
      right_face_grad_constraint = None
      right_face_constraint = (
          dynamic_runtime_params_slice.profile_conditions.psi[-1]
          + dpsi_drhonorm_edge * geo.drho_norm / 2
      )
    else:
      # Use the dpsi/drho calculated above as the right face gradient constraint
      right_face_grad_constraint = dpsi_drhonorm_edge
      right_face_constraint = None

    psi = cell_variable.CellVariable(
        value=dynamic_runtime_params_slice.profile_conditions.psi,
        right_face_grad_constraint=right_face_grad_constraint,
        right_face_constraint=right_face_constraint,
        dr=geo.drho_norm,
    )

    core_profiles = dataclasses.replace(core_profiles, psi=psi)
    currents = _calculate_currents_from_psi(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_runtime_params_slice,
        geo=geo,
        core_profiles=core_profiles,
        source_models=source_models,
    )

  # Case 2: retrieving psi from the standard geometry input.
  elif (
      isinstance(geo, geometry.StandardGeometry)
      and not dynamic_runtime_params_slice.profile_conditions.initial_psi_from_j
  ):
    # psi is already provided from a numerical equilibrium, so no need to
    # first calculate currents. However, non-inductive currents are still
    # calculated and used in current diffusion equation.

    # Calculate the dpsi/drho necessary to achieve the given Ip_tot
    dpsi_drhonorm_edge = _calculate_psi_grad_constraint_from_Ip_tot(
        dynamic_runtime_params_slice,
        geo,
    )

    # Set the BCs, with the option of scaling to the given Ip_tot
    if use_vloop_as_boundary_condition and geo.Ip_from_parameters:
      # Extrapolate the value of psi at the LCFS from the dpsi/drho constraint
      right_face_grad_constraint = None
      right_face_constraint = (
          geo.psi_from_Ip[-1] + dpsi_drhonorm_edge * geo.drho_norm / 2
      )
    elif use_vloop_as_boundary_condition:
      # Use the psi from the equilibrium as the right face constraint
      right_face_grad_constraint = None
      right_face_constraint = geo.psi_from_Ip[-1]
    else:
      # Use the dpsi/drho calculated above as the right face gradient constraint
      right_face_grad_constraint = dpsi_drhonorm_edge
      right_face_constraint = None

    psi = cell_variable.CellVariable(
        value=geo.psi_from_Ip,  # Use psi from equilibrium
        right_face_grad_constraint=right_face_grad_constraint,
        right_face_constraint=right_face_constraint,
        dr=geo.drho_norm,
    )
    core_profiles = dataclasses.replace(core_profiles, psi=psi)

    # Calculate non-inductive currents
    currents = _calculate_currents_from_psi(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_runtime_params_slice,
        geo=geo,
        core_profiles=core_profiles,
        source_models=source_models,
    )

  # Case 3: calculating j according to nu formula and psi from j.
  elif (
      isinstance(geo, geometry.CircularAnalyticalGeometry)
      or dynamic_runtime_params_slice.profile_conditions.initial_psi_from_j
  ):
    currents = _prescribe_currents_no_bootstrap(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_runtime_params_slice,
        geo=geo,
        core_profiles=core_profiles,
        source_models=source_models,
    )
    psi = _update_psi_from_j(
        dynamic_runtime_params_slice,
        geo,
        currents.jtot_hires,
    )
    core_profiles = dataclasses.replace(
        core_profiles, currents=currents, psi=psi
    )
    currents = _prescribe_currents_with_bootstrap(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_runtime_params_slice,
        geo=geo,
        core_profiles=core_profiles,
        source_models=source_models,
    )
    psi = _update_psi_from_j(
        dynamic_runtime_params_slice,
        geo,
        currents.jtot_hires,
    )
    _, _, Ip_profile_face = physics.calc_jtot_from_psi(
        geo,
        psi,
    )
    currents = dataclasses.replace(currents, Ip_profile_face=Ip_profile_face)
  else:
    raise ValueError('Cannot compute psi for given config.')

  core_profiles = dataclasses.replace(
      core_profiles,
      psi=psi,
      currents=currents,
      vloop_lcfs=dynamic_runtime_params_slice.profile_conditions.vloop_lcfs,
  )

  return core_profiles


def initial_core_profiles(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    source_models: source_models_lib.SourceModels,
) -> state.CoreProfiles:
  """Calculates the initial core profiles.

  Args:
    static_runtime_params_slice: Static runtime parameters.
    dynamic_runtime_params_slice: Dynamic runtime parameters at t=t_initial.
    geo: Torus geometry.
    source_models: All models for TORAX sources/sinks.

  Returns:
    Initial core profiles.
  """

  # To set initial values and compute the boundary conditions, we need to handle
  # potentially time-varying inputs from the users.
  # The default time in build_dynamic_runtime_params_slice is t_initial
  temp_ion = _updated_ion_temperature(
      dynamic_runtime_params_slice.profile_conditions, geo
  )
  temp_el = _updated_electron_temperature(
      dynamic_runtime_params_slice.profile_conditions, geo
  )
  ne = _get_ne(
      dynamic_runtime_params_slice.numerics,
      dynamic_runtime_params_slice.profile_conditions,
      geo,
  )

  ni, nimp, Zi, Zi_face, Zimp, Zimp_face = get_ion_density_and_charge_states(
      static_runtime_params_slice,
      dynamic_runtime_params_slice,
      geo,
      ne,
      temp_el,
  )

  # The later calculation needs core profiles.
  # So initialize these quantities with zeros.
  psidot = cell_variable.CellVariable(
      value=jnp.zeros_like(geo.rho),
      dr=geo.drho_norm,
  )
  psi = cell_variable.CellVariable(
      value=jnp.zeros_like(geo.rho), dr=geo.drho_norm
  )
  q_face = jnp.zeros_like(geo.rho_face)
  s_face = jnp.zeros_like(geo.rho_face)
  currents = state.Currents.zeros(geo)

  # Set vloop_lcfs to 0 for the first time step if not provided
  vloop_lcfs = (
      jnp.array(0.0)
      if dynamic_runtime_params_slice.profile_conditions.vloop_lcfs is None
      else jnp.array(dynamic_runtime_params_slice.profile_conditions.vloop_lcfs)
  )

  core_profiles = state.CoreProfiles(
      temp_ion=temp_ion,
      temp_el=temp_el,
      ne=ne,
      ni=ni,
      Zi=Zi,
      Zi_face=Zi_face,
      Ai=dynamic_runtime_params_slice.plasma_composition.main_ion.avg_A,
      nimp=nimp,
      Zimp=Zimp,
      Zimp_face=Zimp_face,
      Aimp=dynamic_runtime_params_slice.plasma_composition.impurity.avg_A,
      psi=psi,
      psidot=psidot,
      currents=currents,
      q_face=q_face,
      s_face=s_face,
      nref=jnp.asarray(dynamic_runtime_params_slice.numerics.nref),
      vloop_lcfs=vloop_lcfs,
  )

  core_profiles = _init_psi_and_current(
      static_runtime_params_slice,
      dynamic_runtime_params_slice,
      geo,
      core_profiles,
      source_models,
  )

  # psidot calculated here with phibdot=0 in geo, since this is initial
  # conditions and we don't yet have information on geo_t_plus_dt for the
  # phibdot calculation.
  psidot = dataclasses.replace(
      core_profiles.psidot,
      value=ohmic_heat_source.calc_psidot(
          static_runtime_params_slice,
          dynamic_runtime_params_slice,
          geo,
          core_profiles,
          source_models,
      ),
  )
  core_profiles = dataclasses.replace(
      core_profiles,
      psidot=psidot,
      vloop_lcfs=psidot.face_value()[-1],
  )

  # Set psi as source of truth and recalculate jtot, q, s
  return physics.update_jtot_q_face_s_face(
      geo=geo,
      core_profiles=core_profiles,
      q_correction_factor=dynamic_runtime_params_slice.numerics.q_correction_factor,
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
  if (
      not static_runtime_params_slice.ion_heat_eq
      and dynamic_runtime_params_slice.numerics.enable_prescribed_profile_evolution
  ):
    temp_ion = _updated_ion_temperature(
        dynamic_runtime_params_slice.profile_conditions, geo
    ).value
  else:
    temp_ion = core_profiles.temp_ion.value
  if (
      not static_runtime_params_slice.el_heat_eq
      and dynamic_runtime_params_slice.numerics.enable_prescribed_profile_evolution
  ):
    temp_el_cell_variable = _updated_electron_temperature(
        dynamic_runtime_params_slice.profile_conditions, geo
    )
    temp_el = temp_el_cell_variable.value
  else:
    temp_el_cell_variable = core_profiles.temp_el
    temp_el = temp_el_cell_variable.value
  if (
      not static_runtime_params_slice.dens_eq
      and dynamic_runtime_params_slice.numerics.enable_prescribed_profile_evolution
  ):
    ne_cell_variable = _get_ne(
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
    dt: jax.Array,
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
    dynamic_runtime_params_slice_t: Dynamic runtime parameters for the current timestep
    dynamic_runtime_params_slice_t_plus_dt: Dynamic runtime parameters for the next timestep
    geo_t_plus_dt: Geometry object for the next timestep
    core_profiles_t: Core profiles at the current timestep

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
  ne = _get_ne(
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

  dilution_factor_edge = physics.get_main_ion_dilution_factor(
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
              _calculate_psi_grad_constraint_from_Ip_tot(
                  dynamic_runtime_params_slice=dynamic_runtime_params_slice_t_plus_dt,
                  geo=geo_t_plus_dt,
              )
              if not dynamic_runtime_params_slice_t_plus_dt.profile_conditions.use_vloop_lcfs_boundary_condition
              else None
          ),
          right_face_constraint=(
              _calculate_psi_value_constraint_from_vloop(
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


def _get_jtot_hires(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    bootstrap_profile: source_profiles_lib.BootstrapCurrentProfile,
    Iohm: jax.Array | float,
    external_current: jax.Array,
) -> jax.Array:
  """Calculates jtot hires."""
  j_bootstrap_hires = jnp.interp(
      geo.rho_hires, geo.rho_face, bootstrap_profile.j_bootstrap_face
  )

  # calculate hi-res "External" current profile (e.g. ECCD) on cell grid.
  external_current_face = math_utils.cell_to_face(
      external_current,
      geo,
      preserved_quantity=math_utils.IntegralPreservationQuantity.SURFACE,
  )
  external_current_hires = jnp.interp(
      geo.rho_hires, geo.rho_face, external_current_face
  )

  # calculate high resolution jtot and Ohmic current profile
  jformula_hires = (
      1 - geo.rho_hires_norm**2
  ) ** dynamic_runtime_params_slice.profile_conditions.nu
  denom = _trapz(jformula_hires * geo.spr_hires, geo.rho_hires_norm)
  if dynamic_runtime_params_slice.profile_conditions.initial_j_is_total_current:
    Ctot_hires = (
        dynamic_runtime_params_slice.profile_conditions.Ip_tot * 1e6 / denom
    )
    jtot_hires = jformula_hires * Ctot_hires
  else:
    Cohm_hires = Iohm * 1e6 / denom
    johm_hires = jformula_hires * Cohm_hires
    jtot_hires = johm_hires + external_current_hires + j_bootstrap_hires
  return jtot_hires

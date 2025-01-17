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
import jax
from jax import numpy as jnp
from torax import constants
from torax import jax_utils
from torax import math_utils
from torax import physics
from torax import state
from torax.config import runtime_params_slice
from torax.fvm import cell_variable
from torax.geometry import geometry
from torax.sources import ohmic_heat_source
from torax.sources import source_models as source_models_lib
from torax.sources import source_profiles as source_profiles_lib

_trapz = jax.scipy.integrate.trapezoid


def updated_ion_temperature(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
) -> cell_variable.CellVariable:
  """Updated ion temp. Used upon initialization and if temp_ion=False."""
  # pylint: disable=invalid-name
  Ti_bound_right = (
      dynamic_runtime_params_slice.profile_conditions.Ti_bound_right
  )

  Ti_bound_right = jax_utils.error_if_not_positive(
      Ti_bound_right,
      'Ti_bound_right',
  )
  temp_ion = cell_variable.CellVariable(
      value=dynamic_runtime_params_slice.profile_conditions.Ti,
      left_face_grad_constraint=jnp.zeros(()),
      right_face_grad_constraint=None,
      right_face_constraint=Ti_bound_right,
      dr=geo.drho_norm,
  )
  # pylint: enable=invalid-name
  return temp_ion


def updated_electron_temperature(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
) -> cell_variable.CellVariable:
  """Updated electron temp. Used upon initialization and if temp_el=False."""
  # pylint: disable=invalid-name
  Te_bound_right = (
      dynamic_runtime_params_slice.profile_conditions.Te_bound_right
  )

  Te_bound_right = jax_utils.error_if_not_positive(
      Te_bound_right,
      'Te_bound_right',
  )
  temp_el = cell_variable.CellVariable(
      value=dynamic_runtime_params_slice.profile_conditions.Te,
      left_face_grad_constraint=jnp.zeros(()),
      right_face_grad_constraint=None,
      right_face_constraint=Te_bound_right,
      dr=geo.drho_norm,
  )
  # pylint: enable=invalid-name
  return temp_el


# pylint: disable=invalid-name
def _get_ne(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
) -> cell_variable.CellVariable:
  """Helper to get the electron density profile at the current timestep."""
  # pylint: disable=invalid-name
  nGW = (
      dynamic_runtime_params_slice.profile_conditions.Ip_tot
      / (jnp.pi * geo.Rmin**2)
      * 1e20
      / dynamic_runtime_params_slice.numerics.nref
  )
  ne_value = jnp.where(
      dynamic_runtime_params_slice.profile_conditions.ne_is_fGW,
      dynamic_runtime_params_slice.profile_conditions.ne * nGW,
      dynamic_runtime_params_slice.profile_conditions.ne,
  )
  # Calculate ne_bound_right.
  ne_bound_right = jnp.where(
      dynamic_runtime_params_slice.profile_conditions.ne_bound_right_is_fGW,
      dynamic_runtime_params_slice.profile_conditions.ne_bound_right * nGW,
      dynamic_runtime_params_slice.profile_conditions.ne_bound_right,
  )

  if dynamic_runtime_params_slice.profile_conditions.normalize_to_nbar:
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
        dynamic_runtime_params_slice.profile_conditions.ne_is_fGW,
        dynamic_runtime_params_slice.profile_conditions.nbar * nGW,
        dynamic_runtime_params_slice.profile_conditions.nbar,
    )
    if (
        not dynamic_runtime_params_slice.profile_conditions.ne_bound_right_is_absolute
    ):
      # In this case, ne_bound_right is taken from ne and we also normalize it.
      C = target_nbar / (_trapz(ne_face, geo.Rout_face) / Rmin_out)
      # pylint: enable=invalid-name
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


def _updated_ion_density(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    ne: cell_variable.CellVariable,
) -> tuple[
    cell_variable.CellVariable,
    cell_variable.CellVariable,
]:
  """Updated ion densities based on electron density and plasma composition."""
  # define ion profile based on Zeff and single assumed impurity
  # with Zimp. main ion limited to hydrogenic species for now.
  # Assume isotopic balance for DT fusion power. Solve for ni based on:
  # Zeff = (ni + Zimp**2 * nimp)/ne  ;  nimp*Zimp + ni = ne ,
  # where all density units are in nref

  Zi = dynamic_runtime_params_slice.plasma_composition.main_ion.avg_Z
  Zimp = dynamic_runtime_params_slice.plasma_composition.impurity.avg_Z
  Zeff = dynamic_runtime_params_slice.plasma_composition.Zeff
  Zeff_face = dynamic_runtime_params_slice.plasma_composition.Zeff_face

  dilution_factor = physics.get_main_ion_dilution_factor(Zi, Zimp, Zeff)
  dilution_factor_edge = physics.get_main_ion_dilution_factor(
      Zi, Zimp, Zeff_face[-1]
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
          ne.right_face_constraint - ni.right_face_constraint * Zi
      )
      / Zimp,
  )
  return ni, nimp


def updated_density(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
) -> tuple[
    cell_variable.CellVariable,
    cell_variable.CellVariable,
    cell_variable.CellVariable,
]:
  """Updated particle density. Used upon initialization and if dens_eq=False."""
  ne = _get_ne(
      dynamic_runtime_params_slice,
      geo,
  )
  ni, nimp = _updated_ion_density(
      dynamic_runtime_params_slice,
      geo,
      ne,
  )

  return ne, ni, nimp


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
  # pylint: disable=invalid-name

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
  # pylint: disable=invalid-name
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
  # pylint: disable=invalid-name
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
  psi_grad_constraint = _calculate_psi_grad_constraint(
      dynamic_runtime_params_slice,
      geo,
  )

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
  # dpsi_dr on the cell grid
  dpsi_drhon_hires = scale * Ip_profile

  # psi on cell grid
  psi_hires = math_utils.cumulative_trapezoid(
      y=dpsi_drhon_hires, x=geo.rho_hires_norm, initial=0.0
  )

  psi_value = jnp.interp(geo.rho_norm, geo.rho_hires_norm, psi_hires)

  psi = cell_variable.CellVariable(
      value=psi_value,
      dr=geo.drho_norm,
      right_face_grad_constraint=psi_grad_constraint,
  )

  return psi


# pylint: enable=invalid-name
def _calculate_psi_grad_constraint(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
) -> jax.Array:
  """Calculates the constraint on the poloidal flux (psi)."""
  return (
      dynamic_runtime_params_slice.profile_conditions.Ip_tot
      * 1e6
      * (16 * jnp.pi**3 * constants.CONSTANTS.mu0 * geo.Phib)
      / (geo.g2g3_over_rhon_face[-1] * geo.F_face[-1])
  )


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
  # Retrieving psi from the profile conditions.
  if dynamic_runtime_params_slice.profile_conditions.psi is not None:
    psi = cell_variable.CellVariable(
        value=dynamic_runtime_params_slice.profile_conditions.psi,
        right_face_grad_constraint=_calculate_psi_grad_constraint(
            dynamic_runtime_params_slice,
            geo,
        ),
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
  # Retrieving psi from the standard geometry input.
  elif (
      isinstance(geo, geometry.StandardGeometry)
      and not dynamic_runtime_params_slice.profile_conditions.initial_psi_from_j
  ):
    # psi is already provided from a numerical equilibrium, so no need to
    # first calculate currents. However, non-inductive currents are still
    # calculated and used in current diffusion equation.
    psi = cell_variable.CellVariable(
        value=geo.psi_from_Ip,
        right_face_grad_constraint=_calculate_psi_grad_constraint(
            dynamic_runtime_params_slice,
            geo,
        ),
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
  # Calculating j according to nu formula and psi from j.
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
    # pylint: disable=invalid-name
    _, _, Ip_profile_face = physics.calc_jtot_from_psi(
        geo,
        psi,
    )
    # pylint: enable=invalid-name
    currents = dataclasses.replace(currents, Ip_profile_face=Ip_profile_face)
  else:
    raise ValueError('Cannot compute psi for given config.')

  core_profiles = dataclasses.replace(core_profiles, psi=psi, currents=currents)

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
  # pylint: disable=invalid-name

  # To set initial values and compute the boundary conditions, we need to handle
  # potentially time-varying inputs from the users.
  # The default time in build_dynamic_runtime_params_slice is t_initial
  temp_ion = updated_ion_temperature(dynamic_runtime_params_slice, geo)
  temp_el = updated_electron_temperature(dynamic_runtime_params_slice, geo)
  ne, ni, nimp = updated_density(dynamic_runtime_params_slice, geo)

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

  core_profiles = state.CoreProfiles(
      temp_ion=temp_ion,
      temp_el=temp_el,
      ne=ne,
      ni=ni,
      Zi=dynamic_runtime_params_slice.plasma_composition.main_ion.avg_Z,
      Ai=dynamic_runtime_params_slice.plasma_composition.main_ion.avg_A,
      nimp=nimp,
      Zimp=dynamic_runtime_params_slice.plasma_composition.impurity.avg_Z,
      Aimp=dynamic_runtime_params_slice.plasma_composition.impurity.avg_A,
      psi=psi,
      psidot=psidot,
      currents=currents,
      q_face=q_face,
      s_face=s_face,
      nref=jnp.asarray(dynamic_runtime_params_slice.numerics.nref),
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
  core_profiles = dataclasses.replace(core_profiles, psidot=psidot)

  # Set psi as source of truth and recalculate jtot, q, s
  core_profiles = physics.update_jtot_q_face_s_face(
      geo=geo,
      core_profiles=core_profiles,
      q_correction_factor=dynamic_runtime_params_slice.numerics.q_correction_factor,
  )

  # pylint: enable=invalid-name
  return core_profiles


def updated_prescribed_core_profiles(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
) -> dict[str, jax.Array]:
  """Updates core profiles which are not being evolved by PDE.

  Uses same functions as for profile initialization.

  Args:
    static_runtime_params_slice: Static simulation runtime parameters.
    dynamic_runtime_params_slice: Dynamic runtime parameters at t=t_initial.
    geo: Torus geometry.
    core_profiles: Core profiles dataclass to be updated

  Returns:
    Updated core profiles.
  """
  # pylint: disable=invalid-name

  # If profiles are not evolved, they can still potential be time-evolving,
  # depending on the runtime params. If so, they are updated below.
  if (
      not static_runtime_params_slice.ion_heat_eq
      and dynamic_runtime_params_slice.numerics.enable_prescribed_profile_evolution
  ):
    temp_ion = updated_ion_temperature(dynamic_runtime_params_slice, geo).value
  else:
    temp_ion = core_profiles.temp_ion.value
  if (
      not static_runtime_params_slice.el_heat_eq
      and dynamic_runtime_params_slice.numerics.enable_prescribed_profile_evolution
  ):
    temp_el = updated_electron_temperature(
        dynamic_runtime_params_slice, geo
    ).value
  else:
    temp_el = core_profiles.temp_el.value
  if (
      not static_runtime_params_slice.dens_eq
      and dynamic_runtime_params_slice.numerics.enable_prescribed_profile_evolution
  ):
    ne, ni, nimp = updated_density(dynamic_runtime_params_slice, geo)
    ne = ne.value
    ni = ni.value
    nimp = nimp.value
  else:
    ne = core_profiles.ne.value
    ni = core_profiles.ni.value
    nimp = core_profiles.nimp.value

  return {
      'temp_ion': temp_ion,
      'temp_el': temp_el,
      'ne': ne,
      'ni': ni,
      'nimp': nimp,
  }


def update_evolving_core_profiles(
    x_new: tuple[cell_variable.CellVariable, ...],
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    evolving_names: tuple[str, ...],
) -> state.CoreProfiles:
  """Returns the new core profiles after updating the evolving variables.

  Args:
    x_new: The new values of the evolving variables.
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

  ni, nimp = _updated_ion_density(dynamic_runtime_params_slice, geo, ne)

  return dataclasses.replace(
      core_profiles,
      temp_ion=temp_ion,
      temp_el=temp_el,
      psi=psi,
      ne=ne,
      ni=ni,
      nimp=nimp,
  )


def compute_boundary_conditions(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
) -> dict[str, dict[str, jax.Array | None]]:
  """Computes boundary conditions for time t and returns updates to State.

  Args:
    dynamic_runtime_params_slice: Runtime parameters at time t.
    geo: Geometry object

  Returns:
    Mapping from State attribute names to dictionaries updating attributes of
    each CellVariable in the state. This dict can in theory recursively replace
    values in a State object.
  """
  Ti_bound_right = jax_utils.error_if_not_positive(  # pylint: disable=invalid-name
      dynamic_runtime_params_slice.profile_conditions.Ti_bound_right,
      'Ti_bound_right',
  )

  Te_bound_right = jax_utils.error_if_not_positive(  # pylint: disable=invalid-name
      dynamic_runtime_params_slice.profile_conditions.Te_bound_right,
      'Te_bound_right',
  )

  ne = _get_ne(
      dynamic_runtime_params_slice,
      geo,
  )
  ne_bound_right = ne.right_face_constraint

  # define ion profile based on (flat) Zeff and single assumed impurity
  # with Zimp. main ion limited to hydrogenic species for now.
  # Assume isotopic balance for DT fusion power. Solve for ni based on:
  # Zeff = (Zi * ni + Zimp**2 * nimp)/ne  ;  nimp*Zimp + ni*Zi = ne

  dilution_factor_edge = physics.get_main_ion_dilution_factor(
      dynamic_runtime_params_slice.plasma_composition.main_ion.avg_Z,
      dynamic_runtime_params_slice.plasma_composition.impurity.avg_Z,
      dynamic_runtime_params_slice.plasma_composition.Zeff_face[-1],
  )

  ni_bound_right = ne_bound_right * dilution_factor_edge
  nimp_bound_right = (
      ne_bound_right
      - ni_bound_right
      * dynamic_runtime_params_slice.plasma_composition.main_ion.avg_Z
  ) / dynamic_runtime_params_slice.plasma_composition.impurity.avg_Z

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
          right_face_grad_constraint=_calculate_psi_grad_constraint(
              dynamic_runtime_params_slice,
              geo,
          ),
      ),
  }


# pylint: disable=invalid-name
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

# pylint: enable=invalid-name

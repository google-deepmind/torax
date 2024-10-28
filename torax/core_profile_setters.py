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
from torax import geometry
from torax import jax_utils
from torax import math_utils
from torax import physics
from torax import state
from torax.config import runtime_params_slice
from torax.fvm import cell_variable
from torax.geometry import Geometry  # pylint: disable=g-importing-member
from torax.sources import generic_current_source
from torax.sources import ohmic_heat_source
from torax.sources import source_models as source_models_lib
from torax.sources import source_profiles as source_profiles_lib

_trapz = jax.scipy.integrate.trapezoid


def updated_ion_temperature(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: Geometry,
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
    geo: Geometry,
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
    geo: Geometry,
) -> cell_variable.CellVariable:
  """Helper to get the electron density profile at the current timestep."""
  # pylint: disable=invalid-name
  nGW = (
      dynamic_runtime_params_slice.profile_conditions.Ip
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
    # find normalization factor such that desired line-averaged n is set
    # Assumes line-averaged central chord on outer midplane
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
    geo: Geometry,
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

  Zimp = dynamic_runtime_params_slice.plasma_composition.Zimp
  Zeff = dynamic_runtime_params_slice.plasma_composition.Zeff
  Zeff_face = dynamic_runtime_params_slice.plasma_composition.Zeff_face

  dilution_factor = physics.get_main_ion_dilution_factor(Zimp, Zeff)
  dilution_factor_edge = physics.get_main_ion_dilution_factor(
      Zimp, Zeff_face[-1]
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
      value=(ne.value - ni.value) / Zimp,
      dr=geo.drho_norm,
      right_face_grad_constraint=None,
      right_face_constraint=jnp.array(
          ne.right_face_constraint - ni.right_face_constraint
      )
      / Zimp,
  )
  return ni, nimp


def updated_density(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: Geometry,
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
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: Geometry,
    source_models: source_models_lib.SourceModels,
) -> state.Currents:
  """Creates the initial Currents without the bootstrap current.

  Args:
    dynamic_runtime_params_slice: General runtime parameters at t_initial.
    geo: Geometry of the tokamak.
    source_models: All TORAX source/sink functions.

  Returns:
    currents: Initial Currents
  """

  # Many variables throughout this function are capitalized based on physics
  # notational conventions rather than on Google Python style
  # pylint: disable=invalid-name

  # Calculate splitting of currents depending on input runtime params.
  Ip = dynamic_runtime_params_slice.profile_conditions.Ip

  dynamic_generic_current_params = get_generic_current_params(
      dynamic_runtime_params_slice, source_models
  )
  if dynamic_generic_current_params.use_absolute_current:
    Iext = dynamic_generic_current_params.Iext
  else:
    Iext = Ip * dynamic_generic_current_params.fext
  # Total Ohmic current
  Iohm = Ip - Iext

  # Set zero bootstrap current
  bootstrap_profile = source_profiles_lib.BootstrapCurrentProfile.zero_profile(
      geo
  )

  # calculate "External" current profile (e.g. ECCD)
  # form of external current on face grid
  generic_current_face = source_models.generic_current_source.get_value(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      dynamic_source_runtime_params=dynamic_generic_current_params,
      geo=geo,
  )
  generic_current = geometry.face_to_cell(generic_current_face)

  # construct prescribed current formula on grid.
  jformula_face = (
      1 - geo.rho_face_norm**2
  ) ** dynamic_runtime_params_slice.profile_conditions.nu
  # calculate total and Ohmic current profiles
  denom = _trapz(jformula_face * geo.spr_face, geo.rho_face_norm)
  if dynamic_runtime_params_slice.profile_conditions.initial_j_is_total_current:
    Ctot = Ip * 1e6 / denom
    jtot_face = jformula_face * Ctot
    jtot = geometry.face_to_cell(jtot_face)
    johm = jtot - generic_current
  else:
    Cohm = Iohm * 1e6 / denom
    johm_face = jformula_face * Cohm
    johm = geometry.face_to_cell(johm_face)
    jtot_face = johm_face + generic_current_face
    jtot = geometry.face_to_cell(jtot_face)

  jtot_hires = _get_jtot_hires(
      dynamic_runtime_params_slice,
      dynamic_generic_current_params,
      geo,
      bootstrap_profile,
      Iohm,
      source_models.generic_current_source,
  )

  currents = state.Currents(
      jtot=jtot,
      jtot_face=jtot_face,
      jtot_hires=jtot_hires,
      johm=johm,
      generic_current_source=generic_current,
      j_bootstrap=bootstrap_profile.j_bootstrap,
      j_bootstrap_face=bootstrap_profile.j_bootstrap_face,
      I_bootstrap=bootstrap_profile.I_bootstrap,
      Ip=Ip,
      sigma=bootstrap_profile.sigma,
  )

  return currents


def _prescribe_currents_with_bootstrap(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: Geometry,
    temp_ion: cell_variable.CellVariable,
    temp_el: cell_variable.CellVariable,
    ne: cell_variable.CellVariable,
    ni: cell_variable.CellVariable,
    psi: cell_variable.CellVariable,
    source_models: source_models_lib.SourceModels,
) -> state.Currents:
  """Creates the initial Currents.

  Args:
    dynamic_runtime_params_slice: General runtime parameters at t_initial.
    geo: Geometry of the tokamak.
    temp_ion: Ion temperature.
    temp_el: Electron temperature.
    ne: Electron density.
    ni: Main ion density.
    psi: Poloidal flux.
    source_models: All TORAX source/sink functions. If not provided, uses the
      default sources.

  Returns:
    currents: Plasma currents
  """

  # Many variables throughout this function are capitalized based on physics
  # notational conventions rather than on Google Python style
  # pylint: disable=invalid-name
  Ip = dynamic_runtime_params_slice.profile_conditions.Ip

  bootstrap_profile = source_models.j_bootstrap.get_value(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      dynamic_source_runtime_params=dynamic_runtime_params_slice.sources[
          source_models.j_bootstrap_name
      ],
      geo=geo,
      temp_ion=temp_ion,
      temp_el=temp_el,
      ne=ne,
      ni=ni,
      psi=psi,
  )
  f_bootstrap = bootstrap_profile.I_bootstrap / (Ip * 1e6)

  # Calculate splitting of currents depending on input runtime params
  dynamic_generic_current_params = get_generic_current_params(
      dynamic_runtime_params_slice, source_models
  )
  if dynamic_generic_current_params.use_absolute_current:
    Iext = dynamic_generic_current_params.Iext
  else:
    Iext = Ip * dynamic_generic_current_params.fext
  Iohm = Ip - Iext - f_bootstrap * Ip

  # calculate "External" current profile (e.g. ECCD)
  # form of external current on face grid
  generic_current_face = source_models.generic_current_source.get_value(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      dynamic_source_runtime_params=dynamic_generic_current_params,
      geo=geo,
  )
  generic_current = geometry.face_to_cell(generic_current_face)

  # construct prescribed current formula on grid.
  jformula_face = (
      1 - geo.rho_face_norm**2
  ) ** dynamic_runtime_params_slice.profile_conditions.nu
  denom = _trapz(jformula_face * geo.spr_face, geo.rho_face_norm)
  # calculate total and Ohmic current profiles
  if dynamic_runtime_params_slice.profile_conditions.initial_j_is_total_current:
    Ctot = Ip * 1e6 / denom
    jtot_face = jformula_face * Ctot
    johm_face = (
        jtot_face - generic_current_face - bootstrap_profile.j_bootstrap_face
    )
  else:
    Cohm = Iohm * 1e6 / denom
    johm_face = jformula_face * Cohm
    jtot_face = (
        johm_face + generic_current_face + bootstrap_profile.j_bootstrap_face
    )

  jtot = geometry.face_to_cell(jtot_face)
  johm = geometry.face_to_cell(johm_face)

  jtot_hires = _get_jtot_hires(
      dynamic_runtime_params_slice,
      dynamic_generic_current_params,
      geo,
      bootstrap_profile,
      Iohm,
      source_models.generic_current_source,
  )

  currents = state.Currents(
      jtot=jtot,
      jtot_face=jtot_face,
      jtot_hires=jtot_hires,
      johm=johm,
      generic_current_source=generic_current,
      j_bootstrap=bootstrap_profile.j_bootstrap,
      j_bootstrap_face=bootstrap_profile.j_bootstrap_face,
      I_bootstrap=bootstrap_profile.I_bootstrap,
      Ip=dynamic_runtime_params_slice.profile_conditions.Ip,
      sigma=bootstrap_profile.sigma,
  )

  return currents


def _calculate_currents_from_psi(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: Geometry,
    temp_ion: cell_variable.CellVariable,
    temp_el: cell_variable.CellVariable,
    ne: cell_variable.CellVariable,
    ni: cell_variable.CellVariable,
    psi: cell_variable.CellVariable,
    source_models: source_models_lib.SourceModels,
) -> state.Currents:
  """Creates the initial Currents using psi to calculate jtot.

  Args:
    dynamic_runtime_params_slice: General runtime parameters at t_initial.
    geo: Geometry of the tokamak.
    temp_ion: Ion temperature.
    temp_el: Electron temperature.
    ne: Electron density.
    ni: Main ion density.
    psi: Poloidal flux.
    source_models: All TORAX source/sink functions. If not provided, uses the
      default sources.

  Returns:
    currents: Plasma currents
  """

  # Many variables throughout this function are capitalized based on physics
  # notational conventions rather than on Google Python style
  # pylint: disable=invalid-name

  jtot, jtot_face = physics.calc_jtot_from_psi(
      geo,
      psi,
  )

  bootstrap_profile = source_models.j_bootstrap.get_value(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      dynamic_source_runtime_params=dynamic_runtime_params_slice.sources[
          source_models.j_bootstrap_name
      ],
      geo=geo,
      temp_ion=temp_ion,
      temp_el=temp_el,
      ne=ne,
      ni=ni,
      psi=psi,
  )

  # Calculate splitting of currents depending on input runtime params.
  dynamic_generic_current_params = get_generic_current_params(
      dynamic_runtime_params_slice, source_models
  )

  # calculate "External" current profile (e.g. ECCD)
  # form of external current on face grid
  generic_current_face = source_models.generic_current_source.get_value(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      dynamic_source_runtime_params=dynamic_generic_current_params,
      geo=geo,
  )
  generic_current = geometry.face_to_cell(generic_current_face)

  # TODO(b/336995925): TORAX currently only uses the external current source,
  # generic_current, when computing the jtot initial currents from psi.
  # should be summing over all sources that can contribute current i.e. ECCD,
  # ICRH, NBI, LHCD.
  johm = jtot - generic_current - bootstrap_profile.j_bootstrap

  currents = state.Currents(
      jtot=jtot,
      jtot_face=jtot_face,
      jtot_hires=None,
      johm=johm,
      generic_current_source=generic_current,
      j_bootstrap=bootstrap_profile.j_bootstrap,
      j_bootstrap_face=bootstrap_profile.j_bootstrap_face,
      I_bootstrap=bootstrap_profile.I_bootstrap,
      Ip=dynamic_runtime_params_slice.profile_conditions.Ip,
      sigma=bootstrap_profile.sigma,
  )

  return currents


def _update_psi_from_j(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: Geometry,
    currents: state.Currents,
) -> cell_variable.CellVariable:
  """Calculates poloidal flux (psi) consistent with plasma current.

  For increased accuracy of psi, a hi-res grid is used, due to the double
    integration.

  Args:
    dynamic_runtime_params_slice: Dynamic runtime parameters.
    geo: Torus geometry.
    currents: Currents structure including high resolution version of jtot.

  Returns:
    psi: Poloidal flux cell variable.
  """
  psi_grad_constraint = _calculate_psi_grad_constraint(
      dynamic_runtime_params_slice,
      geo,
  )

  y = currents.jtot_hires * geo.spr_hires
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
    geo: Geometry,
) -> jax.Array:
  """Calculates the constraint on the poloidal flux (psi)."""
  return (
      dynamic_runtime_params_slice.profile_conditions.Ip
      * 1e6
      * (16 * jnp.pi**3 * constants.CONSTANTS.mu0 * geo.Phib)
      / (geo.g2g3_over_rhon_face[-1] * geo.F_face[-1])
  )


def _initial_psi(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: Geometry,
    temp_ion: cell_variable.CellVariable,
    temp_el: cell_variable.CellVariable,
    ne: cell_variable.CellVariable,
    ni: cell_variable.CellVariable,
    source_models: source_models_lib.SourceModels,
) -> tuple[cell_variable.CellVariable, state.Currents]:
  """Calculates poloidal flux (psi) consistent with plasma current.

  There are three modes of initialising psi that are supported:
  1. Providing psi from the profile conditions.
  2. Calculating j according to the "nu formula".
  3. Retrieving psi from the standard geometry input.

  Args:
    dynamic_runtime_params_slice: Dynamic runtime parameters.
    geo: Torus geometry.
    temp_ion: Ion temperature.
    temp_el: Electron temperature.
    ne: Electron density.
    ni: Ion density.
    source_models: All TORAX source/sink functions.

  Returns:
    psi: Poloidal flux cell variable.
    currents: Plasma currents.
  """
  if dynamic_runtime_params_slice.profile_conditions.psi is not None:
    psi = cell_variable.CellVariable(
        value=dynamic_runtime_params_slice.profile_conditions.psi,
        right_face_grad_constraint=_calculate_psi_grad_constraint(
            dynamic_runtime_params_slice,
            geo,
        ),
        dr=geo.drho_norm,
    )
    currents = _calculate_currents_from_psi(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        temp_ion=temp_ion,
        temp_el=temp_el,
        ne=ne,
        ni=ni,
        psi=psi,
        source_models=source_models,
    )
    return psi, currents

  # set up initial psi profile based on current profile
  if (
      isinstance(geo, geometry.CircularAnalyticalGeometry)
      or dynamic_runtime_params_slice.profile_conditions.initial_psi_from_j
  ):
    # set up initial current profile without bootstrap current, to get
    # q-profile approximation (needed for bootstrap)
    currents_no_bootstrap = _prescribe_currents_no_bootstrap(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        source_models=source_models,
    )

    psi_no_bootstrap = _update_psi_from_j(
        dynamic_runtime_params_slice,
        geo,
        currents_no_bootstrap,
    )

    # second iteration, with bootstrap current
    currents = _prescribe_currents_with_bootstrap(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        temp_ion=temp_ion,
        temp_el=temp_el,
        ne=ne,
        ni=ni,
        psi=psi_no_bootstrap,
        source_models=source_models,
    )

    psi = _update_psi_from_j(
        dynamic_runtime_params_slice,
        geo,
        currents,
    )

  elif (
      isinstance(geo, geometry.StandardGeometry)
      and not dynamic_runtime_params_slice.profile_conditions.initial_psi_from_j
  ):
    # psi is already provided from a numerical equilibrium, so no need to
    # first calculate currents. However, non-inductive currents are still
    # calculated and used in current diffusion equation.
    psi_grad_constraint = _calculate_psi_grad_constraint(
        dynamic_runtime_params_slice,
        geo,
    )
    psi = cell_variable.CellVariable(
        value=geo.psi_from_Ip,
        right_face_grad_constraint=psi_grad_constraint,
        dr=geo.drho_norm,
    )

    # Calculate external currents
    currents = _calculate_currents_from_psi(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        source_models=source_models,
        temp_ion=temp_ion,
        temp_el=temp_el,
        ne=ne,
        ni=ni,
        psi=psi,
    )
  else:
    raise ValueError(f'Unknown geometry type provided: {geo}')

  return psi, currents


def initial_core_profiles(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: Geometry,
    source_models: source_models_lib.SourceModels,
) -> state.CoreProfiles:
  """Calculates the initial core profiles.

  Args:
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
  psi, currents = _initial_psi(
      dynamic_runtime_params_slice,
      geo,
      temp_ion=temp_ion,
      temp_el=temp_el,
      ne=ne,
      ni=ni,
      source_models=source_models,
  )
  s_face = physics.calc_s_from_psi(geo, psi)
  q_face, _ = physics.calc_q_from_psi(
      geo=geo,
      psi=psi,
      q_correction_factor=dynamic_runtime_params_slice.numerics.q_correction_factor,
  )

  # the psidot calculation needs core profiles. So psidot first initialized
  # with zeros.
  psidot = cell_variable.CellVariable(
      value=jnp.zeros_like(psi.value),
      dr=geo.drho_norm,
  )

  core_profiles = state.CoreProfiles(
      temp_ion=temp_ion,
      temp_el=temp_el,
      ne=ne,
      ni=ni,
      Zi=dynamic_runtime_params_slice.plasma_composition.Zi,
      Ai=dynamic_runtime_params_slice.plasma_composition.Ai,
      nimp=nimp,
      Zimp=dynamic_runtime_params_slice.plasma_composition.Zimp,
      psi=psi,
      psidot=psidot,
      currents=currents,
      q_face=q_face,
      s_face=s_face,
      nref=jnp.asarray(dynamic_runtime_params_slice.numerics.nref),
  )

  # psidot calculated here with phibdot=0 in geo, since this is initial
  # conditions and we don't yet have information on geo_t_plus_dt for the
  # phibdot calculation.
  psidot = dataclasses.replace(
      psidot,
      value=ohmic_heat_source.calc_psidot(
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
    geo: Geometry,
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
    geo: Geometry,
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
  # Zeff = (ni + Zimp**2 * nimp)/ne  ;  nimp*Zimp + ni = ne

  dilution_factor_edge = physics.get_main_ion_dilution_factor(
      dynamic_runtime_params_slice.plasma_composition.Zimp,
      dynamic_runtime_params_slice.plasma_composition.Zeff_face[-1],
  )

  ni_bound_right = ne_bound_right * dilution_factor_edge
  nimp_bound_right = (
      ne_bound_right - ni_bound_right
  ) / dynamic_runtime_params_slice.plasma_composition.Zimp

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
    dynamic_generic_current_params: generic_current_source.DynamicRuntimeParams,
    geo: Geometry,
    bootstrap_profile: source_profiles_lib.BootstrapCurrentProfile,
    Iohm: jax.Array | float,
    generic_current: generic_current_source.GenericCurrentSource,
) -> jax.Array:
  """Calculates jtot hires."""
  j_bootstrap_hires = jnp.interp(
      geo.rho_hires, geo.rho_face, bootstrap_profile.j_bootstrap_face
  )

  # calculate hi-res "External" current profile (e.g. ECCD) on cell grid.
  generic_current_hires = generic_current.generic_current_source_hires(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      dynamic_source_runtime_params=dynamic_generic_current_params,
      geo=geo,
  )

  # calculate high resolution jtot and Ohmic current profile
  jformula_hires = (
      1 - geo.rho_hires_norm**2
  ) ** dynamic_runtime_params_slice.profile_conditions.nu
  denom = _trapz(jformula_hires * geo.spr_hires, geo.rho_hires_norm)
  if dynamic_runtime_params_slice.profile_conditions.initial_j_is_total_current:
    Ctot_hires = (
        dynamic_runtime_params_slice.profile_conditions.Ip * 1e6 / denom
    )
    jtot_hires = jformula_hires * Ctot_hires
  else:
    Cohm_hires = Iohm * 1e6 / denom
    johm_hires = jformula_hires * Cohm_hires
    jtot_hires = johm_hires + generic_current_hires + j_bootstrap_hires
  return jtot_hires


def get_generic_current_params(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    source_models: source_models_lib.SourceModels,
) -> generic_current_source.DynamicRuntimeParams:
  """Returns dynamic runtime params for the external current source."""
  assert (
      source_models.generic_current_source_name
      in dynamic_runtime_params_slice.sources
  ), (
      f'{source_models.generic_current_source_name} not found in'
      ' dynamic_runtime_params_slice.sources. Check to make sure the'
      ' DynamicRuntimeParamsSlice was built with `sources` that include the'
      ' external current source.'
  )
  dynamic_generic_current_params = dynamic_runtime_params_slice.sources[
      source_models.generic_current_source_name
  ]
  assert isinstance(
      dynamic_generic_current_params,
      generic_current_source.DynamicRuntimeParams,
  )
  return dynamic_generic_current_params


# pylint: enable=invalid-name

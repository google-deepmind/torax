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

"""Functions used for initializing core profiles."""

import dataclasses
import jax
from jax import numpy as jnp
from torax import array_typing
from torax import constants
from torax import math_utils
from torax import state
from torax.config import runtime_params_slice
from torax.core_profiles import getters
from torax.fvm import cell_variable
from torax.geometry import geometry
from torax.geometry import standard_geometry
from torax.physics import psi_calculations
from torax.sources import source_models as source_models_lib
from torax.sources import source_operations
from torax.sources import source_profile_builders
from torax.sources import source_profiles as source_profiles_lib

_trapz = jax.scipy.integrate.trapezoid

# pylint: disable=invalid-name


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
  temp_ion = getters.get_updated_ion_temperature(
      dynamic_runtime_params_slice.profile_conditions, geo
  )
  temp_el = getters.get_updated_electron_temperature(
      dynamic_runtime_params_slice.profile_conditions, geo
  )
  ne = getters.get_updated_electron_density(
      dynamic_runtime_params_slice.numerics,
      dynamic_runtime_params_slice.profile_conditions,
      geo,
  )

  ni, nimp, Zi, Zi_face, Zimp, Zimp_face = (
      getters.get_ion_density_and_charge_states(
          static_runtime_params_slice,
          dynamic_runtime_params_slice,
          geo,
          ne,
          temp_el,
      )
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

  core_profiles = _init_psi_psidot_vloop_and_current(
      static_runtime_params_slice,
      dynamic_runtime_params_slice,
      geo,
      core_profiles,
      source_models,
  )

  jtot, jtot_face, Ip_profile_face = psi_calculations.calc_jtot(
      geo, core_profiles.psi
  )
  currents = dataclasses.replace(
      core_profiles.currents,
      jtot=jtot,
      jtot_face=jtot_face,
      Ip_profile_face=Ip_profile_face,
  )
  core_profiles = dataclasses.replace(
      core_profiles,
      currents=currents,
      q_face=psi_calculations.calc_q_face(geo, core_profiles.psi),
      s_face=psi_calculations.calc_s_face(geo, core_profiles.psi),
  )
  return core_profiles


def _prescribe_currents(
    bootstrap_profile: source_profiles_lib.BootstrapCurrentProfile,
    external_current: jax.Array,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
) -> state.Currents:
  """Creates the initial Currents from a given bootstrap profile."""

  Ip = dynamic_runtime_params_slice.profile_conditions.Ip_tot
  f_bootstrap = bootstrap_profile.I_bootstrap / (Ip * 1e6)

  Iext = math_utils.area_integration(external_current, geo) / 10**6
  Iohm = Ip - Iext - f_bootstrap * Ip

  # construct prescribed current formula on grid.
  jformula = (
      1 - geo.rho_norm**2
  ) ** dynamic_runtime_params_slice.profile_conditions.nu
  denom = _trapz(jformula * geo.spr, geo.rho_norm)
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
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_profiles: source_profiles_lib.SourceProfiles,
) -> state.Currents:
  """Creates the initial Currents using psi to calculate jtot."""
  jtot, jtot_face, Ip_profile_face = psi_calculations.calc_jtot(
      geo,
      core_profiles.psi,
  )
  bootstrap_profile = source_profiles.j_bootstrap
  # Note that the psi sources here are the standard sources and don't include
  # the bootstrap current.
  external_current = sum(source_profiles.psi.values())
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
    Ip_tot: array_typing.ScalarFloat,
    geo: geometry.Geometry,
    jtot_hires: jax.Array,
    use_vloop_lcfs_boundary_condition: bool = False,
) -> cell_variable.CellVariable:
  """Calculates poloidal flux (psi) consistent with plasma current.

  For increased accuracy of psi, a hi-res grid is used, due to the double
    integration. Presently used only for initialization. Therefore Ip_tot is
    a valid source of truth for Ip, even if use_vloop_lcfs_boundary_condition
    is True.

  Args:
    Ip_tot: Total plasma current [MA].
    geo: Torus geometry.
    jtot_hires: High resolution version of jtot [A/m^2].
    use_vloop_lcfs_boundary_condition: Whether to set the loop voltage from
      Ip_tot.

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
  dpsi_drhonorm_edge = (
      psi_calculations.calculate_psi_grad_constraint_from_Ip_tot(
          Ip_tot,
          geo,
      )
  )

  if use_vloop_lcfs_boundary_condition:
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


def _init_psi_psidot_vloop_and_current(
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
  use_vloop_bc = (
      dynamic_runtime_params_slice.profile_conditions.use_vloop_lcfs_boundary_condition
  )

  source_profiles = source_profiles_lib.SourceProfiles(
      j_bootstrap=source_profiles_lib.BootstrapCurrentProfile.zero_profile(geo),
      qei=source_profiles_lib.QeiInfo.zeros(geo),
  )
  # Updates the calculated source profiles with the standard source profiles.
  source_profile_builders.build_standard_source_profiles(
      static_runtime_params_slice=static_runtime_params_slice,
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      geo=geo,
      core_profiles=core_profiles,
      source_models=source_models,
      psi_only=True,
      calculate_anyway=True,
      calculated_source_profiles=source_profiles,
  )

  # Case 1: retrieving psi from the profile conditions, using the prescribed
  # profile and Ip_tot
  if dynamic_runtime_params_slice.profile_conditions.psi is not None:
    # Calculate the dpsi/drho necessary to achieve the given Ip_tot
    dpsi_drhonorm_edge = (
        psi_calculations.calculate_psi_grad_constraint_from_Ip_tot(
            dynamic_runtime_params_slice.profile_conditions.Ip_tot,
            geo,
        )
    )

    # Set the BCs to ensure the correct Ip_tot
    if use_vloop_bc:
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

    core_profiles = dataclasses.replace(
        core_profiles,
        psi=psi,
        q_face=psi_calculations.calc_q_face(geo, psi),
        s_face=psi_calculations.calc_s_face(geo, psi),
    )
    bootstrap_profile = source_models.j_bootstrap.get_bootstrap(
        static_runtime_params_slice=static_runtime_params_slice,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        core_profiles=core_profiles,
    )
    source_profiles = dataclasses.replace(
        source_profiles, j_bootstrap=bootstrap_profile
    )
    currents = _calculate_currents_from_psi(
        geo=geo,
        core_profiles=core_profiles,
        source_profiles=source_profiles,
    )

  # Case 2: retrieving psi from the standard geometry input.
  elif (
      isinstance(geo, standard_geometry.StandardGeometry)
      and not dynamic_runtime_params_slice.profile_conditions.initial_psi_from_j
  ):
    # psi is already provided from a numerical equilibrium, so no need to
    # first calculate currents. However, non-inductive currents are still
    # calculated and used in current diffusion equation.

    # Calculate the dpsi/drho necessary to achieve the given Ip_tot
    dpsi_drhonorm_edge = (
        psi_calculations.calculate_psi_grad_constraint_from_Ip_tot(
            dynamic_runtime_params_slice.profile_conditions.Ip_tot,
            geo,
        )
    )

    # Use the psi from the equilibrium as the right face constraint
    # This has already been made consistent with the desired Ip_tot
    # by make_ip_consistent
    psi = cell_variable.CellVariable(
        value=geo.psi_from_Ip,  # Use psi from equilibrium
        right_face_grad_constraint=None if use_vloop_bc else dpsi_drhonorm_edge,
        right_face_constraint=geo.psi_from_Ip_face[-1]
        if use_vloop_bc
        else None,
        dr=geo.drho_norm,
    )
    core_profiles = dataclasses.replace(
        core_profiles,
        psi=psi,
        q_face=psi_calculations.calc_q_face(geo, psi),
        s_face=psi_calculations.calc_s_face(geo, psi),
    )

    # Calculate non-inductive currents
    bootstrap_profile = source_models.j_bootstrap.get_bootstrap(
        static_runtime_params_slice=static_runtime_params_slice,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        core_profiles=core_profiles,
    )
    source_profiles = dataclasses.replace(
        source_profiles, j_bootstrap=bootstrap_profile
    )
    currents = _calculate_currents_from_psi(
        geo=geo,
        core_profiles=core_profiles,
        source_profiles=source_profiles,
    )

  # Case 3: calculating j according to nu formula and psi from j.
  else:
    # First calculate currents without bootstrap.
    external_current = sum(source_profiles.psi.values())
    currents = _prescribe_currents(
        bootstrap_profile=source_profiles.j_bootstrap,
        external_current=external_current,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
    )
    psi = _update_psi_from_j(
        dynamic_runtime_params_slice.profile_conditions.Ip_tot,
        geo,
        currents.jtot_hires,
        use_vloop_lcfs_boundary_condition=use_vloop_bc,
    )
    core_profiles = dataclasses.replace(
        core_profiles,
        currents=currents,
        psi=psi,
        q_face=psi_calculations.calc_q_face(geo, psi),
        s_face=psi_calculations.calc_s_face(geo, psi),
    )
    # Now calculate currents with bootstrap.
    bootstrap_profile = source_models.j_bootstrap.get_bootstrap(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_runtime_params_slice,
        geo=geo,
        core_profiles=core_profiles,
    )
    currents = _prescribe_currents(
        bootstrap_profile=bootstrap_profile,
        external_current=external_current,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
    )
    psi = _update_psi_from_j(
        dynamic_runtime_params_slice.profile_conditions.Ip_tot,
        geo,
        currents.jtot_hires,
        use_vloop_lcfs_boundary_condition=use_vloop_bc,
    )
    _, _, Ip_profile_face = psi_calculations.calc_jtot(
        geo,
        psi,
    )
    currents = dataclasses.replace(currents, Ip_profile_face=Ip_profile_face)

  core_profiles = dataclasses.replace(
      core_profiles,
      psi=psi,
      q_face=psi_calculations.calc_q_face(geo, psi),
      s_face=psi_calculations.calc_s_face(geo, psi),
      currents=currents,
      vloop_lcfs=dynamic_runtime_params_slice.profile_conditions.vloop_lcfs,
  )
  bootstrap_profile = source_models.j_bootstrap.get_bootstrap(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      static_runtime_params_slice=static_runtime_params_slice,
      geo=geo,
      core_profiles=core_profiles,
  )
  source_profiles = dataclasses.replace(
      source_profiles, j_bootstrap=bootstrap_profile
  )
  # psidot calculated here with phibdot=0 in geo, since this is initial
  # conditions and we don't yet have information on geo_t_plus_dt for the
  # phibdot calculation.
  psi_sources = source_operations.sum_sources_psi(geo, source_profiles)
  sigma = source_profiles.j_bootstrap.sigma
  sigma_face = source_profiles.j_bootstrap.sigma_face
  psidot = psi_calculations.calculate_psidot_from_psi_sources(
      psi_sources=psi_sources,
      sigma=sigma,
      sigma_face=sigma_face,
      resistivity_multiplier=dynamic_runtime_params_slice.numerics.resistivity_mult,
      psi=psi,
      geo=geo,
  )
  psidot_cell_var = dataclasses.replace(core_profiles.psidot, value=psidot)
  # TODO(b/396374895): For Ip_tot BC, introduce a feature for calculating
  # vloop_lcfs in final post-processing and test to check vloop equivalence
  # between vloop BC and Ip_tot BC
  core_profiles = dataclasses.replace(
      core_profiles,
      psidot=psidot_cell_var,
      vloop_lcfs=(
          dynamic_runtime_params_slice.profile_conditions.vloop_lcfs
          if use_vloop_bc
          else 0.0
      ),
  )

  return core_profiles


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

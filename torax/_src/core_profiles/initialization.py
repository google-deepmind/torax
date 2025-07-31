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
import numpy as np
from torax._src import array_typing
from torax._src import constants
from torax._src import jax_utils
from torax._src import math_utils
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.core_profiles import getters
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.geometry import standard_geometry
from torax._src.neoclassical import neoclassical_models as neoclassical_models_lib
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.physics import psi_calculations
from torax._src.sources import source_models as source_models_lib
from torax._src.sources import source_profile_builders

_trapz = jax.scipy.integrate.trapezoid

# pylint: disable=invalid-name


def initial_core_profiles(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    source_models: source_models_lib.SourceModels,
    neoclassical_models: neoclassical_models_lib.NeoclassicalModels,
) -> state.CoreProfiles:
  """Calculates the initial core profiles.

  Args:
    static_runtime_params_slice: Static runtime parameters.
    dynamic_runtime_params_slice: Dynamic runtime parameters at t=t_initial.
    geo: Torus geometry.
    source_models: All models for TORAX sources/sinks.
    neoclassical_models: All models for neoclassical calculations.

  Returns:
    Initial core profiles.
  """

  # To set initial values and compute the boundary conditions, we need to handle
  # potentially time-varying inputs from the users.
  # The default time in build_dynamic_runtime_params_slice is t_initial
  T_i = getters.get_updated_ion_temperature(
      dynamic_runtime_params_slice.profile_conditions, geo
  )
  T_e = getters.get_updated_electron_temperature(
      dynamic_runtime_params_slice.profile_conditions, geo
  )
  n_e = getters.get_updated_electron_density(
      dynamic_runtime_params_slice.profile_conditions,
      geo,
  )

  ions = getters.get_updated_ions(
      static_runtime_params_slice,
      dynamic_runtime_params_slice,
      geo,
      n_e,
      T_e,
  )
  # Set v_loop_lcfs. Two branches:
  # 1. Set the v_loop_lcfs from profile_conditions if using the v_loop BC option
  # 2. Initialize v_loop_lcfs to 0 if using the Ip boundary condition for psi.
  # In case 2, v_loop_lcfs will be updated every timestep based on the psi_lcfs
  # values across the time interval. Since there is is one more time value than
  # time intervals, the v_loop_lcfs time-series is underconstrained. Therefore,
  # after the first timestep we reset v_loop_lcfs[0] to v_loop_lcfs[1].

  v_loop_lcfs = (
      np.array(dynamic_runtime_params_slice.profile_conditions.v_loop_lcfs)
      if dynamic_runtime_params_slice.profile_conditions.use_v_loop_lcfs_boundary_condition
      else np.array(0.0, dtype=jax_utils.get_dtype())
  )

  # Initialise psi and derived quantities to zero before they are calculated.
  psidot = cell_variable.CellVariable(
      value=np.zeros_like(geo.rho),
      dr=geo.drho_norm,
  )
  psi = cell_variable.CellVariable(
      value=np.zeros_like(geo.rho), dr=geo.drho_norm
  )

  core_profiles = state.CoreProfiles(
      T_i=T_i,
      T_e=T_e,
      n_e=n_e,
      n_i=ions.n_i,
      Z_i=ions.Z_i,
      Z_i_face=ions.Z_i_face,
      A_i=ions.A_i,
      n_impurity=ions.n_impurity,
      Z_impurity=ions.Z_impurity,
      Z_impurity_face=ions.Z_impurity_face,
      A_impurity=ions.A_impurity,
      Z_eff=ions.Z_eff,
      Z_eff_face=ions.Z_eff_face,
      psi=psi,
      psidot=psidot,
      q_face=np.zeros_like(geo.rho_face),
      s_face=np.zeros_like(geo.rho_face),
      v_loop_lcfs=v_loop_lcfs,
      sigma=np.zeros_like(geo.rho),
      sigma_face=np.zeros_like(geo.rho_face),
      j_total=np.zeros_like(geo.rho),
      j_total_face=np.zeros_like(geo.rho_face),
      Ip_profile_face=np.zeros_like(geo.rho_face),
  )

  return _init_psi_and_psi_derived(
      static_runtime_params_slice,
      dynamic_runtime_params_slice,
      geo,
      core_profiles,
      source_models,
      neoclassical_models,
  )


def update_psi_from_j(
    Ip: array_typing.ScalarFloat,
    geo: geometry.Geometry,
    j_total_hires: jax.Array,
    use_v_loop_lcfs_boundary_condition: bool = False,
) -> cell_variable.CellVariable:
  """Calculates poloidal flux (psi) consistent with plasma current.

  For increased accuracy of psi, a hi-res grid is used, due to the double
    integration. Presently used only for initialization. Therefore Ip is
    a valid source of truth for Ip, even if use_v_loop_lcfs_boundary_condition
    is True.

  Args:
    Ip: Total plasma current [A].
    geo: Torus geometry.
    j_total_hires: High resolution version of j_total [A/m^2].
    use_v_loop_lcfs_boundary_condition: Whether to set the loop voltage from Ip.

  Returns:
    psi: Poloidal flux cell variable.
  """
  y = j_total_hires * geo.spr_hires
  assert y.ndim == 1
  assert geo.rho_hires.ndim == 1
  Ip_profile = math_utils.cumulative_trapezoid(
      y=y, x=geo.rho_hires_norm, initial=0.0
  )
  scale = jnp.concatenate((
      jnp.zeros((1,)),
      (16 * jnp.pi**3 * constants.CONSTANTS.mu0 * geo.Phi_b)
      / (geo.F_hires[1:] * geo.g2g3_over_rhon_hires[1:]),
  ))
  # dpsi_dr on hires cell grid
  dpsi_drhon_hires = scale * Ip_profile

  # psi on hires cell grid
  psi_hires = math_utils.cumulative_trapezoid(
      y=dpsi_drhon_hires, x=geo.rho_hires_norm, initial=0.0
  )

  psi_value = jnp.interp(geo.rho_norm, geo.rho_hires_norm, psi_hires)

  # Set the BCs for psi to ensure the correct Ip
  dpsi_drhonorm_edge = psi_calculations.calculate_psi_grad_constraint_from_Ip(
      Ip,
      geo,
  )

  if use_v_loop_lcfs_boundary_condition:
    # For v_loop_lcfs, we will prescribe a rate of change of psi at the LCFS
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


def _init_psi_and_psi_derived(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
    neoclassical_models: neoclassical_models_lib.NeoclassicalModels,
) -> state.CoreProfiles:
  """Initialises psi and currents in core profiles.

  There are three modes of doing this that are supported:
    1. Retrieving psi from the profile conditions.
    2. Retrieving psi from the standard geometry input.
    3. Calculating j according to the nu formula and then
    calculating psi from that. As we are calculating j using a guess for psi,
    this method is iterated to converge to the true psi.

  Args:
    static_runtime_params_slice: Static runtime parameters.
    dynamic_runtime_params_slice: Dynamic runtime parameters.
    geo: Torus geometry.
    core_profiles: Core profiles.
    source_models: All TORAX source/sink functions.
    neoclassical_models: All models for neoclassical calculations.

  Returns:
    Refined core profiles.
  """
  use_v_loop_bc = (
      dynamic_runtime_params_slice.profile_conditions.use_v_loop_lcfs_boundary_condition
  )

  source_profiles = source_profile_builders.build_all_zero_profiles(geo)
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
  # profile and Ip
  if dynamic_runtime_params_slice.profile_conditions.psi is not None:
    # Calculate the dpsi/drho necessary to achieve the given Ip
    dpsi_drhonorm_edge = psi_calculations.calculate_psi_grad_constraint_from_Ip(
        dynamic_runtime_params_slice.profile_conditions.Ip,
        geo,
    )

    # Set the BCs to ensure the correct Ip
    if use_v_loop_bc:
      # Extrapolate the value of psi at the LCFS from the dpsi/drho constraint
      # to achieve the desired Ip
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

  # Case 2: retrieving psi from the standard geometry input.
  elif (
      isinstance(geo, standard_geometry.StandardGeometry)
      and not dynamic_runtime_params_slice.profile_conditions.initial_psi_from_j
  ):
    # psi is already provided from a numerical equilibrium, so no need to
    # first calculate currents. However, non-inductive currents are still
    # calculated and used in current diffusion equation.

    # Calculate the dpsi/drho necessary to achieve the given Ip
    dpsi_drhonorm_edge = psi_calculations.calculate_psi_grad_constraint_from_Ip(
        dynamic_runtime_params_slice.profile_conditions.Ip,
        geo,
    )

    # Use the psi from the equilibrium as the right face constraint
    # This has already been made consistent with the desired Ip
    # by make_ip_consistent
    psi = cell_variable.CellVariable(
        value=geo.psi_from_Ip,  # Use psi from equilibrium
        right_face_grad_constraint=None
        if use_v_loop_bc
        else dpsi_drhonorm_edge,
        right_face_constraint=geo.psi_from_Ip_face[-1]
        if use_v_loop_bc
        else None,
        dr=geo.drho_norm,
    )

  # Case 3: calculating j according to nu formula and psi from j.
  else:
    # First calculate currents without bootstrap.
    external_current = sum(source_profiles.psi.values())
    j_total_hires = _get_j_total_hires(
        bootstrap_profile=source_profiles.bootstrap_current,
        external_current=external_current,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
    )
    psi = update_psi_from_j(
        dynamic_runtime_params_slice.profile_conditions.Ip,
        geo,
        j_total_hires,
        use_v_loop_lcfs_boundary_condition=use_v_loop_bc,
    )
    core_profiles = dataclasses.replace(
        core_profiles,
        psi=psi,
        q_face=psi_calculations.calc_q_face(geo, psi),
        s_face=psi_calculations.calc_s_face(geo, psi),
    )
    # Now calculate currents with bootstrap.
    bootstrap_profile = (
        neoclassical_models.bootstrap_current.calculate_bootstrap_current(
            dynamic_runtime_params_slice, geo, core_profiles
        )
    )
    j_total_hires = _get_j_total_hires(
        bootstrap_profile=bootstrap_profile,
        external_current=external_current,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
    )
    psi = update_psi_from_j(
        dynamic_runtime_params_slice.profile_conditions.Ip,
        geo,
        j_total_hires,
        use_v_loop_lcfs_boundary_condition=use_v_loop_bc,
    )

  j_total, j_total_face, Ip_profile_face = psi_calculations.calc_j_total(
      geo, psi
  )

  core_profiles = dataclasses.replace(
      core_profiles,
      psi=psi,
      q_face=psi_calculations.calc_q_face(geo, psi),
      s_face=psi_calculations.calc_s_face(geo, psi),
      j_total=j_total,
      j_total_face=j_total_face,
      Ip_profile_face=Ip_profile_face,
  )
  bootstrap_profile = (
      neoclassical_models.bootstrap_current.calculate_bootstrap_current(
          dynamic_runtime_params_slice, geo, core_profiles
      )
  )
  conductivity = neoclassical_models.conductivity.calculate_conductivity(
      geo,
      core_profiles,
  )
  source_profiles = dataclasses.replace(
      source_profiles, bootstrap_current=bootstrap_profile
  )
  # psidot calculated here with phibdot=0 in geo, since this is initial
  # conditions and we don't yet have information on geo_t_plus_dt for the
  # phibdot calculation.
  psi_sources = source_profiles.total_psi_sources(geo)
  psidot = psi_calculations.calculate_psidot_from_psi_sources(
      psi_sources=psi_sources,
      sigma=conductivity.sigma,
      resistivity_multiplier=dynamic_runtime_params_slice.numerics.resistivity_multiplier,
      psi=psi,
      geo=geo,
  )

  # psidot boundary condition. If v_loop_lcfs is not prescribed then we set it
  # to the last calculated psidot for the initialisation since we have no
  # other information.
  v_loop_lcfs = (
      dynamic_runtime_params_slice.profile_conditions.v_loop_lcfs
      if use_v_loop_bc
      else psidot[-1]
  )
  psidot = dataclasses.replace(
      core_profiles.psidot,
      value=psidot,
      right_face_constraint=v_loop_lcfs,
      right_face_grad_constraint=None,
  )
  core_profiles = dataclasses.replace(
      core_profiles,
      psidot=psidot,
      sigma=conductivity.sigma,
      sigma_face=conductivity.sigma_face,
  )

  return core_profiles


def _get_j_total_hires(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    bootstrap_profile: bootstrap_current_base.BootstrapCurrent,
    external_current: jax.Array,
) -> jax.Array:
  """Calculates j_total hires."""
  Ip = dynamic_runtime_params_slice.profile_conditions.Ip
  psi_current = external_current + bootstrap_profile.j_bootstrap

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

  # calculate high resolution j_total and Ohmic current profile
  jformula_hires = (
      1 - geo.rho_hires_norm**2
  ) ** dynamic_runtime_params_slice.profile_conditions.current_profile_nu
  denom = _trapz(jformula_hires * geo.spr_hires, geo.rho_hires_norm)
  if dynamic_runtime_params_slice.profile_conditions.initial_j_is_total_current:
    Ctot_hires = Ip / denom
    j_total_hires = jformula_hires * Ctot_hires
  else:
    I_non_inductive = math_utils.area_integration(psi_current, geo)
    Iohm = Ip - I_non_inductive
    Cohm_hires = Iohm / denom
    j_ohmic_hires = jformula_hires * Cohm_hires
    j_total_hires = j_ohmic_hires + external_current_hires + j_bootstrap_hires
  return j_total_hires

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

"""Physics calculations for the initial states/profiles only."""

import dataclasses
import jax
from jax import numpy as jnp
from torax import config_slice
from torax import constants
from torax import fvm
from torax import geometry
from torax import jax_utils
from torax import math_utils
from torax import physics
from torax import state
from torax.geometry import Geometry  # pylint: disable=g-importing-member
from torax.sources import external_current_source
from torax.sources import source_models as source_models_lib
from torax.sources import source_profiles as source_profiles_lib

_trapz = jax.scipy.integrate.trapezoid


def _update_ti(
    dynamic_config_slice: config_slice.DynamicConfigSlice,
    static_config_slice: config_slice.StaticConfigSlice,
    geo: Geometry,
) -> fvm.CellVariable:
  """Update ion temp. Used upon initialization and if temp_ion=False."""
  # pylint: disable=invalid-name
  Ti_bound_left = jax_utils.error_if_not_positive(
      dynamic_config_slice.Ti_bound_left, 'Ti_bound_left'
  )
  Ti_bound_right = jax_utils.error_if_not_positive(
      dynamic_config_slice.Ti_bound_right,
      'Ti_bound_right',
  )
  temp_ion_face = jnp.linspace(
      start=Ti_bound_left,
      stop=Ti_bound_right,
      num=static_config_slice.nr + 1,
  )
  temp_ion = geometry.face_to_cell(temp_ion_face)
  temp_ion = fvm.CellVariable(
      value=temp_ion,
      left_face_grad_constraint=jnp.zeros(()),
      right_face_grad_constraint=None,
      right_face_constraint=Ti_bound_right,
      dr=geo.dr_norm,
  )
  # pylint: enable=invalid-name
  return temp_ion


def _update_te(
    dynamic_config_slice: config_slice.DynamicConfigSlice,
    static_config_slice: config_slice.StaticConfigSlice,
    geo: Geometry,
) -> fvm.CellVariable:
  """Update electron temp. Used upon initialization and if temp_el=False."""
  # pylint: disable=invalid-name
  Te_bound_left = jax_utils.error_if_not_positive(
      dynamic_config_slice.Te_bound_left, 'Te_bound_left'
  )
  Te_bound_right = jax_utils.error_if_not_positive(
      dynamic_config_slice.Te_bound_right,
      'Te_bound_right',
  )
  temp_el_face = jnp.linspace(
      start=Te_bound_left,
      stop=Te_bound_right,
      num=static_config_slice.nr + 1,
  )
  temp_el = geometry.face_to_cell(temp_el_face)
  temp_el = fvm.CellVariable(
      value=temp_el,
      left_face_grad_constraint=jnp.zeros(()),
      right_face_grad_constraint=None,
      right_face_constraint=Te_bound_right,
      dr=geo.dr_norm,
  )
  # pylint: enable=invalid-name
  return temp_el


def _update_dens(
    dynamic_config_slice: config_slice.DynamicConfigSlice,
    static_config_slice: config_slice.StaticConfigSlice,
    geo: Geometry,
) -> tuple[fvm.CellVariable, fvm.CellVariable]:
  """Update particle density. Used upon initialization and if dens_eq=False."""
  # pylint: disable=invalid-name
  nGW = (
      dynamic_config_slice.Ip
      / (jnp.pi * dynamic_config_slice.Rmin**2)
      * 1e20
      / dynamic_config_slice.nref
  )
  nbar_unnorm = jnp.where(
      dynamic_config_slice.nbar_is_fGW,
      dynamic_config_slice.nbar * nGW,
      dynamic_config_slice.nbar,
  )
  # calculate ne_bound_right
  ne_bound_right = jnp.where(
      dynamic_config_slice.ne_bound_right_is_fGW,
      dynamic_config_slice.ne_bound_right * nGW,
      dynamic_config_slice.ne_bound_right,
  )

  # set peaking (limited to linear profile)
  nshape_face = jnp.linspace(
      dynamic_config_slice.npeak, 1, static_config_slice.nr + 1
  )
  nshape = geometry.face_to_cell(nshape_face)

  # find normalization factor such that desired line-averaged n is set
  # Assumes line-averaged central chord on outer midplane
  Rmin_out = geo.Rout_face[-1] - geo.Rout_face[0]
  C = nbar_unnorm / (_trapz(nshape_face, geo.Rout_face) / Rmin_out)
  # pylint: enable=invalid-name

  ne_value = C * nshape
  ne = fvm.CellVariable(
      value=ne_value,
      dr=geo.dr_norm,
      right_face_grad_constraint=None,
      right_face_constraint=jnp.array(ne_bound_right),
  )

  # define ion profile based on (flat) Zeff and single assumed impurity
  # with Zimp. main ion limited to hydrogenic species for now.
  # Assume isotopic balance for DT fusion power. Solve for ni based on:
  # Zeff = (ni + Zimp**2 * nimp)/ne  ;  nimp*Zimp + ni = ne

  dilution_factor = physics.get_main_ion_dilution_factor(
      dynamic_config_slice.Zimp, dynamic_config_slice.Zeff
  )

  ni = fvm.CellVariable(
      value=ne_value * dilution_factor,
      dr=geo.dr_norm,
      right_face_grad_constraint=None,
      right_face_constraint=jnp.array(ne_bound_right * dilution_factor),
  )
  return ne, ni


def _prescribe_currents_no_bootstrap(
    dynamic_config_slice: config_slice.DynamicConfigSlice,
    geo: Geometry,
    source_models: source_models_lib.SourceModels,
) -> state.Currents:
  """Creates the initial Currents without the bootstrap current.

  Args:
    dynamic_config_slice: General configuration parameters at t_initial.
    geo: Geometry of the tokamak.
    source_models: All TORAX source/sink functions.

  Returns:
    currents: Initial Currents
  """

  # Many variables throughout this function are capitalized based on physics
  # notational conventions rather than on Google Python style
  # pylint: disable=invalid-name

  # Calculate splitting of currents depending on config
  Ip = dynamic_config_slice.Ip

  if dynamic_config_slice.use_absolute_jext:
    Iext = dynamic_config_slice.Iext
  else:
    Iext = Ip * dynamic_config_slice.fext
  if dynamic_config_slice.initial_j_is_total_current:
    Iohm = Ip
  else:
    Iohm = Ip - Iext

  # Set zero bootstrap current
  bootstrap_profile = source_profiles_lib.BootstrapCurrentProfile.zero_profile(
      geo
  )

  # calculate "External" current profile (e.g. ECCD)
  # form of external current on face grid
  jext_source = source_models.jext
  jext_face, jext = jext_source.get_value(
      source_type=dynamic_config_slice.sources[jext_source.name].source_type,
      dynamic_config_slice=dynamic_config_slice,
      geo=geo,
  )

  # calculate "Ohmic" current profile
  # form of Ohmic current on face grid
  johmform_face = (1 - geo.r_face_norm**2) ** dynamic_config_slice.nu
  Cohm = Iohm * 1e6 / _trapz(johmform_face * geo.spr_face, geo.r_face)
  johm_face = Cohm * johmform_face  # ohmic current profile on face grid
  johm = geometry.face_to_cell(johm_face)

  jtot_face = johm_face + jext_face
  jtot = geometry.face_to_cell(jtot_face)

  jtot_hires = _get_jtot_hires(
      dynamic_config_slice, geo, bootstrap_profile, Iohm, jext_source
  )

  currents = state.Currents(
      jtot=jtot,
      jtot_face=jtot_face,
      jtot_hires=jtot_hires,
      johm=johm,
      johm_face=johm_face,
      jext=jext,
      jext_face=jext_face,
      j_bootstrap=bootstrap_profile.j_bootstrap,
      j_bootstrap_face=bootstrap_profile.j_bootstrap_face,
      I_bootstrap=bootstrap_profile.I_bootstrap,
      sigma=bootstrap_profile.sigma,
  )

  return currents


def _prescribe_currents_with_bootstrap(
    dynamic_config_slice: config_slice.DynamicConfigSlice,
    geo: Geometry,
    source_models: source_models_lib.SourceModels,
    temp_ion: fvm.CellVariable,
    temp_el: fvm.CellVariable,
    ne: fvm.CellVariable,
    ni: fvm.CellVariable,
    jtot_face: jax.Array,
    psi: fvm.CellVariable,
) -> state.Currents:
  """Creates the initial Currents.

  Args:
    dynamic_config_slice: General configuration parameters at t_initial.
    geo: Geometry of the tokamak.
    source_models: All TORAX source/sink functions. If not provided, uses the
      default sources.
    temp_ion: Ion temperature.
    temp_el: Electron temperature.
    ne: Electron density.
    ni: Main ion density.
    jtot_face: Total current density on face grid.
    psi: Poloidal flux.

  Returns:
    currents: Plasma currents
  """

  # Many variables throughout this function are capitalized based on physics
  # notational conventions rather than on Google Python style
  # pylint: disable=invalid-name
  Ip = dynamic_config_slice.Ip

  bootstrap_profile = source_models.j_bootstrap.get_value(
      dynamic_config_slice=dynamic_config_slice,
      geo=geo,
      temp_ion=temp_ion,
      temp_el=temp_el,
      ne=ne,
      ni=ni,
      jtot_face=jtot_face,
      psi=psi,
  )
  f_bootstrap = bootstrap_profile.I_bootstrap / (Ip * 1e6)

  # Calculate splitting of currents depending on config
  if dynamic_config_slice.use_absolute_jext:
    Iext = dynamic_config_slice.Iext
  else:
    Iext = Ip * dynamic_config_slice.fext
  if dynamic_config_slice.initial_j_is_total_current:
    Iohm = Ip
  else:
    Iohm = Ip - Iext - f_bootstrap * Ip

  # calculate "Ohmic" current profile
  # form of Ohmic current on face grid
  johmform_face = (1 - geo.r_face_norm**2) ** dynamic_config_slice.nu
  Cohm = Iohm * 1e6 / _trapz(johmform_face * geo.spr_face, geo.r_face)
  johm_face = Cohm * johmform_face  # ohmic current profile on face grid
  johm = geometry.face_to_cell(johm_face)

  # calculate "External" current profile (e.g. ECCD)
  # form of external current on face grid
  jext_source = source_models.jext
  jext_face, jext = jext_source.get_value(
      source_type=dynamic_config_slice.sources[jext_source.name].source_type,
      dynamic_config_slice=dynamic_config_slice,
      geo=geo,
  )

  jtot_face = johm_face + jext_face + bootstrap_profile.j_bootstrap_face
  jtot = geometry.face_to_cell(jtot_face)

  jtot_hires = _get_jtot_hires(
      dynamic_config_slice, geo, bootstrap_profile, Iohm, jext_source
  )

  currents = state.Currents(
      jtot=jtot,
      jtot_face=jtot_face,
      jtot_hires=jtot_hires,
      johm=johm,
      johm_face=johm_face,
      jext=jext,
      jext_face=jext_face,
      j_bootstrap=bootstrap_profile.j_bootstrap,
      j_bootstrap_face=bootstrap_profile.j_bootstrap_face,
      I_bootstrap=bootstrap_profile.I_bootstrap,
      sigma=bootstrap_profile.sigma,
  )

  return currents


def _calculate_currents_from_psi(
    dynamic_config_slice: config_slice.DynamicConfigSlice,
    geo: Geometry,
    source_models: source_models_lib.SourceModels,
    temp_ion: fvm.CellVariable,
    temp_el: fvm.CellVariable,
    ne: fvm.CellVariable,
    ni: fvm.CellVariable,
    psi: fvm.CellVariable,
) -> state.Currents:
  """Creates the initial Currents using psi to calculate jtot.

  Args:
    dynamic_config_slice: General configuration parameters at t_initial.
    geo: Geometry of the tokamak.
    source_models: All TORAX source/sink functions. If not provided, uses the
      default sources.
    temp_ion: Ion temperature.
    temp_el: Electron temperature.
    ne: Electron density.
    ni: Main ion density.
    psi: Poloidal flux.

  Returns:
    currents: Plasma currents
  """

  # Many variables throughout this function are capitalized based on physics
  # notational conventions rather than on Google Python style
  # pylint: disable=invalid-name
  Ip = dynamic_config_slice.Ip

  jtot, jtot_face = physics.calc_jtot_from_psi(
      geo,
      psi,
      dynamic_config_slice.Rmaj,
  )

  bootstrap_profile = source_models.j_bootstrap.get_value(
      dynamic_config_slice=dynamic_config_slice,
      geo=geo,
      temp_ion=temp_ion,
      temp_el=temp_el,
      ne=ne,
      ni=ni,
      jtot_face=jtot_face,
      psi=psi,
  )

  # Calculate splitting of currents depending on config
  if dynamic_config_slice.use_absolute_jext:
    Iext = dynamic_config_slice.Iext
  else:
    Iext = Ip * dynamic_config_slice.fext

  Iohm = Ip - Iext - bootstrap_profile.I_bootstrap

  # calculate "External" current profile (e.g. ECCD)
  # form of external current on face grid
  jext_source = source_models.jext
  jext_face, jext = jext_source.get_value(
      source_type=dynamic_config_slice.sources[jext_source.name].source_type,
      dynamic_config_slice=dynamic_config_slice,
      geo=geo,
  )

  johm = jtot - jext - bootstrap_profile.j_bootstrap
  johm_face = jtot_face - jext_face - bootstrap_profile.j_bootstrap_face

  jtot_hires = _get_jtot_hires(
      dynamic_config_slice, geo, bootstrap_profile, Iohm, jext_source
  )

  currents = state.Currents(
      jtot=jtot,
      jtot_face=jtot_face,
      jtot_hires=jtot_hires,
      johm=johm,
      johm_face=johm_face,
      jext=jext,
      jext_face=jext_face,
      j_bootstrap=bootstrap_profile.j_bootstrap,
      j_bootstrap_face=bootstrap_profile.j_bootstrap_face,
      I_bootstrap=bootstrap_profile.I_bootstrap,
      sigma=bootstrap_profile.sigma,
  )

  return currents
  # pylint: enable=invalid-name


def _update_psi_from_j(
    dynamic_config_slice: config_slice.DynamicConfigSlice,
    geo: Geometry,
    currents: state.Currents,
) -> fvm.CellVariable:
  """Calculates poloidal flux (psi) consistent with plasma current.

  For increased accuracy of psi, a hi-res grid is used, due to the double
    integration.

  Args:
    dynamic_config_slice: Dynamic configuration parameters.
    geo: Torus geometry.
    currents: Currents structure including high resolution version of jtot.

  Returns:
    psi: Poloidal flux cell variable.
  """

  psi_constraint = (
      dynamic_config_slice.Ip
      * 1e6
      * constants.CONSTANTS.mu0
      / geo.G2_face[-1]
      * geo.rmax
  )

  y = currents.jtot_hires * geo.vpr_hires
  assert y.ndim == 1
  assert geo.r_hires.ndim == 1
  integrated = math_utils.cumulative_trapezoid(
      geo.r_hires, y, initial=jnp.zeros(())
  )
  scale = jnp.concatenate((
      jnp.zeros((1,)),
      constants.CONSTANTS.mu0
      / (2 * jnp.pi * dynamic_config_slice.Rmaj * geo.G2_hires[1:]),
  ))
  # dpsi_dr on the cell grid
  dpsi_dr_hires = scale * integrated

  # psi on cell grid
  psi_hires = math_utils.cumulative_trapezoid(
      geo.r_hires,
      dpsi_dr_hires,
      initial=jnp.zeros(()),
  )

  psi_value = jnp.interp(geo.r, geo.r_hires, psi_hires)

  psi = fvm.CellVariable(
      value=psi_value,
      dr=geo.dr_norm,
      right_face_grad_constraint=psi_constraint,
  )

  return psi


def initial_core_profiles(
    dynamic_config_slice: config_slice.DynamicConfigSlice,
    static_config_slice: config_slice.StaticConfigSlice,
    geo: Geometry,
    source_models: source_models_lib.SourceModels | None = None,
) -> state.CoreProfiles:
  """Calculates the initial core profiles.

  Args:
    dynamic_config_slice: Dynamic configuration parameters at t=t_initial.
    static_config_slice: Static simulation configuration parameters.
    geo: Torus geometry.
    source_models: All models for TORAX sources/sinks. If not provided, uses the
      default source_models.

  Returns:
    Initial core profiles.
  """
  # pylint: disable=invalid-name

  source_models = (
      source_models_lib.SourceModels()
      if source_models is None
      else source_models
  )

  # To set initial values and compute the boundary conditions, we need to handle
  # potentially time-varying inputs from the users.
  # The default time in build_dynamic_config_slice is t_initial
  temp_ion = _update_ti(dynamic_config_slice, static_config_slice, geo)
  temp_el = _update_te(dynamic_config_slice, static_config_slice, geo)
  ne, ni = _update_dens(dynamic_config_slice, static_config_slice, geo)

  # set up initial psi profile based on current profile
  if isinstance(geo, geometry.CircularGeometry):
    # set up initial current profile without bootstrap current, to get
    # q-profile approximation (needed for bootstrap)
    currents_no_bootstrap = _prescribe_currents_no_bootstrap(
        dynamic_config_slice=dynamic_config_slice,
        geo=geo,
        source_models=source_models,
    )

    psi_no_bootstrap = _update_psi_from_j(
        dynamic_config_slice,
        geo,
        currents_no_bootstrap,
    )

    # second iteration, with bootstrap current
    currents = _prescribe_currents_with_bootstrap(
        dynamic_config_slice=dynamic_config_slice,
        geo=geo,
        source_models=source_models,
        temp_ion=temp_ion,
        temp_el=temp_el,
        ne=ne,
        ni=ni,
        jtot_face=currents_no_bootstrap.jtot_face,
        psi=psi_no_bootstrap,
    )

    psi = _update_psi_from_j(
        dynamic_config_slice,
        geo,
        currents,
    )

    q_face, _ = physics.calc_q_from_jtot_psi(
        geo=geo,
        jtot_face=currents.jtot_face,
        psi=psi,
        Rmaj=dynamic_config_slice.Rmaj,
        q_correction_factor=dynamic_config_slice.q_correction_factor,
    )
    s_face = physics.calc_s_from_psi(geo, psi)

  elif isinstance(geo, geometry.CHEASEGeometry):
    # psi is already provided from the CHEASE equilibrium, so no need to first
    # calculate currents. However, non-inductive currents are still calculated
    # and used in current diffusion equation.
    psi_constraint = (
        dynamic_config_slice.Ip
        * 1e6
        * constants.CONSTANTS.mu0
        / geo.G2_face[-1]
        * geo.rmax
    )
    psi = fvm.CellVariable(
        value=geo.psi_from_chease_Ip,
        right_face_grad_constraint=psi_constraint,
        dr=geo.dr_norm,
    )
    q_face, _ = physics.calc_q_from_jtot_psi(
        geo=geo,
        jtot_face=geo.jtot_face,
        psi=psi,
        Rmaj=dynamic_config_slice.Rmaj,
        q_correction_factor=dynamic_config_slice.q_correction_factor,
    )
    s_face = physics.calc_s_from_psi(geo, psi)

    # calculation external currents
    currents = _calculate_currents_from_psi(
        dynamic_config_slice=dynamic_config_slice,
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

  # the psidot calculation needs core profiles. So psidot first initialized
  # with zeros.
  psidot = fvm.CellVariable(
      value=jnp.zeros_like(psi.value),
      dr=geo.dr_norm,
  )

  core_profiles = state.CoreProfiles(
      temp_ion=temp_ion,
      temp_el=temp_el,
      ne=ne,
      ni=ni,
      psi=psi,
      psidot=psidot,
      currents=currents,
      q_face=q_face,
      s_face=s_face,
  )

  psidot = dataclasses.replace(
      psidot,
      value=source_models_lib.calc_psidot(
          source_models, dynamic_config_slice, geo, core_profiles
      ),
  )

  core_profiles = dataclasses.replace(core_profiles, psidot=psidot)

  core_profiles = physics.update_jtot_q_face_s_face(
      geo=geo,
      core_profiles=core_profiles,
      Rmaj=dynamic_config_slice.Rmaj,
      q_correction_factor=dynamic_config_slice.q_correction_factor,
  )

  # pylint: enable=invalid-name
  return core_profiles


# pylint: disable=invalid-name
def _get_jtot_hires(
    dynamic_config_slice: config_slice.DynamicConfigSlice,
    geo: Geometry,
    bootstrap_profile: source_profiles_lib.BootstrapCurrentProfile,
    Iohm: jax.Array | float,
    jext_source: external_current_source.ExternalCurrentSource,
) -> jax.Array:
  """Calculates jtot hires."""
  j_bootstrap_hires = jnp.interp(
      geo.r_hires, geo.r_face, bootstrap_profile.j_bootstrap_face
  )
  # calculate high resolution "Ohmic" current profile
  # form of Ohmic current on cell grid
  johmform_hires = (1 - geo.r_hires_norm**2) ** dynamic_config_slice.nu
  denom = _trapz(johmform_hires * geo.spr_hires, geo.r_hires)
  Cohm_hires = Iohm * 1e6 / denom

  # Ohmic current profile on cell grid
  johm_hires = Cohm_hires * johmform_hires

  # calculate "External" current profile (e.g. ECCD) on cell grid.
  jext_hires = jext_source.jext_hires(
      source_type=dynamic_config_slice.sources[jext_source.name].source_type,
      dynamic_config_slice=dynamic_config_slice,
      geo=geo,
  )
  # jtot on the various grids
  jtot_hires = johm_hires + jext_hires + j_bootstrap_hires
  return jtot_hires


# pylint: enable=invalid-name

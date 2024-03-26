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
from typing import Optional
import jax
from jax import numpy as jnp
from torax import config as config_lib
from torax import config_slice
from torax import constants
from torax import fvm
from torax import geometry
from torax import jax_utils
from torax import math_utils
from torax import physics
from torax import state
from torax.geometry import Geometry  # pylint: disable=g-importing-member
from torax.sources import bootstrap_current_source
from torax.sources import external_current_source
from torax.sources import source_profiles as source_profiles_lib

_trapz = jax.scipy.integrate.trapezoid


def initial_core_profiles(
    config: config_lib.Config,
    geo: Geometry,
    sources: source_profiles_lib.Sources | None = None,
) -> state.CoreProfiles:
  """Calculates the initial core profiles.

  Args:
    config: General configuration parameters.
    geo: Torus geometry.
    sources: All TORAX sources/sinks. If not provided, uses the default sources.

  Returns:
    Initial core profiles.
  """
  # pylint: disable=invalid-name

  sources = source_profiles_lib.Sources() if sources is None else sources

  # To set initial values and compute the boundary conditions, we need to handle
  # potentially time-varying inputs from the users.
  # The default time in build_dynamic_config_slice is t_initial
  dynamic_config_slice = config_slice.build_dynamic_config_slice(config)

  # plasma current at initial condition (may also be used for initial density)
  Ip = dynamic_config_slice.Ip

  # Set temperature initial condition. T_bound refers to face grid boundaries
  Ti_bound_left = jax_utils.error_if_not_positive(
      config.Ti_bound_left, 'Ti_bound_left'
  )
  Ti_bound_right = jax_utils.error_if_not_positive(
      dynamic_config_slice.Ti_bound_right,
      'Ti_bound_right',
  )
  initial_temp_ion_face = jnp.linspace(
      start=Ti_bound_left,
      stop=Ti_bound_right,
      num=config.nr + 1,
  )
  initial_temp_ion = geometry.face_to_cell(initial_temp_ion_face)
  temp_ion = fvm.CellVariable(
      value=initial_temp_ion,
      left_face_grad_constraint=jnp.zeros(()),
      right_face_grad_constraint=None,
      right_face_constraint=Ti_bound_right,
      dr=geo.dr_norm,
  )
  Te_bound_left = jax_utils.error_if_not_positive(
      config.Te_bound_left, 'Te_bound_left'
  )
  Te_bound_right = jax_utils.error_if_not_positive(
      dynamic_config_slice.Te_bound_right,
      'Te_bound_right',
  )
  initial_temp_el_face = jnp.linspace(
      start=Te_bound_left,
      stop=Te_bound_right,
      num=config.nr + 1,
  )
  initial_temp_el = geometry.face_to_cell(initial_temp_el_face)
  temp_el = fvm.CellVariable(
      value=initial_temp_el,
      left_face_grad_constraint=jnp.zeros(()),
      right_face_grad_constraint=None,
      right_face_constraint=Te_bound_right,
      dr=geo.dr_norm,
  )

  # set up density profile
  if config.set_fGW:
    # set nbar to desired fraction of Greenwald density
    # Use normalized nbar
    nGW = Ip / (jnp.pi * dynamic_config_slice.Rmin**2) * 1e20 / config.nref
    nbar = config.fGW * nGW  # normalized nbar
  else:
    nbar = config.nbar

  # set peaking (limited to linear profile)
  nshape_face = jnp.linspace(config.npeak, 1, config.nr + 1)
  nshape = geometry.face_to_cell(nshape_face)

  # find normalization factor such that desired line-averaged n is set
  # Assumes line-averaged central chord on outer midplane
  Rmin_out = geo.Rout_face[-1] - geo.Rout_face[0]
  C = nbar / (_trapz(nshape_face, geo.Rout_face) / Rmin_out)

  # set flat electron density profile
  ne_value = C * nshape
  ne_bound_right = dynamic_config_slice.ne_bound_right
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

  # set up initial psi profile (from current profile without bootstrap
  # current)
  if isinstance(geo, geometry.CircularGeometry):
    # set up initial current profile without bootstrap current, to get
    # q-profile approximation (needed for bootstrap)
    currents_no_bootstrap = initial_currents(
        dynamic_config_slice=dynamic_config_slice,
        geo=geo,
        bootstrap=False,
        sources=sources,
    )

    psi_constraint = (
        Ip * 1e6 * constants.CONSTANTS.mu0 / geo.G2_face[-1] * geo.rmax
    )
    psi_no_bootstrap = fvm.CellVariable(
        value=initial_psi(
            dynamic_config_slice,
            geo,
            currents_no_bootstrap.jtot_hires,
        ),
        dr=geo.dr_norm,
        right_face_grad_constraint=psi_constraint,
    )

    # second iteration, with bootstrap current
    currents = initial_currents(
        dynamic_config_slice=dynamic_config_slice,
        geo=geo,
        bootstrap=True,
        sources=sources,
        temp_ion=temp_ion,
        temp_el=temp_el,
        ne=ne,
        ni=ni,
        jtot_face=currents_no_bootstrap.jtot_face,
        psi=psi_no_bootstrap,
    )
    psi_value = initial_psi(config, geo, currents.jtot_hires)

    psi = fvm.CellVariable(
        value=psi_value,
        right_face_grad_constraint=psi_constraint,
        dr=geo.dr_norm,
    )

    q_face, _ = physics.calc_q_from_jtot_psi(
        geo=geo,
        jtot_face=currents.jtot_face,
        psi=psi,
        Rmaj=config.Rmaj,
        q_correction_factor=config.q_correction_factor,
    )
    s_face = physics.calc_s_from_psi(geo, psi)

  elif isinstance(geo, geometry.CHEASEGeometry):
    # psi is already provided from the CHEASE equilibrium, so no need to first
    # calculate currents. However, non-inductive currents are still calculated
    # and used in current diffusion equation.
    psi_constraint = (
        Ip * 1e6 * constants.CONSTANTS.mu0 / geo.G2_face[-1] * geo.rmax
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
        Rmaj=config.Rmaj,
        q_correction_factor=config.q_correction_factor,
    )
    s_face = physics.calc_s_from_psi(geo, psi)

    # calculation external currents
    currents = initial_currents(
        dynamic_config_slice=dynamic_config_slice,
        geo=geo,
        bootstrap=True,
        sources=sources,
        temp_ion=temp_ion,
        temp_el=temp_el,
        ne=ne,
        ni=ni,
        jtot_face=geo.jtot_face,
        psi=psi,
        hires=False,
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
      value=source_profiles_lib.calc_psidot(
          sources, dynamic_config_slice, geo, core_profiles
      ),
  )

  core_profiles = dataclasses.replace(core_profiles, psidot=psidot)

  core_profiles = physics.update_jtot_q_face_s_face(
      geo=geo,
      core_profiles=core_profiles,
      Rmaj=config.Rmaj,
      q_correction_factor=config.q_correction_factor,
  )

  # pylint: enable=invalid-name
  return core_profiles


# pylint: disable=invalid-name
def _get_jtot_hires(
    dynamic_config_slice: config_slice.DynamicConfigSlice,
    geo: Geometry,
    bootstrap_profile: bootstrap_current_source.BootstrapCurrentProfile,
    Iohm: jax.Array,
    jext_source: external_current_source.ExternalCurrentSource,
) -> jax.Array:
  """Calculates jtot hires."""
  assert isinstance(geo, geometry.CircularGeometry)
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
  # TODO(b/323504363): Replace ad-hoc circular equilibrium
  # with more accurate analytical equilibrium
  jext_hires = jext_source.jext_hires(
      source_type=dynamic_config_slice.sources[jext_source.name].source_type,
      dynamic_config_slice=dynamic_config_slice,
      geo=geo,
  )
  # jtot on the various grids
  jtot_hires = johm_hires + jext_hires + j_bootstrap_hires
  return jtot_hires


# pylint: enable=invalid-name


# TODO(b/323504363): Clean this up so it is less hacky and with less branches.
# Potentially split this up into several methods.
def initial_currents(
    dynamic_config_slice: config_slice.DynamicConfigSlice,
    geo: Geometry,
    bootstrap: bool,
    sources: source_profiles_lib.Sources,
    temp_ion: Optional[fvm.CellVariable] = None,
    temp_el: Optional[fvm.CellVariable] = None,
    ne: Optional[fvm.CellVariable] = None,
    ni: Optional[fvm.CellVariable] = None,
    jtot_face: Optional[jax.Array] = None,
    psi: Optional[fvm.CellVariable] = None,
    hires: bool = True,
) -> state.Currents:
  """Creates the initial Currents.

  Args:
    dynamic_config_slice: General configuration parameters at t_initial.
    geo: Geometry of the tokamak.
    bootstrap: Whether to include bootstrap current.
    sources: All TORAX sources/sinks used to compute the initial currents.
    temp_ion: Ion temperature. Needed only `if bootstrap`. We don't just use a
      `State`, because `initial_currents` must be called to make the Currents
      for the first `State`.
    temp_el: Electron temperature. Needed only `if bootstrap`.
    ne: Electron density. Needed only `if bootstrap`.
    ni: Main ion density. Needed only `if bootstrap`.
    jtot_face: Total current density on face grid. Needed only `if bootstrap`.
    psi: Poloidal flux. Needed only `if bootstrap`.
    hires: If True, uses the hires parameters of the geometry. Only use for
      circular geometries.

  Returns:
    currents: Initial Currents
  """

  # Many variables throughout this function are capitalized based on physics
  # notational conventions rather than on Google Python style
  # pylint: disable=invalid-name
  Ip = dynamic_config_slice.Ip

  if bootstrap:
    if any([x is None for x in [temp_ion, temp_el, ne, jtot_face, psi]]):
      raise ValueError('All optional arguments must be specified for bootstrap')
    bootstrap_profile = sources.j_bootstrap.get_value(
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
  else:
    # A scalar zero would work here, but we want to make sure out Currents
    # pytree always has exactly the same types and shapes
    # sigma=0 is unphysical, but will be overwritten in subsequent call.
    # TODO(b/323504363): cleanup through separating first call with circular
    # geometry to dedicated method
    bootstrap_profile = bootstrap_current_source.BootstrapCurrentProfile(
        sigma=jnp.zeros_like(geo.r),
        j_bootstrap=jnp.zeros_like(geo.r),
        j_bootstrap_face=jnp.zeros_like(geo.r_face),
        I_bootstrap=jnp.zeros(()),
    )
    f_bootstrap = 0.0

  if dynamic_config_slice.use_absolute_jext:
    Iohm = (
        Ip * (1 - f_bootstrap) - dynamic_config_slice.Iext
    )  # total Ohmic current MA
  else:
    Iohm = Ip * (
        1 - dynamic_config_slice.fext - f_bootstrap
    )  # total Ohmic current MA

  # calculate "Ohmic" current profile
  # form of Ohmic current on face grid
  johmform_face = (1 - geo.r_face_norm**2) ** dynamic_config_slice.nu
  Cohm = Iohm * 1e6 / _trapz(johmform_face * geo.spr_face, geo.r_face)
  johm_face = Cohm * johmform_face  # ohmic current profile on face grid
  johm = geometry.face_to_cell(johm_face)

  # calculate "External" current profile (e.g. ECCD)
  # form of external current on face grid

  # form of external current on face grid
  jext_source = sources.jext
  jext_face, jext = jext_source.get_value(
      source_type=dynamic_config_slice.sources[jext_source.name].source_type,
      dynamic_config_slice=dynamic_config_slice,
      geo=geo,
  )

  jtot_face = johm_face + jext_face + bootstrap_profile.j_bootstrap_face
  jtot = geometry.face_to_cell(jtot_face)

  if hires:
    jtot_hires = _get_jtot_hires(
        dynamic_config_slice, geo, bootstrap_profile, Iohm, jext_source
    )
  else:
    jtot_hires = jnp.zeros(0)

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


def initial_psi(
    dynamic_config_slice: config_slice.DynamicConfigSlice,
    geo: Geometry,
    jtot_hires: jax.Array,
) -> jax.Array:
  """Calculates initial value of psi.

  For increased accuracy of psi, hi-res grid is used for the ad-hoc
  circular geometry model (due to double integration).

  Args:
    dynamic_config_slice: Configuration parameters for t_initial.
    geo: Torus geometry.
    jtot_hires: High resolution version of jtot.

  Returns:
    psi: Poloidal flux
  """
  assert isinstance(geo, geometry.CircularGeometry)
  y = jtot_hires * geo.vpr_hires
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

  psi = jnp.interp(geo.r, geo.r_hires, psi_hires)

  return psi

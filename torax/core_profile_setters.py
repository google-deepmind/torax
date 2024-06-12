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
from torax import fvm
from torax import geometry
from torax import jax_utils
from torax import math_utils
from torax import physics
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import Geometry  # pylint: disable=g-importing-member
from torax.sources import external_current_source
from torax.sources import source_models as source_models_lib
from torax.sources import source_profiles as source_profiles_lib

_trapz = jax.scipy.integrate.trapezoid


def _updated_ti(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: Geometry,
) -> fvm.CellVariable:
  """Updated ion temp. Used upon initialization and if temp_ion=False."""
  # pylint: disable=invalid-name
  Ti_bound_left = jax_utils.error_if_not_positive(
      dynamic_runtime_params_slice.profile_conditions.Ti_bound_left,
      'Ti_bound_left',
  )
  Ti_bound_right = jax_utils.error_if_not_positive(
      dynamic_runtime_params_slice.profile_conditions.Ti_bound_right,
      'Ti_bound_right',
  )
  if dynamic_runtime_params_slice.profile_conditions.Ti is not None:
    temp_ion = dynamic_runtime_params_slice.profile_conditions.Ti
  else:
    temp_ion_face = jnp.linspace(
        start=Ti_bound_left,
        stop=Ti_bound_right,
        num=geo.mesh.nx + 1,
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


def _updated_te(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: Geometry,
) -> fvm.CellVariable:
  """Updated electron temp. Used upon initialization and if temp_el=False."""
  # pylint: disable=invalid-name
  Te_bound_left = jax_utils.error_if_not_positive(
      dynamic_runtime_params_slice.profile_conditions.Te_bound_left,
      'Te_bound_left',
  )
  Te_bound_right = jax_utils.error_if_not_positive(
      dynamic_runtime_params_slice.profile_conditions.Te_bound_right,
      'Te_bound_right',
  )
  if dynamic_runtime_params_slice.profile_conditions.Te is not None:
    temp_el = dynamic_runtime_params_slice.profile_conditions.Te
  else:
    temp_el_face = jnp.linspace(
        start=Te_bound_left,
        stop=Te_bound_right,
        num=geo.mesh.nx + 1,
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


def _updated_dens(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: Geometry,
) -> tuple[fvm.CellVariable, fvm.CellVariable]:
  """Updated particle density. Used upon initialization and if dens_eq=False."""
  # pylint: disable=invalid-name
  nGW = (
      dynamic_runtime_params_slice.profile_conditions.Ip
      / (jnp.pi * geo.Rmin**2)
      * 1e20
      / dynamic_runtime_params_slice.numerics.nref
  )
  # calculate ne_bound_right
  ne_bound_right = jnp.where(
      dynamic_runtime_params_slice.profile_conditions.ne_bound_right_is_fGW,
      dynamic_runtime_params_slice.profile_conditions.ne_bound_right * nGW,
      dynamic_runtime_params_slice.profile_conditions.ne_bound_right,
  )

  if dynamic_runtime_params_slice.profile_conditions.ne is not None:
    ne_value = dynamic_runtime_params_slice.profile_conditions.ne
  else:
    nbar_unnorm = jnp.where(
        dynamic_runtime_params_slice.profile_conditions.nbar_is_fGW,
        dynamic_runtime_params_slice.profile_conditions.nbar * nGW,
        dynamic_runtime_params_slice.profile_conditions.nbar,
    )
    # set peaking (limited to linear profile)
    nshape_face = jnp.linspace(
        dynamic_runtime_params_slice.profile_conditions.npeak,
        1,
        geo.mesh.nx + 1,
    )
    nshape = geometry.face_to_cell(nshape_face)

    # find normalization factor such that desired line-averaged n is set
    # Assumes line-averaged central chord on outer midplane
    # pylint: disable=invalid-name
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
      dynamic_runtime_params_slice.plasma_composition.Zimp,
      dynamic_runtime_params_slice.plasma_composition.Zeff,
  )

  if dynamic_runtime_params_slice.profile_conditions.ni is not None:
    ni_value = dynamic_runtime_params_slice.profile_conditions.ni
  else:
    ni_value = ne_value * dilution_factor

  ni = fvm.CellVariable(
      value=ni_value,
      dr=geo.dr_norm,
      right_face_grad_constraint=None,
      right_face_constraint=jnp.array(ne_bound_right * dilution_factor),
  )
  return ne, ni


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

  dynamic_jext_params = _get_jext_params(
      dynamic_runtime_params_slice, source_models
  )
  if dynamic_jext_params.use_absolute_jext:
    Iext = dynamic_jext_params.Iext
  else:
    Iext = Ip * dynamic_jext_params.fext
  # Total Ohmic current
  Iohm = Ip - Iext

  # Set zero bootstrap current
  bootstrap_profile = source_profiles_lib.BootstrapCurrentProfile.zero_profile(
      geo
  )

  # calculate "External" current profile (e.g. ECCD)
  # form of external current on face grid
  jext_face, jext = source_models.jext.get_value(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      dynamic_source_runtime_params=dynamic_jext_params,
      geo=geo,
  )

  # construct prescribed current formula on grid.
  jformula_face = (
      1 - geo.r_face_norm**2
  ) ** dynamic_runtime_params_slice.profile_conditions.nu
  # calculate total and Ohmic current profiles
  denom = _trapz(jformula_face * geo.spr_face, geo.r_face)
  if dynamic_runtime_params_slice.profile_conditions.initial_j_is_total_current:
    Ctot = Ip * 1e6 / denom
    jtot_face = jformula_face * Ctot
    johm_face = jtot_face - jext_face
  else:
    Cohm = Iohm * 1e6 / denom
    johm_face = jformula_face * Cohm
    jtot_face = johm_face + jext_face

  jtot = geometry.face_to_cell(jtot_face)
  johm = geometry.face_to_cell(johm_face)

  jtot_hires = _get_jtot_hires(
      dynamic_runtime_params_slice,
      dynamic_jext_params,
      geo,
      bootstrap_profile,
      Iohm,
      source_models.jext,
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
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: Geometry,
    temp_ion: fvm.CellVariable,
    temp_el: fvm.CellVariable,
    ne: fvm.CellVariable,
    ni: fvm.CellVariable,
    jtot_face: jax.Array,
    psi: fvm.CellVariable,
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
    jtot_face: Total current density on face grid.
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
      jtot_face=jtot_face,
      psi=psi,
  )
  f_bootstrap = bootstrap_profile.I_bootstrap / (Ip * 1e6)

  # Calculate splitting of currents depending on input runtime params
  dynamic_jext_params = _get_jext_params(
      dynamic_runtime_params_slice, source_models
  )
  if dynamic_jext_params.use_absolute_jext:
    Iext = dynamic_jext_params.Iext
  else:
    Iext = Ip * dynamic_jext_params.fext
  Iohm = Ip - Iext - f_bootstrap * Ip

  # calculate "External" current profile (e.g. ECCD)
  # form of external current on face grid
  jext_face, jext = source_models.jext.get_value(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      dynamic_source_runtime_params=dynamic_jext_params,
      geo=geo,
  )

  # construct prescribed current formula on grid.
  jformula_face = (
      1 - geo.r_face_norm**2
  ) ** dynamic_runtime_params_slice.profile_conditions.nu
  denom = _trapz(jformula_face * geo.spr_face, geo.r_face)
  # calculate total and Ohmic current profiles
  if dynamic_runtime_params_slice.profile_conditions.initial_j_is_total_current:
    Ctot = Ip * 1e6 / denom
    jtot_face = jformula_face * Ctot
    johm_face = jtot_face - jext_face - bootstrap_profile.j_bootstrap_face
  else:
    Cohm = Iohm * 1e6 / denom
    johm_face = jformula_face * Cohm
    jtot_face = johm_face + jext_face + bootstrap_profile.j_bootstrap_face

  jtot = geometry.face_to_cell(jtot_face)
  johm = geometry.face_to_cell(johm_face)

  jtot_hires = _get_jtot_hires(
      dynamic_runtime_params_slice,
      dynamic_jext_params,
      geo,
      bootstrap_profile,
      Iohm,
      source_models.jext,
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
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: Geometry,
    temp_ion: fvm.CellVariable,
    temp_el: fvm.CellVariable,
    ne: fvm.CellVariable,
    ni: fvm.CellVariable,
    psi: fvm.CellVariable,
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
  Ip = dynamic_runtime_params_slice.profile_conditions.Ip

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
      jtot_face=jtot_face,
      psi=psi,
  )

  # Calculate splitting of currents depending on input runtime params.
  dynamic_jext_params = _get_jext_params(
      dynamic_runtime_params_slice, source_models
  )
  if dynamic_jext_params.use_absolute_jext:
    Iext = dynamic_jext_params.Iext
  else:
    Iext = Ip * dynamic_jext_params.fext

  Iohm = Ip - Iext - bootstrap_profile.I_bootstrap

  # calculate "External" current profile (e.g. ECCD)
  # form of external current on face grid
  jext_face, jext = source_models.jext.get_value(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      dynamic_source_runtime_params=dynamic_jext_params,
      geo=geo,
  )

  johm = jtot - jext - bootstrap_profile.j_bootstrap
  johm_face = jtot_face - jext_face - bootstrap_profile.j_bootstrap_face

  # TODO(b/336995925): TORAX currently only uses the external current source,
  # jext, when computing the jtot initial currents from psi. Really, though, we
  # should be summing over all sources that can contribute current i.e. ECCD,
  # ICRH, NBI, LHCD.
  jtot_hires = _get_jtot_hires(
      dynamic_runtime_params_slice,
      dynamic_jext_params,
      geo,
      bootstrap_profile,
      Iohm,
      source_models.jext,
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
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: Geometry,
    currents: state.Currents,
) -> fvm.CellVariable:
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

  psi_constraint = (
      dynamic_runtime_params_slice.profile_conditions.Ip
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
      constants.CONSTANTS.mu0 / (2 * jnp.pi * geo.Rmaj * geo.G2_hires[1:]),
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
  temp_ion = _updated_ti(
      dynamic_runtime_params_slice, geo
  )
  temp_el = _updated_te(
      dynamic_runtime_params_slice, geo
  )
  ne, ni = _updated_dens(
      dynamic_runtime_params_slice, geo
  )

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
        jtot_face=currents_no_bootstrap.jtot_face,
        psi=psi_no_bootstrap,
        source_models=source_models,
    )

    psi = _update_psi_from_j(
        dynamic_runtime_params_slice,
        geo,
        currents,
    )

    q_face, _ = physics.calc_q_from_jtot_psi(
        geo=geo,
        psi=psi,
        jtot_face=currents.jtot_face,
        q_correction_factor=dynamic_runtime_params_slice.numerics.q_correction_factor,
    )
    s_face = physics.calc_s_from_psi(geo, psi)

  elif (
      isinstance(geo, geometry.StandardGeometry)
      and not dynamic_runtime_params_slice.profile_conditions.initial_psi_from_j
  ):
    # psi is already provided from the CHEASE equilibrium, so no need to first
    # calculate currents. However, non-inductive currents are still calculated
    # and used in current diffusion equation.
    psi_constraint = (
        dynamic_runtime_params_slice.profile_conditions.Ip
        * 1e6
        * constants.CONSTANTS.mu0
        / geo.G2_face[-1]
        * geo.rmax
    )
    psi = fvm.CellVariable(
        value=geo.psi_from_Ip,
        right_face_grad_constraint=psi_constraint,
        dr=geo.dr_norm,
    )
    q_face, _ = physics.calc_q_from_jtot_psi(
        geo=geo,
        psi=psi,
        jtot_face=geo.jtot_face,
        q_correction_factor=dynamic_runtime_params_slice.numerics.q_correction_factor,
    )
    s_face = physics.calc_s_from_psi(geo, psi)

    # calculation external currents
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
      nref=jnp.asarray(dynamic_runtime_params_slice.numerics.nref),
  )

  psidot = dataclasses.replace(
      psidot,
      value=source_models_lib.calc_psidot(
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
    temp_ion = _updated_ti(
        dynamic_runtime_params_slice, geo
    ).value
  else:
    temp_ion = core_profiles.temp_ion.value
  if (
      not static_runtime_params_slice.el_heat_eq
      and dynamic_runtime_params_slice.numerics.enable_prescribed_profile_evolution
  ):
    temp_el = _updated_te(
        dynamic_runtime_params_slice, geo
    ).value
  else:
    temp_el = core_profiles.temp_el.value
  if (
      not static_runtime_params_slice.dens_eq
      and dynamic_runtime_params_slice.numerics.enable_prescribed_profile_evolution
  ):
    ne, ni = _updated_dens(
        dynamic_runtime_params_slice, geo
    )
    ne = ne.value
    ni = ni.value
  else:
    ne = core_profiles.ne.value
    ni = core_profiles.ni.value

  return {'temp_ion': temp_ion, 'temp_el': temp_el, 'ne': ne, 'ni': ni}


def update_evolving_core_profiles(
    x_new: tuple[fvm.cell_variable.CellVariable, ...],
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    core_profiles: state.CoreProfiles,
    evolving_names: tuple[str, ...],
) -> state.CoreProfiles:
  """Returns the new core profiles after updating the evolving variables.

  Args:
    x_new: The new values of the evolving variables.
    dynamic_runtime_params_slice: The dynamic runtime params slice.
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
  ni = dataclasses.replace(
      core_profiles.ni,
      value=ne.value
      * physics.get_main_ion_dilution_factor(
          dynamic_runtime_params_slice.plasma_composition.Zimp,
          dynamic_runtime_params_slice.plasma_composition.Zeff,
      ),
  )

  return dataclasses.replace(
      core_profiles,
      temp_ion=temp_ion,
      temp_el=temp_el,
      psi=psi,
      ne=ne,
      ni=ni,
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
  Ip = dynamic_runtime_params_slice.profile_conditions.Ip  # pylint: disable=invalid-name
  Ti_bound_right = jax_utils.error_if_not_positive(  # pylint: disable=invalid-name
      dynamic_runtime_params_slice.profile_conditions.Ti_bound_right,
      'Ti_bound_right',
  )
  Te_bound_right = jax_utils.error_if_not_positive(  # pylint: disable=invalid-name
      dynamic_runtime_params_slice.profile_conditions.Te_bound_right,
      'Te_bound_right',
  )

  # calculate ne_bound_right
  # pylint: disable=invalid-name
  nGW = (
      dynamic_runtime_params_slice.profile_conditions.Ip
      / (jnp.pi * geo.Rmin**2)
      * 1e20
      / dynamic_runtime_params_slice.numerics.nref
  )
  # pylint: enable=invalid-name
  ne_bound_right = jnp.where(
      dynamic_runtime_params_slice.profile_conditions.ne_bound_right_is_fGW,
      dynamic_runtime_params_slice.profile_conditions.ne_bound_right * nGW,
      dynamic_runtime_params_slice.profile_conditions.ne_bound_right,
  )
  # define ion profile based on (flat) Zeff and single assumed impurity
  # with Zimp. main ion limited to hydrogenic species for now.
  # Assume isotopic balance for DT fusion power. Solve for ni based on:
  # Zeff = (ni + Zimp**2 * nimp)/ne  ;  nimp*Zimp + ni = ne

  dilution_factor = physics.get_main_ion_dilution_factor(
      dynamic_runtime_params_slice.plasma_composition.Zimp,
      dynamic_runtime_params_slice.plasma_composition.Zeff,
  )
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
          right_face_constraint=jnp.array(ne_bound_right * dilution_factor),
      ),
      'psi': dict(
          right_face_grad_constraint=Ip
          * 1e6
          * constants.CONSTANTS.mu0
          / geo.G2_face[-1]
          * geo.rmax,
          right_face_constraint=None,
      ),
  }


# pylint: disable=invalid-name
def _get_jtot_hires(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    dynamic_jext_params: external_current_source.DynamicRuntimeParams,
    geo: Geometry,
    bootstrap_profile: source_profiles_lib.BootstrapCurrentProfile,
    Iohm: jax.Array | float,
    jext_source: external_current_source.ExternalCurrentSource,
) -> jax.Array:
  """Calculates jtot hires."""
  j_bootstrap_hires = jnp.interp(
      geo.r_hires, geo.r_face, bootstrap_profile.j_bootstrap_face
  )

  # calculate hi-res "External" current profile (e.g. ECCD) on cell grid.
  jext_hires = jext_source.jext_hires(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      dynamic_source_runtime_params=dynamic_jext_params,
      geo=geo,
  )

  # calculate high resolution jtot and Ohmic current profile
  jformula_hires = (
      1 - geo.r_hires_norm**2
  ) ** dynamic_runtime_params_slice.profile_conditions.nu
  denom = _trapz(jformula_hires * geo.spr_hires, geo.r_hires)
  if dynamic_runtime_params_slice.profile_conditions.initial_j_is_total_current:
    Ctot_hires = (
        dynamic_runtime_params_slice.profile_conditions.Ip * 1e6 / denom
    )
    jtot_hires = jformula_hires * Ctot_hires
  else:
    Cohm_hires = Iohm * 1e6 / denom
    johm_hires = jformula_hires * Cohm_hires
    jtot_hires = johm_hires + jext_hires + j_bootstrap_hires
  return jtot_hires


def _get_jext_params(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    source_models: source_models_lib.SourceModels,
) -> external_current_source.DynamicRuntimeParams:
  """Returns dynamic runtime params for the external current source."""
  assert source_models.jext_name in dynamic_runtime_params_slice.sources, (
      f'{source_models.jext_name} not found in'
      ' dynamic_runtime_params_slice.sources. Check to make sure the'
      ' DynamicRuntimeParamsSlice was built with `sources` that include the'
      ' external current source.'
  )
  dynamic_jext_params = dynamic_runtime_params_slice.sources[
      source_models.jext_name
  ]
  assert isinstance(
      dynamic_jext_params, external_current_source.DynamicRuntimeParams
  )
  return dynamic_jext_params


# pylint: enable=invalid-name

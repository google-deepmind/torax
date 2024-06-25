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

"""Calculates Block1DCoeffs for a time step."""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
from torax import constants
from torax import geometry
from torax import jax_utils
from torax import physics
from torax import state
from torax.config import runtime_params_slice
from torax.fvm import block_1d_coeffs
from torax.sources import source_models as source_models_lib
from torax.sources import source_profiles as source_profiles_lib
from torax.transport_model import transport_model as transport_model_lib


def calculate_pereverzev_flux(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Adds Pereverzev-Corrigan flux to diffusion terms."""

  consts = constants.CONSTANTS
  true_ne_face = (
      core_profiles.ne.face_value() * dynamic_runtime_params_slice.numerics.nref
  )
  true_ni_face = (
      core_profiles.ni.face_value() * dynamic_runtime_params_slice.numerics.nref
  )

  geo_factor = jnp.concatenate(
      [jnp.ones(1), geo.g1_over_vpr_face[1:] / geo.g0_face[1:]]
  )

  chi_face_per_ion = (
      geo.g1_over_vpr_face
      * true_ni_face
      * consts.keV2J
      * dynamic_runtime_params_slice.stepper.chi_per
      / geo.rmax**2
  )

  chi_face_per_el = (
      geo.g1_over_vpr_face
      * true_ne_face
      * consts.keV2J
      * dynamic_runtime_params_slice.stepper.chi_per
      / geo.rmax**2
  )

  d_face_per_el = dynamic_runtime_params_slice.stepper.d_per / geo.rmax
  v_face_per_el = (
      core_profiles.ne.face_grad()
      / core_profiles.ne.face_value()
      * d_face_per_el
      * geo_factor
  )

  # remove Pereverzev flux from boundary region if pedestal model is on
  # (for PDE stability)
  chi_face_per_ion = jnp.where(
      jnp.logical_and(
          dynamic_runtime_params_slice.profile_conditions.set_pedestal,
          geo.r_face_norm
          > dynamic_runtime_params_slice.profile_conditions.Ped_top,
      ),
      0.0,
      chi_face_per_ion,
  )
  chi_face_per_el = jnp.where(
      jnp.logical_and(
          dynamic_runtime_params_slice.profile_conditions.set_pedestal,
          geo.r_face_norm
          > dynamic_runtime_params_slice.profile_conditions.Ped_top,
      ),
      0.0,
      chi_face_per_el,
  )
  # set heat convection terms to zero out Pereverzev-Corrigan heat diffusion
  v_heat_face_ion = (
      core_profiles.temp_ion.face_grad()
      / core_profiles.temp_ion.face_value()
      * chi_face_per_ion
  )
  v_heat_face_el = (
      core_profiles.temp_el.face_grad()
      / core_profiles.temp_el.face_value()
      * chi_face_per_el
  )

  d_face_per_el = jnp.where(
      jnp.logical_and(
          dynamic_runtime_params_slice.profile_conditions.set_pedestal,
          geo.r_face_norm
          > dynamic_runtime_params_slice.profile_conditions.Ped_top,
      ),
      0.0,
      d_face_per_el * geo.g1_over_vpr_face / geo.rmax,
  )

  v_face_per_el = jnp.where(
      jnp.logical_and(
          dynamic_runtime_params_slice.profile_conditions.set_pedestal,
          geo.r_face_norm
          > dynamic_runtime_params_slice.profile_conditions.Ped_top,
      ),
      0.0,
      v_face_per_el * geo.g0_face / geo.rmax,
  )

  chi_face_per_ion = chi_face_per_ion.at[0].set(chi_face_per_ion[1])
  chi_face_per_el = chi_face_per_el.at[0].set(chi_face_per_el[1])

  return (
      chi_face_per_ion,
      chi_face_per_el,
      v_heat_face_ion,
      v_heat_face_el,
      d_face_per_el,
      v_face_per_el,
  )


def calc_coeffs(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    transport_model: transport_model_lib.TransportModel,
    explicit_source_profiles: source_profiles_lib.SourceProfiles,
    source_models: source_models_lib.SourceModels,
    evolving_names: tuple[str, ...],
    use_pereverzev: bool = False,
    explicit_call: bool = False,
) -> block_1d_coeffs.Block1DCoeffs:
  """Calculates Block1DCoeffs for the time step described by `core_profiles`.

  Args:
    static_runtime_params_slice: General input parameters which are fixed
      through a simulation run, and if changed, would trigger a recompile.
    dynamic_runtime_params_slice: General input parameters that can change from
      time step to time step or simulation run to run, and do so without
      triggering a recompile.
    geo: Geometry describing the torus.
    core_profiles: Core plasma profiles for this time step during this iteration
      of the solver. Depending on the type of stepper being used, this may or
      may not be equal to the original plasma profiles at the beginning of the
      time step.
    transport_model: A TransportModel subclass, calculates transport coeffs.
    explicit_source_profiles: Precomputed explicit source profiles. These
      profiles either do not depend on the core profiles or depend on the
      original core profiles at the start of the time step, not the "live"
      updating core profiles. For sources that are implicit, their explicit
      profiles are set to all zeros.
    source_models: All TORAX source/sink functions that generate the explicit
      and implicit source profiles used as terms for the core profiles
      equations.
    evolving_names: The names of the evolving variables in the order that their
      coefficients should be written to `coeffs`.
    use_pereverzev: Toggle whether to calculate Pereverzev terms
    explicit_call: If True, indicates that calc_coeffs is being called for the
      explicit component of the PDE. Then calculates a reduced Block1DCoeffs if
      theta_imp=1. This saves computation for the default fully implicit
      implementation.

  Returns:
    coeffs: Block1DCoeffs containing the coefficients at this time step.
  """

  # If we are fully implicit and we are making a call for calc_coeffs for the
  # explicit components of the PDE, only return a cheaper reduced Block1DCoeffs
  if explicit_call and static_runtime_params_slice.stepper.theta_imp == 1.0:
    return _calc_coeffs_reduced(
        geo,
        core_profiles,
        evolving_names,
    )
  else:
    return _calc_coeffs_full(
        static_runtime_params_slice,
        dynamic_runtime_params_slice,
        geo,
        core_profiles,
        transport_model,
        explicit_source_profiles,
        source_models,
        evolving_names,
        use_pereverzev,
    )


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'static_runtime_params_slice',
        'transport_model',
        'source_models',
        'evolving_names',
    ],
)
def _calc_coeffs_full(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    transport_model: transport_model_lib.TransportModel,
    explicit_source_profiles: source_profiles_lib.SourceProfiles,
    source_models: source_models_lib.SourceModels,
    evolving_names: tuple[str, ...],
    use_pereverzev: bool = False,
) -> block_1d_coeffs.Block1DCoeffs:
  """Calculates Block1DCoeffs for the time step described by `core_profiles`.

  Args:
    static_runtime_params_slice: General input parameters which are fixed
      through a simulation run, and if changed, would trigger a recompile.
    dynamic_runtime_params_slice: General input parameters that can change from
      time step to time step or simulation run to run, and do so without
      triggering a recompile.
    geo: Geometry describing the torus.
    core_profiles: Core plasma profiles for this time step during this iteration
      of the solver. Depending on the type of stepper being used, this may or
      may not be equal to the original plasma profiles at the beginning of the
      time step.
    transport_model: A TransportModel subclass, calculates transport coeffs.
    explicit_source_profiles: Precomputed explicit source profiles. These
      profiles either do not depend on the core profiles or depend on the
      original core profiles at the start of the time step, not the "live"
      updating core profiles. For sources that are implicit, their explicit
      profiles are set to all zeros.
    source_models: All TORAX source/sink functions that generate the explicit
      and implicit source profiles used as terms for the core profiles
      equations.
    evolving_names: The names of the evolving variables in the order that their
      coefficients should be written to `coeffs`.
    use_pereverzev: Toggle whether to calculate Pereverzev terms

  Returns:
    coeffs: Block1DCoeffs containing the coefficients at this time step.
  """

  consts = constants.CONSTANTS

  # Boolean mask for enforcing internal temperature boundary conditions to
  # model the pedestal.
  mask = physics.internal_boundary(
      geo,
      dynamic_runtime_params_slice.profile_conditions.Ped_top,
      dynamic_runtime_params_slice.profile_conditions.set_pedestal,
  )

  # This only calculates sources set to implicit in the config. All other
  # sources are set to 0 (and should have their profiles already calculated in
  # explicit_source_profiles).
  implicit_source_profiles = source_models_lib.build_source_profiles(
      source_models=source_models,
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      geo=geo,
      core_profiles=core_profiles,
      explicit=False,
  )
  # The above call calculates the implicit value for the bootstrap current. Note
  # that this is potentially wasteful in case the source is explicit, but
  # recalculate here to avoid issues with JAX branching in the logic.
  # Decide which values to use depending on whether the source is explicit or
  # implicit.
  sigma = jax_utils.select(
      dynamic_runtime_params_slice.sources[
          source_models.j_bootstrap_name
      ].is_explicit,
      explicit_source_profiles.j_bootstrap.sigma,
      implicit_source_profiles.j_bootstrap.sigma,
  )
  j_bootstrap = jax_utils.select(
      dynamic_runtime_params_slice.sources[
          source_models.j_bootstrap_name
      ].is_explicit,
      explicit_source_profiles.j_bootstrap.j_bootstrap,
      implicit_source_profiles.j_bootstrap.j_bootstrap,
  )
  j_bootstrap_face = jax_utils.select(
      dynamic_runtime_params_slice.sources[
          source_models.j_bootstrap_name
      ].is_explicit,
      explicit_source_profiles.j_bootstrap.j_bootstrap_face,
      implicit_source_profiles.j_bootstrap.j_bootstrap_face,
  )
  I_bootstrap = jax_utils.select(  # pylint: disable=invalid-name
      dynamic_runtime_params_slice.sources[
          source_models.j_bootstrap_name
      ].is_explicit,
      explicit_source_profiles.j_bootstrap.I_bootstrap,
      implicit_source_profiles.j_bootstrap.I_bootstrap,
  )

  currents = dataclasses.replace(
      core_profiles.currents,
      j_bootstrap=j_bootstrap,
      j_bootstrap_face=j_bootstrap_face,
      johm=core_profiles.currents.jtot
      - j_bootstrap
      - core_profiles.currents.jext,
      johm_face=(
          core_profiles.currents.jtot_face
          - j_bootstrap_face
          - core_profiles.currents.jext_face
      ),
      I_bootstrap=I_bootstrap,
      sigma=sigma,
  )
  core_profiles = dataclasses.replace(core_profiles, currents=currents)

  # psi source terms. Source matrix is zero for all psi sources
  source_mat_psi = jnp.zeros_like(geo.r)

  # fill source vector based on both original and updated core profiles
  source_psi = source_models_lib.sum_sources_psi(
      geo,
      implicit_source_profiles,
      source_models,
  ) + source_models_lib.sum_sources_psi(
      geo,
      explicit_source_profiles,
      source_models,
  )

  true_ne_face = (
      core_profiles.ne.face_value() * dynamic_runtime_params_slice.numerics.nref
  )
  true_ni_face = (
      core_profiles.ni.face_value() * dynamic_runtime_params_slice.numerics.nref
  )

  # Transient term coefficient vector (has radial dependence through r, n)
  toc_temp_ion = (
      1.5
      * geo.vpr ** (-2.0 / 3.0)
      * consts.keV2J
      * dynamic_runtime_params_slice.numerics.nref
  )
  tic_temp_ion = core_profiles.ni.value * geo.vpr ** (5.0 / 3.0)
  toc_temp_el = (
      1.5
      * geo.vpr ** (-2.0 / 3.0)
      * consts.keV2J
      * dynamic_runtime_params_slice.numerics.nref
  )
  tic_temp_el = core_profiles.ne.value * geo.vpr ** (5.0 / 3.0)
  toc_psi = (
      1.0
      / dynamic_runtime_params_slice.numerics.resistivity_mult
      * geo.r
      * sigma
      * consts.mu0
      / geo.J**2
      / geo.Rmaj
  )
  tic_psi = jnp.ones_like(toc_psi)
  toc_dens_el = jnp.ones_like(geo.vpr)
  tic_dens_el = geo.vpr

  # Diffusion term coefficients
  transport_coeffs = transport_model(
      dynamic_runtime_params_slice, geo, core_profiles
  )
  chi_face_ion = transport_coeffs.chi_face_ion
  chi_face_el = transport_coeffs.chi_face_el
  d_face_el = transport_coeffs.d_face_el
  v_face_el = transport_coeffs.v_face_el
  d_face_psi = (
      geo.g2g3_over_rho_face * geo.Rmaj / (16 * jnp.pi**4 * geo.rmax**2)
  )

  if static_runtime_params_slice.dens_eq:
    if d_face_el is None or v_face_el is None:
      raise NotImplementedError(
          f'{type(transport_model)} does not support the density equation.'
      )

  # Apply inner and outer patch constant transport coefficients. rho_inner and
  # rho_outer are shifted by consts.eps (1e-7) to avoid ambiguities if their
  # values are close to and geo.r_face_norm values.
  # Note that Pereverzev-Corrigan terms will still be included in constant
  # transport regions, to avoid transient discontinuities
  chi_face_ion = jnp.where(
      jnp.logical_and(
          dynamic_runtime_params_slice.transport.apply_inner_patch,
          geo.r_face_norm
          < dynamic_runtime_params_slice.transport.rho_inner + consts.eps,
      ),
      dynamic_runtime_params_slice.transport.chii_inner,
      chi_face_ion,
  )
  chi_face_el = jnp.where(
      jnp.logical_and(
          dynamic_runtime_params_slice.transport.apply_inner_patch,
          geo.r_face_norm
          < dynamic_runtime_params_slice.transport.rho_inner + consts.eps,
      ),
      dynamic_runtime_params_slice.transport.chie_inner,
      chi_face_el,
  )
  d_face_el = jnp.where(
      jnp.logical_and(
          dynamic_runtime_params_slice.transport.apply_inner_patch,
          geo.r_face_norm
          < dynamic_runtime_params_slice.transport.rho_inner + consts.eps,
      ),
      dynamic_runtime_params_slice.transport.De_inner,
      d_face_el,
  )
  v_face_el = jnp.where(
      jnp.logical_and(
          dynamic_runtime_params_slice.transport.apply_inner_patch,
          geo.r_face_norm
          < dynamic_runtime_params_slice.transport.rho_inner + consts.eps,
      ),
      dynamic_runtime_params_slice.transport.Ve_inner,
      v_face_el,
  )

  # Apply outer patch constant transport coefficients.
  # Due to Pereverzev-Corrigan convection, it is required
  # for the convection modes to be 'ghost' to avoid numerical instability
  chi_face_ion = jnp.where(
      jnp.logical_and(
          jnp.logical_and(
              dynamic_runtime_params_slice.transport.apply_outer_patch,
              jnp.logical_not(
                  dynamic_runtime_params_slice.profile_conditions.set_pedestal
              ),
          ),
          geo.r_face_norm
          > dynamic_runtime_params_slice.transport.rho_outer - consts.eps,
      ),
      dynamic_runtime_params_slice.transport.chii_outer,
      chi_face_ion,
  )
  chi_face_el = jnp.where(
      jnp.logical_and(
          jnp.logical_and(
              dynamic_runtime_params_slice.transport.apply_outer_patch,
              jnp.logical_not(
                  dynamic_runtime_params_slice.profile_conditions.set_pedestal
              ),
          ),
          geo.r_face_norm
          > dynamic_runtime_params_slice.transport.rho_outer - consts.eps,
      ),
      dynamic_runtime_params_slice.transport.chie_outer,
      chi_face_el,
  )
  d_face_el = jnp.where(
      jnp.logical_and(
          jnp.logical_and(
              dynamic_runtime_params_slice.transport.apply_outer_patch,
              jnp.logical_not(
                  dynamic_runtime_params_slice.profile_conditions.set_pedestal
              ),
          ),
          geo.r_face_norm
          > dynamic_runtime_params_slice.transport.rho_outer - consts.eps,
      ),
      dynamic_runtime_params_slice.transport.De_outer,
      d_face_el,
  )
  v_face_el = jnp.where(
      jnp.logical_and(
          jnp.logical_and(
              dynamic_runtime_params_slice.transport.apply_outer_patch,
              jnp.logical_not(
                  dynamic_runtime_params_slice.profile_conditions.set_pedestal
              ),
          ),
          geo.r_face_norm
          > dynamic_runtime_params_slice.transport.rho_outer - consts.eps,
      ),
      dynamic_runtime_params_slice.transport.Ve_outer,
      v_face_el,
  )

  # Update the transport coeffs with the new profiles.
  # This version of the core transport is returned to the caller to help with
  # inspection.
  transport_coeffs = dataclasses.replace(
      transport_coeffs,
      chi_face_ion=chi_face_ion,
      chi_face_el=chi_face_el,
      d_face_el=d_face_el,
      v_face_el=v_face_el,
  )

  # entire coefficient preceding dT/dr in heat transport equations
  full_chi_face_ion = (
      geo.g1_over_vpr_face
      * true_ni_face
      * consts.keV2J
      * chi_face_ion
      / geo.rmax**2
  )
  full_chi_face_el = (
      geo.g1_over_vpr_face
      * true_ne_face
      * consts.keV2J
      * chi_face_el
      / geo.rmax**2
  )

  # entire coefficient preceding dne/dr in particle equation
  full_d_face_el = geo.g1_over_vpr_face * d_face_el / geo.rmax**2
  full_v_face_el = geo.g0_face * v_face_el / geo.rmax

  # density source terms. Initialize source matrix to zero
  source_mat_nn = jnp.zeros_like(geo.r)

  # density source vector based both on original and updated core profiles
  source_ne = source_models_lib.sum_sources_ne(
      geo,
      explicit_source_profiles,
      source_models,
  ) + source_models_lib.sum_sources_ne(
      geo,
      implicit_source_profiles,
      source_models,
  )

  # calculate neped
  # pylint: disable=invalid-name
  nGW = (
      dynamic_runtime_params_slice.profile_conditions.Ip
      / (jnp.pi * geo.Rmin**2)
      * 1e20
      / dynamic_runtime_params_slice.numerics.nref
  )
  # pylint: enable=invalid-name
  neped_unnorm = jnp.where(
      dynamic_runtime_params_slice.profile_conditions.neped_is_fGW,
      dynamic_runtime_params_slice.profile_conditions.neped * nGW,
      dynamic_runtime_params_slice.profile_conditions.neped,
  )

  source_ne += jnp.where(
      dynamic_runtime_params_slice.profile_conditions.set_pedestal,
      mask * dynamic_runtime_params_slice.numerics.largeValue_n * neped_unnorm,
      0.0,
  )
  source_mat_nn += jnp.where(
      dynamic_runtime_params_slice.profile_conditions.set_pedestal,
      -(mask * dynamic_runtime_params_slice.numerics.largeValue_n),
      0.0,
  )

  # Pereverzev-Corrigan correction for heat and particle transport
  # (deals with stiff nonlinearity of transport coefficients)
  # TODO(b/311653933) this forces us to include value 0
  # convection terms in discrete system, slowing compilation down by ~10%.
  # See if can improve with a different pattern.
  (
      chi_face_per_ion,
      chi_face_per_el,
      v_heat_face_ion,
      v_heat_face_el,
      d_face_per_el,
      v_face_per_el,
  ) = jax.lax.cond(
      use_pereverzev,
      lambda: calculate_pereverzev_flux(
          dynamic_runtime_params_slice,
          geo,
          core_profiles,
      ),
      lambda: tuple([jnp.zeros_like(geo.r_face)] * 6),
  )

  full_chi_face_ion += chi_face_per_ion
  full_chi_face_el += chi_face_per_el
  full_d_face_el += d_face_per_el
  full_v_face_el += v_face_per_el

  # Ion and electron heat sources.
  qei = source_models.qei_source.get_qei(
      static_runtime_params_slice=static_runtime_params_slice,
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      dynamic_source_runtime_params=dynamic_runtime_params_slice.sources[
          source_models.qei_source_name
      ],
      geo=geo,
      # For Qei, always use the current set of core profiles.
      # In the linear solver, core_profiles is the set of profiles at time t (at
      # the start of the time step) or the updated core_profiles in
      # predictor-corrector, and in the nonlinear solver, calc_coeffs is called
      # at least twice, once with the core_profiles at time t, and again
      # (iteratively) with core_profiles at t+dt.
      core_profiles=core_profiles,
  )
  # Update the implicit profiles with the qei info.
  implicit_source_profiles = dataclasses.replace(
      implicit_source_profiles,
      qei=qei,
  )

  # Fill heat transport equation sources. Initialize source matrices to zero

  source_mat_ii = jnp.zeros_like(geo.r)
  source_mat_ee = jnp.zeros_like(geo.r)

  source_i = source_models_lib.sum_sources_temp_ion(
      geo,
      explicit_source_profiles,
      source_models,
  ) + source_models_lib.sum_sources_temp_ion(
      geo,
      implicit_source_profiles,
      source_models,
  )

  source_e = source_models_lib.sum_sources_temp_el(
      geo,
      explicit_source_profiles,
      source_models,
  ) + source_models_lib.sum_sources_temp_el(
      geo,
      implicit_source_profiles,
      source_models,
  )

  # Add the Qei effects.
  source_mat_ii += qei.implicit_ii * geo.vpr
  source_i += qei.explicit_i * geo.vpr
  source_mat_ee += qei.implicit_ee * geo.vpr
  source_e += qei.explicit_e * geo.vpr
  source_mat_ie = qei.implicit_ie * geo.vpr
  source_mat_ei = qei.implicit_ei * geo.vpr

  # Pedestal
  source_i += jnp.where(
      dynamic_runtime_params_slice.profile_conditions.set_pedestal,
      mask
      * dynamic_runtime_params_slice.numerics.largeValue_T
      * dynamic_runtime_params_slice.profile_conditions.Tiped,
      0.0,
  )
  source_e += jnp.where(
      dynamic_runtime_params_slice.profile_conditions.set_pedestal,
      mask
      * dynamic_runtime_params_slice.numerics.largeValue_T
      * dynamic_runtime_params_slice.profile_conditions.Teped,
      0.0,
  )

  source_mat_ii -= jnp.where(
      dynamic_runtime_params_slice.profile_conditions.set_pedestal,
      mask * dynamic_runtime_params_slice.numerics.largeValue_T,
      0.0,
  )
  source_mat_ee -= jnp.where(
      dynamic_runtime_params_slice.profile_conditions.set_pedestal,
      mask * dynamic_runtime_params_slice.numerics.largeValue_T,
      0.0,
  )

  # Build arguments to solver based on which variables are evolving
  var_to_toc = {
      'temp_ion': toc_temp_ion,
      'temp_el': toc_temp_el,
      'psi': toc_psi,
      'ne': toc_dens_el,
  }
  var_to_tic = {
      'temp_ion': tic_temp_ion,
      'temp_el': tic_temp_el,
      'psi': tic_psi,
      'ne': tic_dens_el,
  }
  transient_out_cell = tuple(var_to_toc[var] for var in evolving_names)
  transient_in_cell = tuple(var_to_tic[var] for var in evolving_names)

  var_to_d_face = {
      'temp_ion': full_chi_face_ion,
      'temp_el': full_chi_face_el,
      'psi': d_face_psi,
      'ne': full_d_face_el,
  }
  d_face = tuple(var_to_d_face[var] for var in evolving_names)

  var_to_v_face = {
      'temp_ion': v_heat_face_ion,
      'temp_el': v_heat_face_el,
      'ne': full_v_face_el,
  }
  v_face = tuple(var_to_v_face.get(var) for var in evolving_names)

  # d maps (row var, col var) to the coefficient for that block of the matrix
  # (Can't use a descriptive name or the nested comprehension to build the
  # matrix gets too long)
  d = {
      ('temp_ion', 'temp_ion'): source_mat_ii,
      ('temp_ion', 'temp_el'): source_mat_ie,
      ('temp_el', 'temp_ion'): source_mat_ei,
      ('temp_el', 'temp_el'): source_mat_ee,
      ('ne', 'ne'): source_mat_nn,
      ('psi', 'psi'): source_mat_psi,
  }
  source_mat_cell = tuple(
      tuple(d.get((row_block, col_block)) for col_block in evolving_names)
      for row_block in evolving_names
  )

  var_to_source = {
      'temp_ion': source_i,
      'temp_el': source_e,
      'psi': source_psi,
      'ne': source_ne,
  }
  source_cell = tuple(var_to_source.get(var) for var in evolving_names)

  coeffs = block_1d_coeffs.Block1DCoeffs(
      transient_out_cell=transient_out_cell,
      transient_in_cell=transient_in_cell,
      d_face=d_face,
      v_face=v_face,
      source_mat_cell=source_mat_cell,
      source_cell=source_cell,
      auxiliary_outputs=(implicit_source_profiles, transport_coeffs),
  )

  return coeffs


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'evolving_names',
    ],
)
def _calc_coeffs_reduced(
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    evolving_names: tuple[str, ...],
) -> block_1d_coeffs.Block1DCoeffs:
  """Calculates only the transient_in_cell terms in Block1DCoeffs."""

  # Only transient_in_cell is used for explicit terms if theta_imp=1
  tic_temp_ion = core_profiles.ni.value * geo.vpr ** (5.0 / 3.0)
  tic_temp_el = core_profiles.ne.value * geo.vpr ** (5.0 / 3.0)
  tic_psi = jnp.ones_like(geo.vpr)
  tic_dens_el = geo.vpr

  var_to_tic = {
      'temp_ion': tic_temp_ion,
      'temp_el': tic_temp_el,
      'psi': tic_psi,
      'ne': tic_dens_el,
  }
  transient_in_cell = tuple(var_to_tic[var] for var in evolving_names)

  coeffs = block_1d_coeffs.Block1DCoeffs(
      transient_in_cell=transient_in_cell,
  )
  return coeffs

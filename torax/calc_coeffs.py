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

import chex
import jax
import jax.numpy as jnp
from torax import config_slice
from torax import constants
from torax import geometry
from torax import jax_utils
from torax import physics
from torax import state as state_module
from torax.fvm import block_1d_coeffs
from torax.sources import qei_source as qei_source_lib
from torax.sources import source_profiles as source_profiles_lib
from torax.transport_model import transport_model as transport_model_lib


def _default_side_output(shape: tuple[int, ...]):
  return jnp.zeros(shape)


@chex.dataclass
class AuxOutput:
  """Auxiliary outputs while calculating Block1DCoeffs.

  During each simulation step (and potentially multiple times per simulation
  step), the coeffs will be calculated by calc_coeffs(). While that function's
  main output is the Block1DCoeffs, calc_coeffs() also outputs this object,
  which provides a hook to include any extra auxiliary outputs useful for
  inspecting any interim values while the coeffs are calculated.

  If extending TORAX, feel free to add more attributes to this class.
  """

  # pylint: disable=invalid-name
  chi_face_ion: jax.Array
  chi_face_el: jax.Array
  source_ion: jax.Array
  source_el: jax.Array
  Pfus_i: jax.Array
  Pfus_e: jax.Array
  Pohm: jax.Array
  Qei: jax.Array
  # pylint: enable=invalid-name

  @classmethod
  def build_from_geo(cls, geo: geometry.Geometry) -> 'AuxOutput':
    return cls(
        chi_face_ion=_default_side_output(geo.r_face.shape),
        chi_face_el=_default_side_output(geo.r_face.shape),
        source_ion=_default_side_output(geo.r.shape),
        source_el=_default_side_output(geo.r.shape),
        Pfus_i=_default_side_output(geo.r.shape),
        Pfus_e=_default_side_output(geo.r.shape),
        Pohm=_default_side_output(geo.r.shape),
        Qei=_default_side_output(geo.r.shape),
    )


def calculate_pereverzev_flux(
    state: state_module.State,
    geo: geometry.Geometry,
    dynamic_config_slice: config_slice.DynamicConfigSlice,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Adds Pereverzev-Corrigan flux to diffusion terms."""

  consts = constants.CONSTANTS
  true_ne_face = state.ne.face_value() * dynamic_config_slice.nref
  true_ni_face = state.ni.face_value() * dynamic_config_slice.nref

  geo_factor = jnp.concatenate(
      [jnp.ones(1), geo.g1_over_vpr_face[1:] / geo.g0_face[1:]]
  )

  chi_face_per_ion = (
      geo.g1_over_vpr_face
      * true_ni_face
      * consts.keV2J
      * dynamic_config_slice.solver.chi_per
      / geo.rmax**2
  )

  chi_face_per_el = (
      geo.g1_over_vpr_face
      * true_ne_face
      * consts.keV2J
      * dynamic_config_slice.solver.chi_per
      / geo.rmax**2
  )

  d_face_per_el = dynamic_config_slice.solver.d_per / geo.rmax
  v_face_per_el = (
      state.ne.face_grad() / state.ne.face_value() * d_face_per_el * geo_factor
  )

  # remove Pereverzev flux from boundary region if pedestal model is on
  # (for PDE stability)
  chi_face_per_ion = jnp.where(
      jnp.logical_and(
          dynamic_config_slice.set_pedestal,
          geo.r_face_norm > dynamic_config_slice.Ped_top,
      ),
      0.0,
      chi_face_per_ion,
  )
  chi_face_per_el = jnp.where(
      jnp.logical_and(
          dynamic_config_slice.set_pedestal,
          geo.r_face_norm > dynamic_config_slice.Ped_top,
      ),
      0.0,
      chi_face_per_el,
  )
  # set heat convection terms to zero out Pereverzev-Corrigan heat diffusion
  v_heat_face_ion = (
      state.temp_ion.face_grad()
      / state.temp_ion.face_value()
      * chi_face_per_ion
  )
  v_heat_face_el = (
      state.temp_el.face_grad() / state.temp_el.face_value() * chi_face_per_el
  )

  d_face_per_el = jnp.where(
      jnp.logical_and(
          dynamic_config_slice.set_pedestal,
          geo.r_face_norm > dynamic_config_slice.Ped_top,
      ),
      0.0,
      d_face_per_el * geo.g1_over_vpr_face / geo.rmax,
  )

  v_face_per_el = jnp.where(
      jnp.logical_and(
          dynamic_config_slice.set_pedestal,
          geo.r_face_norm > dynamic_config_slice.Ped_top,
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
    sim_state: state_module.ToraxSimState,
    evolving_names: tuple[str, ...],
    geo: geometry.Geometry,
    dynamic_config_slice: config_slice.DynamicConfigSlice,
    static_config_slice: config_slice.StaticConfigSlice,
    transport_model: transport_model_lib.TransportModel,
    explicit_source_profiles: source_profiles_lib.SourceProfiles,
    sources: source_profiles_lib.Sources,
    use_pereverzev: bool = False,
    explicit_call: bool = False,
) -> block_1d_coeffs.Block1DCoeffs:
  """Calculates Block1DCoeffs for the time step described by `state`.

  Args:
    sim_state: Full simulation state for this time step during this iteration.
      Depending on the type of stepper being used, this may or may not be equal
      to the original state at the beginning of the time step.
    evolving_names: The names of the evolving variables in the order that their
      coefficients should be written to `coeffs`.
    geo: Geometry describing the torus.
    dynamic_config_slice: General input parameters that can change from time
      step to time step or simulation run to run, and do so without triggering a
      recompile.
    static_config_slice: General input parameters which are fixed through a
      simulation run, and if changed, would trigger a recompile.
    transport_model: A TransportModel subclass, calculates transport coeffs.
    explicit_source_profiles: Precomputed explicit source profiles. These
      profiles either do not depend on the state or depend on the original state
      at the start of the time step (orig_state), not the "live" state (state).
      For sources that are implicit, their explicit profiles are set to all
      zeros.
    sources: All TORAX source/sinks that generate the explicit and implicit
      source profiles used as terms for the mesh state equations.
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
  if explicit_call and static_config_slice.solver.theta_imp == 1.0:
    return _calc_coeffs_reduced(
        state=sim_state.mesh_state,
        evolving_names=evolving_names,
        geo=geo,
    )
  else:
    return _calc_coeffs_full(
        sim_state=sim_state,
        evolving_names=evolving_names,
        geo=geo,
        dynamic_config_slice=dynamic_config_slice,
        static_config_slice=static_config_slice,
        transport_model=transport_model,
        explicit_source_profiles=explicit_source_profiles,
        sources=sources,
        use_pereverzev=use_pereverzev,
    )


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'transport_model',
        'static_config_slice',
        'evolving_names',
        'sources',
    ],
)
def _calc_coeffs_full(
    sim_state: state_module.ToraxSimState,
    evolving_names: tuple[str, ...],
    geo: geometry.Geometry,
    dynamic_config_slice: config_slice.DynamicConfigSlice,
    static_config_slice: config_slice.StaticConfigSlice,
    transport_model: transport_model_lib.TransportModel,
    explicit_source_profiles: source_profiles_lib.SourceProfiles,
    sources: source_profiles_lib.Sources,
    use_pereverzev: bool = False,
) -> block_1d_coeffs.Block1DCoeffs:
  """Calculates Block1DCoeffs for the time step described by `state`.

  Args:
    sim_state: Full simulation state for this time step during this iteration.
      Depending on the type of stepper being used, this may or may not be equal
      to the original state at the beginning of the time step.
    evolving_names: The names of the evolving variables in the order that their
      coefficients should be written to `coeffs`.
    geo: Geometry describing the torus.
    dynamic_config_slice: General input parameters that can change from time
      step to time step or simulation run to run, and do so without triggering a
      recompile.
    static_config_slice: General input parameters which are fixed through a
      simulation run, and if changed, would trigger a recompile.
    transport_model: A TransportModel subclass, calculates transport coeffs.
    explicit_source_profiles: Precomputed explicit source profiles. These
      profiles either do not depend on the state or depend on the original state
      at the start of the time step (orig_state), not the "live" state (state).
      For sources that are implicit, their explicit profiles are set to all
      zeros.
    sources: All TORAX source/sinks that generate the explicit and implicit
      source profiles used as terms for the mesh state equations.
    use_pereverzev: Toggle whether to calculate Pereverzev terms

  Returns:
    coeffs: Block1DCoeffs containing the coefficients at this time step.
  """

  consts = constants.CONSTANTS
  #  Initialize AuxOutput object with array sizes taken from geo
  aux_outputs = AuxOutput.build_from_geo(geo)

  # Boolean mask for enforcing internal temperature boundary conditions to
  # model the pedestal.
  mask = physics.internal_boundary(
      geo, dynamic_config_slice.Ped_top, dynamic_config_slice.set_pedestal
  )

  # This only calculates sources set to implicit in the config. All other
  # sources are set to 0 (and should have their profiles already calculated in
  # explicit_source_profiles).
  implicit_source_profiles = source_profiles_lib.build_source_profiles(
      sources=sources,
      dynamic_config_slice=dynamic_config_slice,
      geo=geo,
      sim_state=sim_state,
      explicit=False,
  )
  # The above call calculates the implicit value for the bootstrap current. Note
  # that this is potentially wasteful in case the source is explicit, but
  # recalculate here to avoid issues with JAX branching in the logic.
  # TODO( b/314308399): Remove bootstrap current from the state and
  # simplify this so that we don't treat bootstrap current as a special case and
  # can avoid this recalculation.
  # TODO(b/323504363): Make bootstrap implicit by default. Will need to rerun
  # tests with new baselines.
  # Decide which values to use depending on whether the source is explicit or
  # implicit.
  sigma = jax_utils.select(
      dynamic_config_slice.sources[sources.j_bootstrap.name].is_explicit,
      explicit_source_profiles.j_bootstrap.sigma,
      implicit_source_profiles.j_bootstrap.sigma,
  )
  j_bootstrap = jax_utils.select(
      dynamic_config_slice.sources[sources.j_bootstrap.name].is_explicit,
      explicit_source_profiles.j_bootstrap.j_bootstrap,
      implicit_source_profiles.j_bootstrap.j_bootstrap,
  )
  j_bootstrap_face = jax_utils.select(
      dynamic_config_slice.sources[sources.j_bootstrap.name].is_explicit,
      explicit_source_profiles.j_bootstrap.j_bootstrap_face,
      implicit_source_profiles.j_bootstrap.j_bootstrap_face,
  )
  I_bootstrap = jax_utils.select(  # pylint: disable=invalid-name
      dynamic_config_slice.sources[sources.j_bootstrap.name].is_explicit,
      explicit_source_profiles.j_bootstrap.I_bootstrap,
      implicit_source_profiles.j_bootstrap.I_bootstrap,
  )

  currents = dataclasses.replace(
      sim_state.mesh_state.currents,
      j_bootstrap=j_bootstrap,
      j_bootstrap_face=j_bootstrap_face,
      johm=(
          sim_state.mesh_state.currents.jtot
          - j_bootstrap
          - sim_state.mesh_state.currents.jext
      ),
      johm_face=(
          sim_state.mesh_state.currents.jtot_face
          - j_bootstrap_face
          - sim_state.mesh_state.currents.jext_face
      ),
      I_bootstrap=I_bootstrap,
      sigma=sigma,
  )
  sim_state.mesh_state = dataclasses.replace(
      sim_state.mesh_state, currents=currents
  )

  # psi source terms. Source matrix is zero for all psi sources
  source_mat_psi = jnp.zeros_like(geo.r)

  # fill source vector based on both original and updated state
  source_psi = source_profiles_lib.sum_sources_psi(
      sources,
      implicit_source_profiles,
      geo,
      dynamic_config_slice.Rmaj,
  ) + source_profiles_lib.sum_sources_psi(
      sources,
      explicit_source_profiles,
      geo,
      dynamic_config_slice.Rmaj,
  )

  true_ne_face = (
      sim_state.mesh_state.ne.face_value() * dynamic_config_slice.nref
  )
  true_ni_face = (
      sim_state.mesh_state.ni.face_value() * dynamic_config_slice.nref
  )

  # Transient term coefficient vector (has radial dependence through r, n)
  toc_temp_ion = 1.5 * geo.vpr * consts.keV2J * dynamic_config_slice.nref
  tic_temp_ion = sim_state.mesh_state.ni.value
  toc_temp_el = 1.5 * geo.vpr * consts.keV2J * dynamic_config_slice.nref
  tic_temp_el = sim_state.mesh_state.ne.value
  toc_psi = (
      1.0
      / dynamic_config_slice.resistivity_mult
      * geo.r
      * sigma
      * consts.mu0
      / geo.J**2
      / dynamic_config_slice.Rmaj
  )
  tic_psi = jnp.ones_like(toc_psi)
  toc_dens_el = geo.vpr
  tic_dens_el = jnp.ones_like(geo.vpr)

  # Diffusion term coefficients
  transport_coeffs = transport_model(
      dynamic_config_slice, geo, sim_state.mesh_state
  )
  chi_face_ion = transport_coeffs.chi_face_ion
  chi_face_el = transport_coeffs.chi_face_el
  d_face_el = transport_coeffs.d_face_el
  v_face_el = transport_coeffs.v_face_el
  d_face_psi = geo.G2_face / geo.J_face / geo.rmax**2

  if static_config_slice.dens_eq:
    if d_face_el is None or v_face_el is None:
      raise NotImplementedError(
          f'{type(transport_model)} does not support the density equation.'
      )

  # Apply inner and outer patch constant transport coefficients.
  # Note that Pereverzev-Corrigan terms will still be included in constant
  # transport regions, to avoid transient discontinuities
  chi_face_ion = jnp.where(
      jnp.logical_and(
          dynamic_config_slice.transport.apply_inner_patch,
          geo.r_face_norm < dynamic_config_slice.transport.rho_inner,
      ),
      dynamic_config_slice.transport.chii_inner,
      chi_face_ion,
  )
  chi_face_el = jnp.where(
      jnp.logical_and(
          dynamic_config_slice.transport.apply_inner_patch,
          geo.r_face_norm < dynamic_config_slice.transport.rho_inner,
      ),
      dynamic_config_slice.transport.chie_inner,
      chi_face_el,
  )
  d_face_el = jnp.where(
      jnp.logical_and(
          dynamic_config_slice.transport.apply_inner_patch,
          geo.r_face_norm < dynamic_config_slice.transport.rho_inner,
      ),
      dynamic_config_slice.transport.De_inner,
      d_face_el,
  )
  v_face_el = jnp.where(
      jnp.logical_and(
          dynamic_config_slice.transport.apply_inner_patch,
          geo.r_face_norm < dynamic_config_slice.transport.rho_inner,
      ),
      dynamic_config_slice.transport.Ve_inner,
      v_face_el,
  )

  # Apply outer patch constant transport coefficients.
  # Due to Pereverzev-Corrigan convection, it is required
  # for the convection modes to be 'ghost' to avoid numerical instability
  chi_face_ion = jnp.where(
      jnp.logical_and(
          jnp.logical_and(
              dynamic_config_slice.transport.apply_outer_patch,
              jnp.logical_not(dynamic_config_slice.set_pedestal),
          ),
          geo.r_face_norm > dynamic_config_slice.transport.rho_outer,
      ),
      dynamic_config_slice.transport.chii_outer,
      chi_face_ion,
  )
  chi_face_el = jnp.where(
      jnp.logical_and(
          jnp.logical_and(
              dynamic_config_slice.transport.apply_outer_patch,
              jnp.logical_not(dynamic_config_slice.set_pedestal),
          ),
          geo.r_face_norm > dynamic_config_slice.transport.rho_outer,
      ),
      dynamic_config_slice.transport.chie_outer,
      chi_face_el,
  )
  d_face_el = jnp.where(
      jnp.logical_and(
          jnp.logical_and(
              dynamic_config_slice.transport.apply_outer_patch,
              jnp.logical_not(dynamic_config_slice.set_pedestal),
          ),
          geo.r_face_norm > dynamic_config_slice.transport.rho_outer,
      ),
      dynamic_config_slice.transport.De_outer,
      d_face_el,
  )
  v_face_el = jnp.where(
      jnp.logical_and(
          jnp.logical_and(
              dynamic_config_slice.transport.apply_outer_patch,
              jnp.logical_not(dynamic_config_slice.set_pedestal),
          ),
          geo.r_face_norm > dynamic_config_slice.transport.rho_outer,
      ),
      dynamic_config_slice.transport.Ve_outer,
      v_face_el,
  )

  aux_outputs.chi_face_ion = chi_face_ion
  aux_outputs.chi_face_el = chi_face_el

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

  # density source vector based both on original and updated state
  source_ne = source_profiles_lib.sum_sources_ne(
      sources,
      explicit_source_profiles,
      geo,
  ) + source_profiles_lib.sum_sources_ne(
      sources,
      implicit_source_profiles,
      geo,
  )

  if full_v_face_el is not None:
    # TODO(b/323504363): Move this masking to the constant transport model.
    full_v_face_el = jnp.where(
        jnp.logical_and(
            dynamic_config_slice.set_pedestal,
            geo.r_face_norm > dynamic_config_slice.Ped_top,
        ),
        0.0,
        full_v_face_el,
    )
  source_ne += jnp.where(
      dynamic_config_slice.set_pedestal,
      mask * dynamic_config_slice.largeValue_n * dynamic_config_slice.neped,
      0.0,
  )
  source_mat_nn += jnp.where(
      dynamic_config_slice.set_pedestal,
      -(mask * dynamic_config_slice.largeValue_n),
      0.0,
  )

  # Pereverzev-Corrigan correction for heat and particle transport
  # (deals with stiff nonlinearity of transport coefficients)
  # TODO( b/311653933) this forces us to include value 0
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
          sim_state.mesh_state, geo, dynamic_config_slice
      ),
      lambda: tuple([jnp.zeros_like(geo.r_face)] * 6),
  )

  full_chi_face_ion += chi_face_per_ion
  full_chi_face_el += chi_face_per_el
  full_d_face_el += d_face_per_el
  full_v_face_el += v_face_per_el

  # Ion and electron heat sources.
  # Select which state to use for Qei.
  qei = sources.qei_source.get_qei(
      dynamic_config_slice.sources[sources.qei_source.name].source_type,
      dynamic_config_slice=dynamic_config_slice,
      static_config_slice=static_config_slice,
      geo=geo,
      # For Qei, always use the current state.
      # In the linear solver, state is the state at time t (at the start of the
      # time step) or the updated state in predictor-corrector, and in the
      # nonlinear solver, calc_coeffs is called at least twice, once with the
      # state at time t, and again (iteratively) with state at t+dt.
      sim_state=sim_state,
  )
  _populate_aux_outputs_with_ion_el_heat_sources(
      implicit_source_profiles,
      explicit_source_profiles,
      qei,
      sim_state.mesh_state,
      aux_outputs,
  )

  # Fill heat transport equation sources. Initialize source matrices to zero

  source_mat_ii = jnp.zeros_like(geo.r)
  source_mat_ee = jnp.zeros_like(geo.r)

  source_i = source_profiles_lib.sum_sources_temp_ion(
      sources,
      explicit_source_profiles,
      geo,
  ) + source_profiles_lib.sum_sources_temp_ion(
      sources,
      implicit_source_profiles,
      geo,
  )

  source_e = source_profiles_lib.sum_sources_temp_el(
      sources,
      explicit_source_profiles,
      geo,
  ) + source_profiles_lib.sum_sources_temp_el(
      sources,
      implicit_source_profiles,
      geo,
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
      dynamic_config_slice.set_pedestal,
      mask * dynamic_config_slice.largeValue_T * dynamic_config_slice.Tiped,
      0.0,
  )
  source_e += jnp.where(
      dynamic_config_slice.set_pedestal,
      mask * dynamic_config_slice.largeValue_T * dynamic_config_slice.Teped,
      0.0,
  )

  source_mat_ii -= jnp.where(
      dynamic_config_slice.set_pedestal,
      mask * dynamic_config_slice.largeValue_T,
      0.0,
  )
  source_mat_ee -= jnp.where(
      dynamic_config_slice.set_pedestal,
      mask * dynamic_config_slice.largeValue_T,
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
      auxiliary_outputs=aux_outputs,
  )

  return coeffs


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'evolving_names',
    ],
)
def _calc_coeffs_reduced(
    state: state_module.State,
    evolving_names: tuple[str, ...],
    geo: geometry.Geometry,
) -> block_1d_coeffs.Block1DCoeffs:
  """Calculates only the transient_in_cell terms in Block1DCoeffs."""

  # Only transient_in_cell is used for explicit terms if theta_imp=1
  tic_temp_ion = state.ni.value
  tic_temp_el = state.ne.value
  tic_psi = jnp.ones_like(geo.vpr)
  tic_dens_el = jnp.ones_like(geo.vpr)

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


def _populate_aux_outputs_with_ion_el_heat_sources(
    implicit_source_profiles: source_profiles_lib.SourceProfiles,
    explicit_source_profiles: source_profiles_lib.SourceProfiles,
    qei: qei_source_lib.QeiInfo,
    qei_state: state_module.State,
    aux_outputs: AuxOutput,
) -> None:
  """Observes the values of certain ion and electron heat sources."""
  # For generic and fusion, only one of the implicit or explicit will be
  # non-zero, so it's ok to sum them.
  # The following makes the assumption that the generic and fusion heat sources
  # are included in the TORAX sources, even if they are set to 0.
  generic_ion = (
      implicit_source_profiles.profiles['generic_ion_el_heat_source'][0, ...]
      + explicit_source_profiles.profiles['generic_ion_el_heat_source'][0, ...]
  )
  generic_el = (
      implicit_source_profiles.profiles['generic_ion_el_heat_source'][1, ...]
      + explicit_source_profiles.profiles['generic_ion_el_heat_source'][1, ...]
  )
  fusion_ion = (
      implicit_source_profiles.profiles['fusion_heat_source'][0, ...]
      + explicit_source_profiles.profiles['fusion_heat_source'][0, ...]
  )
  fusion_el = (
      implicit_source_profiles.profiles['fusion_heat_source'][1, ...]
      + explicit_source_profiles.profiles['fusion_heat_source'][1, ...]
  )
  ohmic = (
      implicit_source_profiles.profiles['ohmic_heat_source']
      + explicit_source_profiles.profiles['ohmic_heat_source']
  )
  aux_outputs.source_ion = generic_ion
  aux_outputs.source_el = generic_el
  aux_outputs.Pfus_i = fusion_ion
  aux_outputs.Pfus_e = fusion_el
  aux_outputs.Pohm = ohmic
  aux_outputs.Qei = qei.qei_coef * (
      qei_state.temp_el.value - qei_state.temp_ion.value
  )

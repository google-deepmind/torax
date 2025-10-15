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
import functools

import jax
import jax.numpy as jnp
from torax._src import constants
from torax._src import physics_models as physics_models_lib
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.core_profiles import convertors
from torax._src.core_profiles import updaters
from torax._src.fvm import block_1d_coeffs
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.sources import source_profile_builders
from torax._src.sources import source_profiles as source_profiles_lib
import typing_extensions


# pylint: disable=invalid-name
class CoeffsCallback:
  """Calculates Block1DCoeffs for a state."""

  def __init__(
      self,
      physics_models: physics_models_lib.PhysicsModels,
      evolving_names: tuple[str, ...],
  ):
    self.physics_models = physics_models
    self.evolving_names = evolving_names

  def __hash__(self) -> int:
    return hash((
        self.physics_models,
        self.evolving_names,
    ))

  def __eq__(self, other: typing_extensions.Self) -> bool:
    return (
        self.physics_models == other.physics_models
        and self.evolving_names == other.evolving_names
    )

  def __call__(
      self,
      runtime_params: runtime_params_slice.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      x: tuple[cell_variable.CellVariable, ...],
      explicit_source_profiles: source_profiles_lib.SourceProfiles,
      allow_pereverzev: bool = False,
      # Checks if reduced calc_coeffs for explicit terms when theta_implicit=1
      # should be called
      explicit_call: bool = False,
  ) -> block_1d_coeffs.Block1DCoeffs:
    """Returns coefficients given a state x.

    Used to calculate the coefficients for the implicit or explicit components
    of the PDE system.

    Args:
      runtime_params: Runtime configuration parameters. These values are
        potentially time-dependent and should correspond to the time step of the
        state x.
      geo: The geometry of the system at this time step.
      core_profiles: The core profiles of the system at this time step.
      x: The state with cell-grid values of the evolving variables.
      explicit_source_profiles: Precomputed explicit source profiles. These
        profiles were configured to always depend on state and parameters at
        time t during the solver step. They can thus be inputs, since they are
        not recalculated at time t+plus_dt with updated state during the solver
        iterations. For sources that are implicit, their explicit profiles are
        set to all zeros.
      allow_pereverzev: If True, then the coeffs are being called within a
        linear solver. Thus could be either the use_predictor_corrector solver
        or as part of calculating the initial guess for the nonlinear solver. In
        either case, we allow the inclusion of Pereverzev-Corrigan terms which
        aim to stabilize the linear solver when being used with highly nonlinear
        (stiff) transport coefficients. The nonlinear solver solves the system
        more rigorously and Pereverzev-Corrigan terms are not needed.
      explicit_call: If True, then if theta_implicit=1, only a reduced
        Block1DCoeffs is calculated since most explicit coefficients will not be
        used.

    Returns:
      coeffs: The diffusion, convection, etc. coefficients for this state.
    """

    # Update core_profiles with the subset of new values of evolving variables
    core_profiles = updaters.update_core_profiles_during_step(
        x,
        runtime_params,
        geo,
        core_profiles,
        self.evolving_names,
    )
    if allow_pereverzev:
      use_pereverzev = runtime_params.solver.use_pereverzev
    else:
      use_pereverzev = False

    return calc_coeffs(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        explicit_source_profiles=explicit_source_profiles,
        physics_models=self.physics_models,
        evolving_names=self.evolving_names,
        use_pereverzev=use_pereverzev,
        explicit_call=explicit_call,
    )


def _calculate_pereverzev_flux(
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Adds Pereverzev-Corrigan flux to diffusion terms."""

  consts = constants.CONSTANTS

  geo_factor = jnp.concatenate(
      [jnp.ones(1), geo.g1_over_vpr_face[1:] / geo.g0_face[1:]]
  )

  chi_face_per_ion = (
      geo.g1_over_vpr_face
      * core_profiles.n_i.face_value()
      * consts.keV_to_J
      * runtime_params.solver.chi_pereverzev
  )

  chi_face_per_el = (
      geo.g1_over_vpr_face
      * core_profiles.n_e.face_value()
      * consts.keV_to_J
      * runtime_params.solver.chi_pereverzev
  )

  d_face_per_el = runtime_params.solver.D_pereverzev
  v_face_per_el = (
      core_profiles.n_e.face_grad()
      / core_profiles.n_e.face_value()
      * d_face_per_el
      * geo_factor
  )

  # remove Pereverzev flux from boundary region if pedestal model is on
  # (for PDE stability)
  chi_face_per_ion = jnp.where(
      geo.rho_face_norm > pedestal_model_output.rho_norm_ped_top,
      0.0,
      chi_face_per_ion,
  )
  chi_face_per_el = jnp.where(
      geo.rho_face_norm > pedestal_model_output.rho_norm_ped_top,
      0.0,
      chi_face_per_el,
  )

  # set heat convection terms to zero out Pereverzev-Corrigan heat diffusion
  v_heat_face_ion = (
      core_profiles.T_i.face_grad()
      / core_profiles.T_i.face_value()
      * chi_face_per_ion
  )
  v_heat_face_el = (
      core_profiles.T_e.face_grad()
      / core_profiles.T_e.face_value()
      * chi_face_per_el
  )

  d_face_per_el = jnp.where(
      geo.rho_face_norm > pedestal_model_output.rho_norm_ped_top,
      0.0,
      d_face_per_el * geo.g1_over_vpr_face,
  )

  v_face_per_el = jnp.where(
      geo.rho_face_norm > pedestal_model_output.rho_norm_ped_top,
      0.0,
      v_face_per_el * geo.g0_face,
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
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    explicit_source_profiles: source_profiles_lib.SourceProfiles,
    physics_models: physics_models_lib.PhysicsModels,
    evolving_names: tuple[str, ...],
    use_pereverzev: bool = False,
    explicit_call: bool = False,
) -> block_1d_coeffs.Block1DCoeffs:
  """Calculates Block1DCoeffs for the time step described by `core_profiles`.

  Args:
    runtime_params: General input parameters that can change from time step to
      time step or simulation run to run, and do so without triggering a
      recompile.
    geo: Geometry describing the torus.
    core_profiles: Core plasma profiles for this time step during this iteration
      of the solver. Depending on the type of solver being used, this may or may
      not be equal to the original plasma profiles at the beginning of the time
      step.
    explicit_source_profiles: Precomputed explicit source profiles. These
      profiles either do not depend on the core profiles or depend on the
      original core profiles at the start of the time step, not the "live"
      updating core profiles. For sources that are implicit, their explicit
      profiles are set to all zeros.
    physics_models: The physics models to use for the simulation.
    evolving_names: The names of the evolving variables in the order that their
      coefficients should be written to `coeffs`.
    use_pereverzev: Toggle whether to calculate Pereverzev terms
    explicit_call: If True, indicates that calc_coeffs is being called for the
      explicit component of the PDE. Then calculates a reduced Block1DCoeffs if
      theta_implicit=1. This saves computation for the default fully implicit
      implementation.

  Returns:
    coeffs: Block1DCoeffs containing the coefficients at this time step.
  """

  # If we are fully implicit and we are making a call for calc_coeffs for the
  # explicit components of the PDE, only return a cheaper reduced Block1DCoeffs
  if explicit_call and runtime_params.solver.theta_implicit == 1.0:
    return _calc_coeffs_reduced(
        geo,
        core_profiles,
        evolving_names,
    )
  else:
    return _calc_coeffs_full(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        explicit_source_profiles=explicit_source_profiles,
        physics_models=physics_models,
        evolving_names=evolving_names,
        use_pereverzev=use_pereverzev,
    )


@functools.partial(
    jax.jit,
    static_argnames=[
        'physics_models',
        'evolving_names',
    ],
)
def _calc_coeffs_full(
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    explicit_source_profiles: source_profiles_lib.SourceProfiles,
    physics_models: physics_models_lib.PhysicsModels,
    evolving_names: tuple[str, ...],
    use_pereverzev: bool = False,
) -> block_1d_coeffs.Block1DCoeffs:
  """See `calc_coeffs` for details."""

  consts = constants.CONSTANTS

  pedestal_model_output = physics_models.pedestal_model(
      runtime_params, geo, core_profiles
  )

  # Boolean mask for enforcing internal temperature boundary conditions to
  # model the pedestal.
  # If rho_norm_ped_top_idx is outside of bounds of the mesh, the pedestal is
  # not present and the mask is all False. This is what is used in the case that
  # set_pedestal is False.
  mask = (
      jnp.zeros_like(geo.rho, dtype=bool)
      .at[pedestal_model_output.rho_norm_ped_top_idx]
      .set(True)
  )

  conductivity = (
      physics_models.neoclassical_models.conductivity.calculate_conductivity(
          geo, core_profiles
      )
  )

  # Calculate the implicit source profiles and combines with the explicit
  merged_source_profiles = source_profile_builders.build_source_profiles(
      source_models=physics_models.source_models,
      neoclassical_models=physics_models.neoclassical_models,
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      explicit=False,
      explicit_source_profiles=explicit_source_profiles,
      conductivity=conductivity,
  )

  # psi source terms. Source matrix is zero for all psi sources
  source_mat_psi = jnp.zeros_like(geo.rho)

  # fill source vector based on both original and updated core profiles
  source_psi = merged_source_profiles.total_psi_sources(geo)

  # Transient term coefficient vector (has radial dependence through r, n)
  toc_T_i = 1.5 * geo.vpr ** (-2.0 / 3.0) * consts.keV_to_J
  tic_T_i = core_profiles.n_i.value * geo.vpr ** (5.0 / 3.0)
  toc_T_e = 1.5 * geo.vpr ** (-2.0 / 3.0) * consts.keV_to_J
  tic_T_e = core_profiles.n_e.value * geo.vpr ** (5.0 / 3.0)
  toc_psi = (
      1.0
      / runtime_params.numerics.resistivity_multiplier
      * geo.rho_norm
      * conductivity.sigma
      * consts.mu_0
      * 16
      * jnp.pi**2
      * geo.Phi_b**2
      / geo.F**2
  )
  tic_psi = jnp.ones_like(toc_psi)
  toc_dens_el = jnp.ones_like(geo.vpr)
  tic_dens_el = geo.vpr

  # Diffusion term coefficients
  turbulent_transport = physics_models.transport_model(
      runtime_params, geo, core_profiles, pedestal_model_output
  )
  neoclassical_transport = physics_models.neoclassical_models.transport(
      runtime_params, geo, core_profiles
  )

  chi_face_ion_total = (
      turbulent_transport.chi_face_ion + neoclassical_transport.chi_neo_i
  )
  chi_face_el_total = (
      turbulent_transport.chi_face_el + neoclassical_transport.chi_neo_e
  )
  d_face_el_total = (
      turbulent_transport.d_face_el + neoclassical_transport.D_neo_e
  )
  v_face_el_total = (
      turbulent_transport.v_face_el
      + neoclassical_transport.V_neo_e
      + neoclassical_transport.V_neo_ware_e
  )
  d_face_psi = geo.g2g3_over_rhon_face
  # No poloidal flux convection term
  v_face_psi = jnp.zeros_like(d_face_psi)

  # entire coefficient preceding dT/dr in heat transport equations
  full_chi_face_ion = (
      geo.g1_over_vpr_face
      * core_profiles.n_i.face_value()
      * consts.keV_to_J
      * chi_face_ion_total
  )
  full_chi_face_el = (
      geo.g1_over_vpr_face
      * core_profiles.n_e.face_value()
      * consts.keV_to_J
      * chi_face_el_total
  )

  # entire coefficient preceding dne/dr in particle equation
  full_d_face_el = geo.g1_over_vpr_face * d_face_el_total
  full_v_face_el = geo.g0_face * v_face_el_total

  # density source terms. Initialize source matrix to zero
  source_mat_nn = jnp.zeros_like(geo.rho)

  # density source vector based both on original and updated core profiles
  source_n_e = merged_source_profiles.total_sources('n_e', geo)

  source_n_e += (
      mask
      * runtime_params.numerics.adaptive_n_source_prefactor
      * pedestal_model_output.n_e_ped
  )
  source_mat_nn += -(mask * runtime_params.numerics.adaptive_n_source_prefactor)

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
      lambda: _calculate_pereverzev_flux(
          runtime_params,
          geo,
          core_profiles,
          pedestal_model_output,
      ),
      lambda: tuple([jnp.zeros_like(geo.rho_face)] * 6),
  )

  full_chi_face_ion += chi_face_per_ion
  full_chi_face_el += chi_face_per_el
  full_d_face_el += d_face_per_el
  full_v_face_el += v_face_per_el

  # Add Phi_b_dot terms to heat transport convection
  v_heat_face_ion += (
      -3.0
      / 4.0
      * geo.Phi_b_dot
      / geo.Phi_b
      * geo.rho_face_norm
      * geo.vpr_face
      * core_profiles.n_i.face_value()
      * consts.keV_to_J
  )

  v_heat_face_el += (
      -3.0
      / 4.0
      * geo.Phi_b_dot
      / geo.Phi_b
      * geo.rho_face_norm
      * geo.vpr_face
      * core_profiles.n_e.face_value()
      * consts.keV_to_J
  )

  # Add Phi_b_dot terms to particle transport convection
  full_v_face_el += (
      -1.0 / 2.0 * geo.Phi_b_dot / geo.Phi_b * geo.rho_face_norm * geo.vpr_face
  )

  # Fill heat transport equation sources. Initialize source matrices to zero

  source_i = merged_source_profiles.total_sources('T_i', geo)
  source_e = merged_source_profiles.total_sources('T_e', geo)

  # Add the Qei effects.
  qei = merged_source_profiles.qei
  source_mat_ii = qei.implicit_ii * geo.vpr
  source_i += qei.explicit_i * geo.vpr
  source_mat_ee = qei.implicit_ee * geo.vpr
  source_e += qei.explicit_e * geo.vpr
  source_mat_ie = qei.implicit_ie * geo.vpr
  source_mat_ei = qei.implicit_ei * geo.vpr

  # Pedestal
  source_i += (
      mask
      * runtime_params.numerics.adaptive_T_source_prefactor
      * pedestal_model_output.T_i_ped
  )
  source_e += (
      mask
      * runtime_params.numerics.adaptive_T_source_prefactor
      * pedestal_model_output.T_e_ped
  )

  source_mat_ii -= mask * runtime_params.numerics.adaptive_T_source_prefactor

  source_mat_ee -= mask * runtime_params.numerics.adaptive_T_source_prefactor

  # Add effective Phi_b_dot heat source terms

  d_vpr53_rhon_n_e_drhon = jnp.gradient(
      geo.vpr ** (5.0 / 3.0) * geo.rho_norm * core_profiles.n_e.value,
      geo.rho_norm,
  )
  d_vpr53_rhon_n_i_drhon = jnp.gradient(
      geo.vpr ** (5.0 / 3.0) * geo.rho_norm * core_profiles.n_i.value,
      geo.rho_norm,
  )

  source_i += (
      3.0
      / 4.0
      * geo.vpr ** (-2.0 / 3.0)
      * d_vpr53_rhon_n_i_drhon
      * geo.Phi_b_dot
      / geo.Phi_b
      * core_profiles.T_i.value
      * consts.keV_to_J
  )

  source_e += (
      3.0
      / 4.0
      * geo.vpr ** (-2.0 / 3.0)
      * d_vpr53_rhon_n_e_drhon
      * geo.Phi_b_dot
      / geo.Phi_b
      * core_profiles.T_e.value
      * consts.keV_to_J
  )

  d_vpr_rhon_drhon = jnp.gradient(geo.vpr * geo.rho_norm, geo.rho_norm)

  # Add effective Phi_b_dot particle source terms
  source_n_e += (
      1.0
      / 2.0
      * d_vpr_rhon_drhon
      * geo.Phi_b_dot
      / geo.Phi_b
      * core_profiles.n_e.value
  )

  # Add effective Phi_b_dot poloidal flux source term
  source_psi += (
      8.0
      * jnp.pi**2
      * consts.mu_0
      * geo.Phi_b_dot
      * geo.Phi_b
      * geo.rho_norm**2
      * conductivity.sigma
      / geo.F**2
      * core_profiles.psi.grad()
  )

  # Build arguments to solver based on which variables are evolving
  var_to_toc = {
      'T_i': toc_T_i,
      'T_e': toc_T_e,
      'psi': toc_psi,
      'n_e': toc_dens_el,
  }
  var_to_tic = {
      'T_i': tic_T_i,
      'T_e': tic_T_e,
      'psi': tic_psi,
      'n_e': tic_dens_el,
  }
  transient_out_cell = tuple(var_to_toc[var] for var in evolving_names)
  transient_in_cell = tuple(var_to_tic[var] for var in evolving_names)

  var_to_d_face = {
      'T_i': full_chi_face_ion,
      'T_e': full_chi_face_el,
      'psi': d_face_psi,
      'n_e': full_d_face_el,
  }
  d_face = tuple(var_to_d_face[var] for var in evolving_names)

  var_to_v_face = {
      'T_i': v_heat_face_ion,
      'T_e': v_heat_face_el,
      'psi': v_face_psi,
      'n_e': full_v_face_el,
  }
  v_face = tuple(var_to_v_face.get(var) for var in evolving_names)

  # d maps (row var, col var) to the coefficient for that block of the matrix
  # (Can't use a descriptive name or the nested comprehension to build the
  # matrix gets too long)
  d = {
      ('T_i', 'T_i'): source_mat_ii,
      ('T_i', 'T_e'): source_mat_ie,
      ('T_e', 'T_i'): source_mat_ei,
      ('T_e', 'T_e'): source_mat_ee,
      ('n_e', 'n_e'): source_mat_nn,
      ('psi', 'psi'): source_mat_psi,
  }
  source_mat_cell = tuple(
      tuple(d.get((row_block, col_block)) for col_block in evolving_names)
      for row_block in evolving_names
  )

  # var_to_source ends up as a vector in the constructed PDE. Therefore any
  # scalings from CoreProfiles state variables to x must be applied here too.
  var_to_source = {
      'T_i': source_i / convertors.SCALING_FACTORS['T_i'],
      'T_e': source_e / convertors.SCALING_FACTORS['T_e'],
      'psi': source_psi / convertors.SCALING_FACTORS['psi'],
      'n_e': source_n_e / convertors.SCALING_FACTORS['n_e'],
  }
  source_cell = tuple(var_to_source.get(var) for var in evolving_names)

  coeffs = block_1d_coeffs.Block1DCoeffs(
      transient_out_cell=transient_out_cell,
      transient_in_cell=transient_in_cell,
      d_face=d_face,
      v_face=v_face,
      source_mat_cell=source_mat_cell,
      source_cell=source_cell,
  )

  return coeffs


@functools.partial(
    jax.jit,
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

  # Only transient_in_cell is used for explicit terms if theta_implicit=1
  tic_T_i = core_profiles.n_i.value * geo.vpr ** (5.0 / 3.0)
  tic_T_e = core_profiles.n_e.value * geo.vpr ** (5.0 / 3.0)
  tic_psi = jnp.ones_like(geo.vpr)
  tic_dens_el = geo.vpr

  var_to_tic = {
      'T_i': tic_T_i,
      'T_e': tic_T_e,
      'psi': tic_psi,
      'n_e': tic_dens_el,
  }
  transient_in_cell = tuple(var_to_tic[var] for var in evolving_names)

  coeffs = block_1d_coeffs.Block1DCoeffs(
      transient_in_cell=transient_in_cell,
  )
  return coeffs

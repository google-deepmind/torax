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

import dataclasses
import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import models as models_lib
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import convertors
from torax._src.core_profiles import updaters
from torax._src.fvm import block_1d_coeffs
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.internal_boundary_conditions import internal_boundary_conditions as internal_boundary_conditions_lib
from torax._src.pedestal_model import pedestal_model_output as pedestal_model_output_lib
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib
from torax._src.sources import source_profile_builders
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.transport_model import transport_coefficients_builder
import typing_extensions


# pylint: disable=invalid-name
class CoeffsCallback:
  """Calculates Block1DCoeffs for a state."""

  def __init__(
      self,
      models: models_lib.Models,
      evolving_names: tuple[str, ...],
  ):
    self.models = models
    self.evolving_names = evolving_names

  def __hash__(self) -> int:
    return hash((
        self.models,
        self.evolving_names,
    ))

  def __eq__(self, other: typing_extensions.Self) -> bool:
    return (
        self.models == other.models
        and self.evolving_names == other.evolving_names
    )

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      prev_core_profiles: state.CoreProfiles | None,
      dt: array_typing.FloatScalar | None,
      x: tuple[cell_variable.CellVariable, ...],
      explicit_source_profiles: source_profiles_lib.SourceProfiles,
      allow_pereverzev: bool = False,
      # Checks if reduced calc_coeffs for explicit terms when theta_implicit=1
      # should be called
      explicit_call: bool = False,
      pedestal_transition_state: (
          pedestal_transition_state_lib.PedestalTransitionState | None
      ) = None,
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
      prev_core_profiles: The core profiles of the system at the previous time
        step.
      dt: The time step size.
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
      pedestal_transition_state: State for tracking pedestal L-H and H-L
        transitions. Only used when the pedestal mode is ADAPTIVE_SOURCE with
        use_formation_model_with_adaptive_source=True. None otherwise.

    Returns:
      coeffs: The diffusion, convection, etc. coefficients for this state.
    """

    # Update core_profiles with the subset of new values of evolving variables
    core_profiles = updaters.update_core_profiles_during_step(
        x,
        runtime_params,
        geo,
        core_profiles=core_profiles,
        prev_core_profiles=prev_core_profiles,
        dt=dt,
        evolving_names=self.evolving_names,
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
        models=self.models,
        evolving_names=self.evolving_names,
        use_pereverzev=use_pereverzev,
        explicit_call=explicit_call,
        pedestal_transition_state=pedestal_transition_state,
    )


def calc_coeffs(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    explicit_source_profiles: source_profiles_lib.SourceProfiles,
    models: models_lib.Models,
    evolving_names: tuple[str, ...],
    use_pereverzev: bool = False,
    explicit_call: bool = False,
    pedestal_transition_state: (
        pedestal_transition_state_lib.PedestalTransitionState | None
    ) = None,
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
    models: The models to use for the simulation.
    evolving_names: The names of the evolving variables in the order that their
      coefficients should be written to `coeffs`.
    use_pereverzev: Toggle whether to calculate Pereverzev terms
    explicit_call: If True, indicates that calc_coeffs is being called for the
      explicit component of the PDE. Then calculates a reduced Block1DCoeffs if
      theta_implicit=1. This saves computation for the default fully implicit
      implementation.
    pedestal_transition_state: State for tracking pedestal L-H and H-L
      transitions. Only used when the pedestal mode is ADAPTIVE_SOURCE with
      use_formation_model_with_adaptive_source=True. None otherwise.

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
        models=models,
        evolving_names=evolving_names,
        use_pereverzev=use_pereverzev,
        pedestal_transition_state=pedestal_transition_state,
    )


@jax.jit(
    static_argnames=[
        'models',
        'evolving_names',
    ],
)
def _calc_coeffs_full(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    explicit_source_profiles: source_profiles_lib.SourceProfiles,
    models: models_lib.Models,
    evolving_names: tuple[str, ...],
    use_pereverzev: bool = False,
    pedestal_transition_state: (
        pedestal_transition_state_lib.PedestalTransitionState | None
    ) = None,
) -> block_1d_coeffs.Block1DCoeffs:
  """See `calc_coeffs` for details."""

  consts = constants.CONSTANTS

  conductivity = models.neoclassical_models.conductivity.calculate_conductivity(
      geo, core_profiles
  )

  # Calculate the implicit source profiles and combine them with the explicit
  # source profiles. These are needed for the pedestal model, so are computed
  # here rather than in the source terms section.
  merged_source_profiles = source_profile_builders.build_source_profiles(
      source_models=models.source_models,
      neoclassical_models=models.neoclassical_models,
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      explicit=False,
      explicit_source_profiles=explicit_source_profiles,
      conductivity=conductivity,
  )

  # --- Transient term coefficients --- #
  # These have radial dependence through r, n
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

  # --- Diffusion and convection term coefficients --- #
  # 1. Compute transport coefficients from all models.
  transport_coefficients = (
      transport_coefficients_builder.calculate_all_transport_coeffs(
          models.pedestal_model,
          models.transport_model,
          models.neoclassical_models,
          runtime_params,
          geo,
          core_profiles,
          merged_source_profiles,
          use_pereverzev,
          pedestal_transition_state=pedestal_transition_state,
      )
  )

  # Transport coefficients for the psi equation don't come from models
  d_face_psi = geo.g2g3_over_rhon_face
  v_face_psi = jnp.zeros_like(d_face_psi)

  # 2. Convert to "full" coefficients, i.e. the entire coefficient preceding the
  # gradient term (dT/dr, dn/dr, etc.) in each equation.
  # Heat equations
  full_chi_face_ion = (
      geo.g1_over_vpr_face
      * core_profiles.n_i.face_value()
      * consts.keV_to_J
      * transport_coefficients.chi_face_ion_total
  )
  full_chi_face_el = (
      geo.g1_over_vpr_face
      * core_profiles.n_e.face_value()
      * consts.keV_to_J
      * transport_coefficients.chi_face_el_total
  )
  # PereverzevTransport convection terms are already "full" coefficients, and
  # no heat convection terms come from other models.
  full_v_heat_face_ion = transport_coefficients.full_v_heat_face_ion_pereverzev
  full_v_heat_face_el = transport_coefficients.full_v_heat_face_el_pereverzev

  # Particle equations
  full_d_face_el = geo.g1_over_vpr_face * transport_coefficients.d_face_el_total
  full_v_face_el = geo.g0_face * transport_coefficients.v_face_el_total

  # 3. Add Phi_b_dot terms to convection equations.
  # Psi equation doesn't include Phi_b_dot term.
  # Heat equations
  full_v_heat_face_ion += (
      -3.0
      / 4.0
      * geo.Phi_b_dot
      / geo.Phi_b
      * geo.rho_face_norm
      * geo.vpr_face
      * core_profiles.n_i.face_value()
      * consts.keV_to_J
  )
  full_v_heat_face_el += (
      -3.0
      / 4.0
      * geo.Phi_b_dot
      / geo.Phi_b
      * geo.rho_face_norm
      * geo.vpr_face
      * core_profiles.n_e.face_value()
      * consts.keV_to_J
  )

  # Particle equations
  full_v_face_el += (
      -1.0 / 2.0 * geo.Phi_b_dot / geo.Phi_b * geo.rho_face_norm * geo.vpr_face
  )

  # --- Source terms --- #
  # 1. Construct the source vectors
  source_i = merged_source_profiles.total_sources('T_i', geo)
  source_e = merged_source_profiles.total_sources('T_e', geo)
  source_n_e = merged_source_profiles.total_sources('n_e', geo)
  source_psi = merged_source_profiles.total_psi_sources(geo)

  # 2. Initialize source matrices to zero
  # We don't initialize heat source matrices because they are populated by the
  # Qei terms later
  source_mat_nn = jnp.zeros_like(geo.rho)
  source_mat_psi = jnp.zeros_like(geo.rho)

  # 3. Add Qei effects to the heat sources.
  qei = merged_source_profiles.qei
  source_mat_ii = qei.implicit_ii * geo.vpr
  source_i += qei.explicit_i * geo.vpr
  source_mat_ee = qei.implicit_ee * geo.vpr
  source_e += qei.explicit_e * geo.vpr
  source_mat_ie = qei.implicit_ie * geo.vpr
  source_mat_ei = qei.implicit_ei * geo.vpr

  # 4. Add effective Phi_b_dot terms
  # Heat equations
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

  # Particle equations
  d_vpr_rhon_drhon = jnp.gradient(geo.vpr * geo.rho_norm, geo.rho_norm)
  source_n_e += (
      1.0
      / 2.0
      * d_vpr_rhon_drhon
      * geo.Phi_b_dot
      / geo.Phi_b
      * core_profiles.n_e.value
  )

  # Magnetic flux equation
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

  # 5. Add internal boundary condition source terms
  if (
      runtime_params.pedestal.mode
      == pedestal_runtime_params_lib.Mode.ADAPTIVE_SOURCE
  ):
    # TODO(b/500260959): Currently pedestal model is called twice, once in
    # calculate_all_transport_coeffs and once here.
    pedestal_model_output = models.pedestal_model(
        runtime_params, geo, core_profiles, merged_source_profiles
    )
    if runtime_params.pedestal.use_formation_model_with_adaptive_source:
      assert pedestal_transition_state is not None, (
          'pedestal_transition_state must not be None when'
          ' use_formation_model_with_adaptive_source is True.'
      )
      ramp_fraction = _compute_ramp_fraction(
          pedestal_transition_state=pedestal_transition_state,
          transition_time_width=runtime_params.pedestal.transition_time_width,
          t=runtime_params.t,
      )
      # Scale the pedestal output by the ramp fraction during transitions.
      # In H-mode, returns full H-mode values. In L-mode, returns L-mode
      # values. During transitions, linearly interpolates between the two.
      pedestal_model_output = _apply_transition_ramp_scaling(
          pedestal_model_output=pedestal_model_output,
          pedestal_transition_state=pedestal_transition_state,
          ramp_fraction=ramp_fraction,
      )
      # Adaptive source should be applied if we're in H mode or still in the
      # LH/HL ramp.
      apply_adaptive_source = (
          pedestal_transition_state.confinement_mode
          != pedestal_transition_state_lib.ConfinementMode.L_MODE
      )
    else:
      apply_adaptive_source = jnp.bool_(True)

    def _apply_source():
      pedestal_internal_boundary_conditions = (
          pedestal_model_output.to_internal_boundary_conditions(geo)
      )
      return internal_boundary_conditions_lib.apply_adaptive_source(
          source_T_i=source_i,
          source_T_e=source_e,
          source_n_e=source_n_e,
          source_mat_ii=source_mat_ii,
          source_mat_ee=source_mat_ee,
          source_mat_nn=source_mat_nn,
          runtime_params=runtime_params,
          internal_boundary_conditions=pedestal_internal_boundary_conditions,
      )

    def _skip_source():
      return (
          source_i,
          source_e,
          source_n_e,
          source_mat_ii,
          source_mat_ee,
          source_mat_nn,
      )

    (
        source_i,
        source_e,
        source_n_e,
        source_mat_ii,
        source_mat_ee,
        source_mat_nn,
    ) = jax.lax.cond(
        apply_adaptive_source,
        _apply_source,
        _skip_source,
    )

  # --- Build arguments to solver  --- #
  # Selects only necessary coefficients based on which variables are evolving
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
      'T_i': full_v_heat_face_ion,
      'T_e': full_v_heat_face_el,
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
  source_mat_cell = tuple([
      tuple([d.get((row_block, col_block)) for col_block in evolving_names])
      for row_block in evolving_names
  ])

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


@jax.jit(
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


def _compute_ramp_fraction(
    pedestal_transition_state: pedestal_transition_state_lib.PedestalTransitionState,
    transition_time_width: array_typing.FloatScalar,
    t: array_typing.FloatScalar,
) -> array_typing.FloatScalar:
  """Computes the ramp fraction for a pedestal transition.

  Returns a value in [0, 1] representing the progress of the current
  transition. 0 means the transition just started, 1 means it is complete.

  Args:
    pedestal_transition_state: Current transition state.
    transition_time_width: Duration of the transition ramp.
    t: Current simulation time (i.e. t + dt when called from the solver).

  Returns:
    Ramp fraction clipped to [0, 1].
  """
  elapsed = t - pedestal_transition_state.transition_start_time
  fraction = elapsed / transition_time_width
  return jnp.clip(fraction, 0.0, 1.0)


def _apply_transition_ramp_scaling(
    pedestal_model_output: pedestal_model_output_lib.PedestalModelOutput,
    pedestal_transition_state: pedestal_transition_state_lib.PedestalTransitionState,
    ramp_fraction: array_typing.FloatScalar,
) -> pedestal_model_output_lib.PedestalModelOutput:
  """Applies ramp scaling to internal boundary conditions during transitions.

  During an L-H transition, linearly ramps from L-mode values to the H-mode
  targets. During an H-L transition, ramps from the H-mode targets back to
  the L-mode values.

  The L-mode values are stored in the pedestal_transition_state (captured
  at the start of an L->H transition). The H-mode targets are the full
  pedestal model output.

  Args:
    pedestal_model_output: Output from the pedestal model.
    pedestal_transition_state: Current transition state containing L-mode
      baseline values.
    ramp_fraction: Progress of the current transition, in [0, 1].

  Returns:
    Scaled pedestal model output.
  """
  def _interpolate_transition(l_val, h_val):
    """Interpolates between L-mode and H-mode values based on confinement mode.

    Args:
      l_val: L-mode baseline value.
      h_val: H-mode target value from the pedestal model output.

    Returns:
      The interpolated value based on the current confinement mode.
    """
    l_to_h_ramp = l_val + ramp_fraction * (h_val - l_val)
    h_to_l_ramp = h_val + ramp_fraction * (l_val - h_val)
    confinement_mode = pedestal_transition_state.confinement_mode
    return jnp.select(
        [
            confinement_mode
            == pedestal_transition_state_lib.ConfinementMode.L_MODE,
            confinement_mode
            == pedestal_transition_state_lib.ConfinementMode.H_MODE,
            confinement_mode
            == pedestal_transition_state_lib.ConfinementMode.TRANSITIONING_TO_H_MODE,
            confinement_mode
            == pedestal_transition_state_lib.ConfinementMode.TRANSITIONING_TO_L_MODE,
        ],
        [l_val, h_val, l_to_h_ramp, h_to_l_ramp],
    )

  scaled_T_i = _interpolate_transition(
      l_val=pedestal_transition_state.T_i_ped_L_mode,
      h_val=pedestal_model_output.T_i_ped,
  )
  scaled_T_e = _interpolate_transition(
      l_val=pedestal_transition_state.T_e_ped_L_mode,
      h_val=pedestal_model_output.T_e_ped,
  )
  scaled_n_e = _interpolate_transition(
      l_val=pedestal_transition_state.n_e_ped_L_mode,
      h_val=pedestal_model_output.n_e_ped,
  )

  return dataclasses.replace(
      pedestal_model_output,
      T_i_ped=scaled_T_i,
      T_e_ped=scaled_T_e,
      n_e_ped=scaled_n_e,
  )

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

import jax
import jax.numpy as jnp
from torax._src import constants
from torax._src import model_state as model_state_lib
from torax._src import models as models_lib
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import convertors
from torax._src.core_profiles import ion_composition
from torax._src.core_profiles import solver_state as solver_state_lib
from torax._src.fvm import block_1d_coeffs
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.orchestration import (
    explicit_model_outputs as explicit_model_outputs_lib,
)
from torax._src.orchestration import physics_pipeline
from torax._src.transport_model import transport_coeffs as transport_coeffs_lib


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

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, type(self)):
      return False
    return (
        self.models == other.models
        and self.evolving_names == other.evolving_names
    )

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      solver_state: solver_state_lib.SolverState,
      x: tuple[cell_variable.CellVariable, ...],
      explicit_model_outputs: explicit_model_outputs_lib.ExplicitModelOutputs,
      model_state: model_state_lib.ModelState,
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
      solver_state: The primary profiles of the system at this time step.
      x: The state with cell-grid values of the evolving variables.
      explicit_model_outputs: Explicit model outputs evaluated at time t.
      model_state: State of models across timesteps.
      allow_pereverzev: If True, then the coeffs are being called within a
        linear solver. Thus could be either the use_fixed_point solver or as
        part of calculating the initial guess for the nonlinear solver. In
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

    # Update solver_state with new values of evolving variables.
    updated_solver_state = convertors.solver_x_tuple_to_solver_state(
        x,
        self.evolving_names,
        solver_state,
    )
    if allow_pereverzev:
      use_pereverzev = runtime_params.solver.use_pereverzev
    else:
      use_pereverzev = False

    return calc_coeffs(
        runtime_params=runtime_params,
        geo=geo,
        solver_state=updated_solver_state,
        explicit_model_outputs=explicit_model_outputs,
        models=self.models,
        evolving_names=self.evolving_names,
        model_state=model_state,
        use_pereverzev=use_pereverzev,
        explicit_call=explicit_call,
    )


def calc_coeffs(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    solver_state: solver_state_lib.SolverState,
    explicit_model_outputs: explicit_model_outputs_lib.ExplicitModelOutputs,
    models: models_lib.Models,
    evolving_names: tuple[str, ...],
    model_state: model_state_lib.ModelState,
    use_pereverzev: bool = False,
    explicit_call: bool = False,
) -> block_1d_coeffs.Block1DCoeffs:
  """Calculates Block1DCoeffs for the given `solver_state`.

  Args:
    runtime_params: General input parameters that can change from time step to
      time step or simulation run to run, and do so without triggering a
      recompile.
    geo: Geometry describing the torus.
    solver_state: Primary profiles for this time step during this iteration of
      the solver. Depending on the type of solver being used, this may or may
      not be equal to the original plasma profiles at the beginning of the time
      step.
    explicit_model_outputs: Explicit model outputs evaluated at time t.
    models: The models to use for the simulation.
    evolving_names: The names of the evolving variables in the order that their
      coefficients should be written to `coeffs`.
    model_state: Model state carried over from the previous timestep.
    use_pereverzev: Toggle whether to calculate Pereverzev terms.
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
        runtime_params=runtime_params,
        geo=geo,
        solver_state=solver_state,
        evolving_names=evolving_names,
    )
  else:
    return _calc_coeffs_full(
        runtime_params=runtime_params,
        geo=geo,
        solver_state=solver_state,
        explicit_model_outputs=explicit_model_outputs,
        models=models,
        evolving_names=evolving_names,
        model_state=model_state,
        use_pereverzev=use_pereverzev,
    )


@jax.jit(
    static_argnames=[
        'models',
        'evolving_names',
        'use_pereverzev',
    ],
)
def _calc_coeffs_full(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    solver_state: solver_state_lib.SolverState,
    explicit_model_outputs: explicit_model_outputs_lib.ExplicitModelOutputs,
    models: models_lib.Models,
    evolving_names: tuple[str, ...],
    model_state: model_state_lib.ModelState,
    use_pereverzev: bool = False,
) -> block_1d_coeffs.Block1DCoeffs:
  """See `calc_coeffs` for details."""
  physics_outputs, _ = physics_pipeline.evaluate_step_physics(
      runtime_params=runtime_params,
      geo=geo,
      solver_state=solver_state,
      models=models,
      explicit_model_outputs=explicit_model_outputs,
      model_state=model_state,
      v_loop_lcfs=None,
      use_pereverzev=use_pereverzev,
  )
  core_profiles = physics_outputs.core_profiles
  core_sources = physics_outputs.core_sources
  transport_coefficients = physics_outputs.core_transport
  internal_boundary_conditions = physics_outputs.internal_boundary_conditions
  consts = constants.CONSTANTS

  # --- Transient term coefficients --- #
  # These have radial dependence through r, n
  toc_T_i = 1.5 * geo.vpr ** (-2.0 / 3.0) * consts.keV_to_J
  tic_T_i = core_profiles.kinetic.n_i.value * geo.vpr ** (5.0 / 3.0)
  toc_T_e = 1.5 * geo.vpr ** (-2.0 / 3.0) * consts.keV_to_J
  tic_T_e = core_profiles.kinetic.n_e.value * geo.vpr ** (5.0 / 3.0)
  toc_psi = (
      1.0
      / runtime_params.numerics.current_diffusion_time_reduction_factor
      * geo.rho_norm
      * core_profiles.conductivity.sigma
      * consts.mu_0
      * 16
      * jnp.pi**2
      * geo.Phi_b**2
      / geo.F**2
  )
  tic_psi = jnp.ones_like(toc_psi)
  toc_dens_el = jnp.ones_like(geo.vpr)
  tic_dens_el = geo.vpr

  # Transport coefficients for the psi equation don't come from models
  d_face_psi = geo.g2g3_over_rhon_face
  v_face_psi = jnp.zeros_like(d_face_psi)

  # Physical transport excludes Pereverzev-Corrigan stabilization terms.
  physical_transport = transport_coeffs_lib.sum_transport_coeffs(
      transport_coefficients.turbulent.total,
      transport_coefficients.neoclassical,
  )

  # Convert to "full" coefficients, i.e. the entire coefficient preceding the
  # gradient term (dT/dr, dn/dr, etc.) in each equation.
  # Heat equations
  full_chi_face_ion = (
      geo.g1_over_vpr_face
      * core_profiles.kinetic.n_i.face_value()
      * consts.keV_to_J
      * physical_transport.chi_face_ion
  )
  full_chi_face_el = (
      geo.g1_over_vpr_face
      * core_profiles.kinetic.n_e.face_value()
      * consts.keV_to_J
      * physical_transport.chi_face_el
  )

  # Particle equations
  full_d_face_el = geo.g1_over_vpr_face * physical_transport.d_face_el
  full_v_face_el = geo.g0_face * physical_transport.v_face_el

  # Add Phi_b_dot terms to convection equations.
  # Note:
  # - Psi equation has no Phi_b_dot term.
  # - Convection in the heat equations is solely due to Phi_b_dot, with no
  #   contribution from TransportCoefficients.
  phib_dot_factor = geo.Phi_b_dot / geo.Phi_b * geo.rho_face_norm * geo.vpr_face
  # Heat equations
  # The Phi_b_dot term for the heat equation is derived by splitting V'^(5/3) as
  # V'^(2/3) * V' and using the product derivative rule to split the total
  # Phi_b_dot proportional term into a convective term and an effective source
  # term.
  full_v_heat_face_ion = (
      -0.75
      * phib_dot_factor
      * core_profiles.kinetic.n_i.face_value()
      * consts.keV_to_J
  )
  full_v_heat_face_el = (
      -0.75
      * phib_dot_factor
      * core_profiles.kinetic.n_e.face_value()
      * consts.keV_to_J
  )
  # Particle equations. No effective source term will be needed for particles.
  full_v_face_el -= 0.5 * phib_dot_factor

  # Keep Pereverzev-Corrigan terms separate from physical convection so they
  # use central 0.5/0.5 zero-Péclet face weights rather than upwinding,
  # canceling the added Pereverzev diffusion in `discrete_system.py`.
  per = transport_coefficients.pereverzev
  if per is not None:
    d_per_ion_heat = (
        geo.g1_over_vpr_face
        * core_profiles.kinetic.n_i.face_value()
        * consts.keV_to_J
        * per.chi_face_ion
    )
    d_per_el_heat = (
        geo.g1_over_vpr_face
        * core_profiles.kinetic.n_e.face_value()
        * consts.keV_to_J
        * per.chi_face_el
    )
    v_per_el_heat = per.full_v_heat_face_el
    v_per_ion_heat = per.full_v_heat_face_ion

    d_per_el_particle = geo.g1_over_vpr_face * per.d_face_el
    v_per_el_particle = geo.g0_face * per.v_face_el
    # Pereverzev-Corrigan is not applied to psi.
    zero_face = jnp.zeros_like(geo.rho_face)
    var_to_d_per_face = {
        'T_i': d_per_ion_heat,
        'T_e': d_per_el_heat,
        'psi': zero_face,
        'n_e': d_per_el_particle,
    }
    var_to_v_per_face = {
        'T_i': v_per_ion_heat,
        'T_e': v_per_el_heat,
        'psi': zero_face,
        'n_e': v_per_el_particle,
    }
    d_pereverzev_face = tuple(var_to_d_per_face[var] for var in evolving_names)
    v_pereverzev_face = tuple(var_to_v_per_face[var] for var in evolving_names)
  else:
    d_pereverzev_face = None
    v_pereverzev_face = None

  # --- Source terms --- #
  # Construct the source vectors
  source_i = core_sources.total_sources('T_i', geo)
  source_e = core_sources.total_sources('T_e', geo)
  source_n_e = core_sources.total_sources('n_e', geo)
  source_psi = core_sources.total_psi_sources(geo)

  # Initialize source matrices to zero
  # We don't initialize heat source matrices because they are populated by the
  # Qei terms later
  source_mat_nn = jnp.zeros_like(geo.rho)
  source_mat_psi = jnp.zeros_like(geo.rho)

  # Add Qei effects to the heat sources.
  qei = core_sources.qei
  source_mat_ii = qei.implicit_ii * geo.vpr
  source_i += qei.explicit_i * geo.vpr
  source_mat_ee = qei.implicit_ee * geo.vpr
  source_e += qei.explicit_e * geo.vpr
  source_mat_ie = qei.implicit_ie * geo.vpr
  source_mat_ei = qei.implicit_ei * geo.vpr

  # Add effective Phi_b_dot source terms.
  # TODO(b/574457288): Add the missing compression term
  # 0.5 * vpr * Phi_b_dot / Phi_b * n * T to the heat sources.
  # Has a very minor impact on standard scenarios.
  source_i += (
      0.5
      * geo.rho_norm
      * core_profiles.kinetic.n_i.value
      * geo.dvpr_drho_norm
      * geo.Phi_b_dot
      / geo.Phi_b
      * core_profiles.kinetic.T_i.value
      * consts.keV_to_J
  )
  source_e += (
      0.5
      * geo.rho_norm
      * core_profiles.kinetic.n_e.value
      * geo.dvpr_drho_norm
      * geo.Phi_b_dot
      / geo.Phi_b
      * core_profiles.kinetic.T_e.value
      * consts.keV_to_J
  )

  # Magnetic flux equation
  source_psi += (
      8.0
      * jnp.pi**2
      * consts.mu_0
      * geo.Phi_b_dot
      * geo.Phi_b
      * geo.rho_norm**2
      * core_profiles.conductivity.sigma
      / runtime_params.numerics.current_diffusion_time_reduction_factor
      / geo.F**2
      * core_profiles.flux.psi.grad()
  )

  # --- Build arguments to solver  --- #
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
      'T_i': full_v_heat_face_ion,
      'T_e': full_v_heat_face_el,
      'psi': v_face_psi,
      'n_e': full_v_face_el,
  }
  v_face = tuple(var_to_v_face[var] for var in evolving_names)

  # d maps (row var, col var) to the coefficient for that block of the matrix
  # (Can't use a descriptive name or the nested comprehension to build the
  # matrix gets too long)
  d: dict[tuple[str, str], jax.Array] = {
      ('T_i', 'T_i'): jnp.asarray(source_mat_ii),
      ('T_i', 'T_e'): jnp.asarray(source_mat_ie),
      ('T_e', 'T_i'): jnp.asarray(source_mat_ei),
      ('T_e', 'T_e'): jnp.asarray(source_mat_ee),
      ('n_e', 'n_e'): jnp.asarray(source_mat_nn),
      ('psi', 'psi'): jnp.asarray(source_mat_psi),
  }
  source_mat_cell = tuple(
      tuple(d.get((row_block, col_block)) for col_block in evolving_names)
      for row_block in evolving_names
  )

  # var_to_source ends up as a vector in the constructed PDE. Therefore any
  # scalings from CoreProfiles state variables to x must be applied here too.
  var_to_source: dict[str, jax.Array] = {
      'T_i': jnp.asarray(source_i / convertors.STATE_SCALING_FACTORS['T_i']),
      'T_e': jnp.asarray(source_e / convertors.STATE_SCALING_FACTORS['T_e']),
      'psi': jnp.asarray(source_psi / convertors.STATE_SCALING_FACTORS['psi']),
      'n_e': jnp.asarray(source_n_e / convertors.STATE_SCALING_FACTORS['n_e']),
  }
  source_cell = tuple(var_to_source.get(var) for var in evolving_names)

  # Add internal boundary condition source terms
  internal_boundary_condition_mask, internal_boundary_condition_target_vec = (
      internal_boundary_conditions.to_solver_coeffs(
          evolving_names=evolving_names,
          nx=geo.torax_mesh.nx,
      )
  )

  coeffs = block_1d_coeffs.Block1DCoeffs(
      transient_out_cell=transient_out_cell,
      transient_in_cell=transient_in_cell,
      d_face=d_face,
      v_face=v_face,
      d_pereverzev_face=d_pereverzev_face,
      v_pereverzev_face=v_pereverzev_face,
      source_mat_cell=source_mat_cell,
      source_cell=source_cell,
      internal_boundary_condition_mask=internal_boundary_condition_mask,
      internal_boundary_condition_target_vec=(
          internal_boundary_condition_target_vec
      ),
  )

  return coeffs


@jax.jit(
    static_argnames=[
        'evolving_names',
    ],
)
def _calc_coeffs_reduced(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    solver_state: solver_state_lib.SolverState,
    evolving_names: tuple[str, ...],
) -> block_1d_coeffs.Block1DCoeffs:
  """Calculates only the transient_in_cell terms in Block1DCoeffs."""
  # Only transient_in_cell is used for explicit terms if theta_implicit=1
  var_to_tic = {
      'T_e': solver_state.n_e.value * geo.vpr ** (5.0 / 3.0),
      'psi': jnp.ones_like(geo.vpr),
      'n_e': geo.vpr,
  }
  if 'T_i' in evolving_names:
    _, n_i, _ = ion_composition.IonComposition.from_profiles(
        runtime_params.plasma_composition,
        geo,
        solver_state.n_e,
        solver_state.T_e,
    )
    var_to_tic['T_i'] = n_i.value * geo.vpr ** (5.0 / 3.0)

  transient_in_cell = tuple(var_to_tic[var] for var in evolving_names)

  return block_1d_coeffs.Block1DCoeffs(
      transient_in_cell=transient_in_cell,
  )

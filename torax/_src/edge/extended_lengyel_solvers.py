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
"""Solver methods for the extended Lengyel model."""

import dataclasses
import enum
import functools
import jax
from jax import numpy as jnp
from torax._src import constants
from torax._src.edge import collisional_radiative_models
from torax._src.edge import divertor_sol_1d as divertor_sol_1d_lib
from torax._src.edge import extended_lengyel_formulas
from torax._src.fvm import jax_root_finding
# pylint: disable=invalid-name

# Scale factors for physics calculations to avoid numerical issues in fp32.
_LINT_SCALE_FACTOR = 1e30
_DENSITY_SCALE_FACTOR = 1e-20

# _LINT_SCALE_FACTOR * _DENSITY_SCALE_FACTOR**2
_LINT_K_INVERSE_SCALE_FACTOR = 1e-10


class SolveStatus(enum.IntEnum):
  """Status of the _solve_for_qcc or _solve_for_c_z_prefactor calculations.

  Attributes:
    SUCCESS: The calculation was successful.
    Q_DIV_SQUARED_NEGATIVE: q_div_squared was negative. This is unphysical and
      indicates that the required power loss is too low for the given plasma
      parameters. This can happen if the power lost in the cc region is
      sufficient to reach detachment even no seeded impurities.
    QCC_SQUARED_NEGATIVE: qcc_squared was negative. This is unphysical and
      indicates that so much power has been lost such that the target
      temperature would be negative according to the formulations used in the
      model.
  """

  SUCCESS = 0
  Q_DIV_SQUARED_NEGATIVE = 1
  Q_CC_SQUARED_NEGATIVE = 2


def inverse_mode_fixed_step_solver(
    sol_model: divertor_sol_1d_lib.DivertorSOL1D,
    iterations: int,
) -> divertor_sol_1d_lib.DivertorSOL1D:
  """Runs the fixed-step iterative solver for the inverse mode."""

  def body_fun(_, current_sol_model):

    current_sol_model.state.q_parallel = extended_lengyel_formulas.calculate_q_parallel(
        separatrix_electron_temp=current_sol_model.separatrix_electron_temp,
        average_ion_mass=current_sol_model.params.average_ion_mass,
        separatrix_average_poloidal_field=current_sol_model.params.separatrix_average_poloidal_field,
        alpha_t=current_sol_model.state.alpha_t,
        ratio_of_upstream_to_average_poloidal_field=current_sol_model.params.ratio_of_upstream_to_average_poloidal_field,
        fraction_of_PSOL_to_divertor=current_sol_model.params.fraction_of_P_SOL_to_divertor,
        minor_radius=current_sol_model.params.minor_radius,
        major_radius=current_sol_model.params.major_radius,
        power_crossing_separatrix=current_sol_model.params.power_crossing_separatrix,
        fieldline_pitch_at_omp=current_sol_model.params.fieldline_pitch_at_omp,
    )

    # Solve for the impurity concentration required to achieve the target
    # temperature for a given q_parallel. This also updates the divertor and
    # separatrix Z_eff values in sol_model, used downstream.
    current_sol_model.state.c_z_prefactor, _ = _solve_for_c_z_prefactor(
        sol_model=current_sol_model
    )

    # Update alpha_t for the next loop iteration.
    current_sol_model.state.alpha_t = extended_lengyel_formulas.calc_alpha_t(
        separatrix_electron_density=current_sol_model.params.separatrix_electron_density,
        separatrix_electron_temp=current_sol_model.separatrix_electron_temp
        / 1e3,
        cylindrical_safety_factor=current_sol_model.params.cylindrical_safety_factor,
        major_radius=current_sol_model.params.major_radius,
        average_ion_mass=current_sol_model.params.average_ion_mass,
        Z_eff=current_sol_model.separatrix_Z_eff,
        mean_ion_charge_state=1.0,
    )

    # Update kappa_e for the next loop iteration.
    current_sol_model.state.kappa_e = extended_lengyel_formulas.calc_kappa_e(
        current_sol_model.divertor_Z_eff
    )

    # Returning the updated-in-place current_sol_model.
    return current_sol_model

  sol_model = jax.lax.fori_loop(
      lower=0, upper=iterations, body_fun=body_fun, init_val=sol_model
  )
  return sol_model


def forward_mode_fixed_step_solver(
    sol_model: divertor_sol_1d_lib.DivertorSOL1D,
    iterations: int,
) -> divertor_sol_1d_lib.DivertorSOL1D:
  """Runs the fixed-step iterative solver for the forward mode."""

  # Relaxation function needed for fixed point iteration in forward mode for
  # stability.
  def _relax(new_value, prev_value, relaxation_factor=0.4):
    return relaxation_factor * new_value + (1 - relaxation_factor) * prev_value

  def body_fun(i, current_sol_model):

    # Store current values for the next relaxation step
    prev_sol_model = current_sol_model

    # Update q_parallel based on the current separatrix temperature and alpha_t.
    current_sol_model.state.q_parallel = extended_lengyel_formulas.calculate_q_parallel(
        separatrix_electron_temp=current_sol_model.separatrix_electron_temp,
        average_ion_mass=current_sol_model.params.average_ion_mass,
        separatrix_average_poloidal_field=current_sol_model.params.separatrix_average_poloidal_field,
        alpha_t=current_sol_model.state.alpha_t,
        ratio_of_upstream_to_average_poloidal_field=current_sol_model.params.ratio_of_upstream_to_average_poloidal_field,
        fraction_of_PSOL_to_divertor=current_sol_model.params.fraction_of_P_SOL_to_divertor,
        minor_radius=current_sol_model.params.minor_radius,
        major_radius=current_sol_model.params.major_radius,
        power_crossing_separatrix=current_sol_model.params.power_crossing_separatrix,
        fieldline_pitch_at_omp=current_sol_model.params.fieldline_pitch_at_omp,
    )

    # Calculate heat flux at the cc-interface for fixed impurity concentrations.
    new_q_cc, _ = _solve_for_qcc(sol_model=current_sol_model)

    # Calculate new target electron temperature with forward two-point model.
    current_sol_model.state.target_electron_temp = divertor_sol_1d_lib.calc_target_electron_temp(
        sol_model=current_sol_model,
        parallel_heat_flux_at_cc_interface=new_q_cc,
        previous_target_electron_temp=current_sol_model.state.target_electron_temp,
    )

    # Update kappa_e and alpha_t for the next iteration.
    current_sol_model.state.kappa_e = extended_lengyel_formulas.calc_kappa_e(
        current_sol_model.divertor_Z_eff
    )

    current_sol_model.state.alpha_t = extended_lengyel_formulas.calc_alpha_t(
        separatrix_electron_density=current_sol_model.params.separatrix_electron_density,
        separatrix_electron_temp=current_sol_model.separatrix_electron_temp
        / 1e3,
        cylindrical_safety_factor=current_sol_model.params.cylindrical_safety_factor,
        major_radius=current_sol_model.params.major_radius,
        average_ion_mass=current_sol_model.params.average_ion_mass,
        Z_eff=current_sol_model.separatrix_Z_eff,
        mean_ion_charge_state=1.0,
    )

    # Relaxation step after the first iteration
    current_sol_model = jax.lax.cond(
        i > 0,
        lambda sol_model: dataclasses.replace(
            sol_model,
            state=dataclasses.replace(
                sol_model.state,
                target_electron_temp=_relax(
                    sol_model.state.target_electron_temp,
                    prev_sol_model.state.target_electron_temp,
                ),
                alpha_t=_relax(
                    sol_model.state.alpha_t, prev_sol_model.state.alpha_t
                ),
            ),
        ),
        lambda sol_model: sol_model,
        current_sol_model,
    )

    # Returning the updated-in-place current_sol_model.
    return current_sol_model

  sol_model = jax.lax.fori_loop(
      lower=0, upper=iterations, body_fun=body_fun, init_val=sol_model
  )

  return sol_model


def forward_mode_newton_solver(
    initial_sol_model: divertor_sol_1d_lib.DivertorSOL1D,
    maxiter: int = 30,
    tol: float = 1e-5,
) -> tuple[divertor_sol_1d_lib.DivertorSOL1D, jax_root_finding.RootMetadata]:
  """Runs the Newton-Raphson solver for the forward mode.

  Solves for {q_parallel, alpha_t, kappa_e, target_electron_temp} given fixed
  impurities.

  Args:
    initial_sol_model: A DivertorSOL1D object containing the initial plasma
      parameters and fixed impurity concentrations.
    maxiter: Maximum number of iterations for the Newton-Raphson solver.
    tol: Tolerance for convergence of the Newton-Raphson solver.

  Returns:
    final_sol_model: The updated DivertorSOL1D object with the solved state
      variables.
    metadata: Metadata from the root-finding process, including convergence
      status and number of iterations.
  """
  # 1. Create initial guess state vector.
  # Uses log space for strictly positive variables and to improve conditioning.
  # alpha_t is left linear since should always remain O(1) and log steps
  # can lead to numerical issues due to exponential amplification. Positivity is
  # enforced via softplus when unpacking.
  x0 = jnp.stack([
      jnp.log(initial_sol_model.state.q_parallel),
      initial_sol_model.state.alpha_t,
      jnp.log(initial_sol_model.state.kappa_e),
      jnp.log(initial_sol_model.state.target_electron_temp),
  ])

  # 2. Define residual function, closing over params and fixed c_z.
  fixed_cz = initial_sol_model.state.c_z_prefactor
  params = initial_sol_model.params

  residual_fun = functools.partial(
      _forward_residual, params=params, fixed_cz=fixed_cz
  )

  # 3. Run Newton-Raphson.
  x_root, metadata = jax_root_finding.root_newton_raphson(
      residual_fun, x0, maxiter=maxiter, tol=tol, use_jax_custom_root=True
  )

  # 4. Construct final model.
  final_state = divertor_sol_1d_lib.ExtendedLengyelState(
      q_parallel=jnp.exp(x_root[0]),
      alpha_t=jax.nn.softplus(x_root[1]),
      kappa_e=jnp.exp(x_root[2]),
      target_electron_temp=jnp.exp(x_root[3]),
      c_z_prefactor=fixed_cz,
  )
  final_sol_model = divertor_sol_1d_lib.DivertorSOL1D(
      params=params, state=final_state
  )

  return final_sol_model, metadata


def inverse_mode_newton_solver(
    initial_sol_model: divertor_sol_1d_lib.DivertorSOL1D,
    maxiter: int = 30,
    tol: float = 1e-5,
) -> tuple[divertor_sol_1d_lib.DivertorSOL1D, jax_root_finding.RootMetadata]:
  """Runs the Newton-Raphson solver for the inverse mode.

  Solves for {q_parallel, alpha_t, kappa_e, c_z_prefactor} given a fixed
  target electron temperature.

  Args:
    initial_sol_model: A DivertorSOL1D object containing the initial plasma
      parameters and a fixed target electron temperature.
    maxiter: Maximum number of iterations for the Newton-Raphson solver.
    tol: Tolerance for convergence of the Newton-Raphson solver.

  Returns:
    final_sol_model: The updated DivertorSOL1D object with the solved state
      variables.
    metadata: Metadata from the root-finding process, including convergence
      status and number of iterations.
  """
  # 1. Create initial guess state vector.

  # Uses log space for strictly positive variables and to improve conditioning.
  # alpha_t is left linear since should always remain O(1) and log steps
  # can lead to numerical issues due to exponential amplification. Positivity is
  # enforced via softplus when unpacking.

  x0 = jnp.stack([
      jnp.log(initial_sol_model.state.q_parallel),
      initial_sol_model.state.alpha_t,
      jnp.log(initial_sol_model.state.kappa_e),
      initial_sol_model.state.c_z_prefactor,
  ])

  # 2. Define residual function, closing over params and fixed T_t.
  fixed_Tt = initial_sol_model.state.target_electron_temp
  params = initial_sol_model.params

  residual_fun = functools.partial(
      _inverse_residual, params=params, fixed_Tt=fixed_Tt
  )

  # 3. Run Newton-Raphson.
  x_root, metadata = jax_root_finding.root_newton_raphson(
      residual_fun, x0, maxiter=maxiter, tol=tol, use_jax_custom_root=True
  )

  # 4. Construct final model.
  final_state = divertor_sol_1d_lib.ExtendedLengyelState(
      q_parallel=jnp.exp(x_root[0]),
      alpha_t=jax.nn.softplus(x_root[1]),
      kappa_e=jnp.exp(x_root[2]),
      c_z_prefactor=x_root[3],
      target_electron_temp=fixed_Tt,
  )

  final_sol_model = divertor_sol_1d_lib.DivertorSOL1D(
      params=params, state=final_state
  )

  return final_sol_model, metadata


def _forward_residual(
    x_vec: jax.Array,
    params: divertor_sol_1d_lib.ExtendedLengyelParameters,
    fixed_cz: jax.Array,
) -> jax.Array:
  """Calculates the residual vector for Forward Mode F(x) = 0."""
  # 1. Construct physical state from vector guess (uses exp/softplus).
  current_state = divertor_sol_1d_lib.ExtendedLengyelState(
      q_parallel=jnp.exp(x_vec[0]),
      alpha_t=jax.nn.softplus(x_vec[1]),
      kappa_e=jnp.exp(x_vec[2]),
      target_electron_temp=jnp.exp(x_vec[3]),
      c_z_prefactor=fixed_cz,
  )
  temp_model = divertor_sol_1d_lib.DivertorSOL1D(
      params=params, state=current_state
  )

  # 2. Calculate next guess of state variables.

  # a) q_parallel
  qp_calc = extended_lengyel_formulas.calculate_q_parallel(
      separatrix_electron_temp=temp_model.separatrix_electron_temp,
      average_ion_mass=params.average_ion_mass,
      separatrix_average_poloidal_field=params.separatrix_average_poloidal_field,
      alpha_t=current_state.alpha_t,
      ratio_of_upstream_to_average_poloidal_field=params.ratio_of_upstream_to_average_poloidal_field,
      fraction_of_PSOL_to_divertor=params.fraction_of_P_SOL_to_divertor,
      minor_radius=params.minor_radius,
      major_radius=params.major_radius,
      power_crossing_separatrix=params.power_crossing_separatrix,
      fieldline_pitch_at_omp=params.fieldline_pitch_at_omp,
  )

  # b) alpha_t
  at_calc = extended_lengyel_formulas.calc_alpha_t(
      separatrix_electron_density=params.separatrix_electron_density,
      separatrix_electron_temp=temp_model.separatrix_electron_temp / 1e3,
      cylindrical_safety_factor=params.cylindrical_safety_factor,
      major_radius=params.major_radius,
      average_ion_mass=params.average_ion_mass,
      Z_eff=temp_model.separatrix_Z_eff,
      mean_ion_charge_state=1.0,
  )

  # c) kappa_e
  ke_calc = extended_lengyel_formulas.calc_kappa_e(temp_model.divertor_Z_eff)

  # d) T_t
  q_cc_calc, _ = _solve_for_qcc(sol_model=temp_model)
  Tt_calc = divertor_sol_1d_lib.calc_target_electron_temp(
      sol_model=temp_model,
      parallel_heat_flux_at_cc_interface=q_cc_calc,
      previous_target_electron_temp=current_state.target_electron_temp,
  )

  # 3. Compute residuals in solver space for conditioning.
  # Enforce positivity for numerical stability.
  qp_calc_safe = jnp.maximum(qp_calc, constants.CONSTANTS.eps)
  ke_calc_safe = jnp.maximum(ke_calc, constants.CONSTANTS.eps)
  Tt_calc_safe = jnp.maximum(Tt_calc, constants.CONSTANTS.eps)
  at_calc_safe = jnp.maximum(at_calc, constants.CONSTANTS.eps)

  r_qp = jnp.log(qp_calc_safe) - x_vec[0]
  r_at = at_calc_safe - current_state.alpha_t
  r_ke = jnp.log(ke_calc_safe) - x_vec[2]
  r_Tt = jnp.log(Tt_calc_safe) - x_vec[3]

  return jnp.stack([r_qp, r_at, r_ke, r_Tt])


def _inverse_residual(
    x_vec: jax.Array,
    params: divertor_sol_1d_lib.ExtendedLengyelParameters,
    fixed_Tt: jax.Array,
) -> jax.Array:
  """Calculates the residual vector for Inverse Mode F(x) = 0."""
  # 1. Construct physical state from vector guess.
  current_state = divertor_sol_1d_lib.ExtendedLengyelState(
      q_parallel=jnp.exp(x_vec[0]),
      alpha_t=jax.nn.softplus(x_vec[1]),
      kappa_e=jnp.exp(x_vec[2]),
      c_z_prefactor=x_vec[3],
      target_electron_temp=fixed_Tt,
  )

  temp_model = divertor_sol_1d_lib.DivertorSOL1D(
      params=params, state=current_state
  )

  # 2. Calculate next guess of state variables.

  # a) q_parallel
  qp_calc = extended_lengyel_formulas.calculate_q_parallel(
      separatrix_electron_temp=temp_model.separatrix_electron_temp,
      average_ion_mass=params.average_ion_mass,
      separatrix_average_poloidal_field=params.separatrix_average_poloidal_field,
      alpha_t=current_state.alpha_t,
      ratio_of_upstream_to_average_poloidal_field=params.ratio_of_upstream_to_average_poloidal_field,
      fraction_of_PSOL_to_divertor=params.fraction_of_P_SOL_to_divertor,
      minor_radius=params.minor_radius,
      major_radius=params.major_radius,
      power_crossing_separatrix=params.power_crossing_separatrix,
      fieldline_pitch_at_omp=params.fieldline_pitch_at_omp,
  )

  # b) alpha_t
  at_calc = extended_lengyel_formulas.calc_alpha_t(
      separatrix_electron_density=params.separatrix_electron_density,
      separatrix_electron_temp=temp_model.separatrix_electron_temp / 1e3,
      cylindrical_safety_factor=params.cylindrical_safety_factor,
      major_radius=params.major_radius,
      average_ion_mass=params.average_ion_mass,
      Z_eff=temp_model.separatrix_Z_eff,
      mean_ion_charge_state=1.0,
  )

  # c) kappa_e
  ke_calc = extended_lengyel_formulas.calc_kappa_e(temp_model.divertor_Z_eff)

  # d) c_z_prefactor
  cz_calc, _ = _solve_for_c_z_prefactor(sol_model=temp_model)

  # 3. Compute residuals in solver space for conditioning.
  qp_calc_safe = jnp.maximum(qp_calc, constants.CONSTANTS.eps)
  ke_calc_safe = jnp.maximum(ke_calc, constants.CONSTANTS.eps)
  at_calc_safe = jnp.maximum(at_calc, constants.CONSTANTS.eps)

  r_qp = jnp.log(qp_calc_safe) - x_vec[0]
  r_at = at_calc_safe - current_state.alpha_t
  r_ke = jnp.log(ke_calc_safe) - x_vec[2]
  r_cz = cz_calc - current_state.c_z_prefactor

  return jnp.stack([r_qp, r_at, r_ke, r_cz])


def _solve_for_c_z_prefactor(
    sol_model: divertor_sol_1d_lib.DivertorSOL1D,
) -> tuple[jax.Array, jax.Array]:
  """Solves the extended Lengyel model for the required impurity concentration.

  This function implements the extended Lengyel model in inverse mode,
  calculating the seeded impurity concentration (`c_z`) needed to achieve the
  an input target temperature consistent with a given set of plasma parameters.

  See Section 5 of T. Body et al 2025 Nucl. Fusion 65 086002 for the derivation.
  https://doi.org/10.1088/1741-4326/ade4d9

  Args:
    sol_model: A DivertorSOL1D object containing the plasma parameters.

  Returns:
      c_z_prefactor: The scaling factor for the seeded impurity concentrations.
        To be multiplied by seed_impurity_weights to get each seeded impurity
        concentration.
      status: A SolveCzStatus enum indicating the outcome of the calculation.
  """
  # Temperatures must be in keV for the L_INT calculation.
  cc_temp_keV = sol_model.electron_temp_at_cc_interface / 1000.0
  div_temp_keV = sol_model.divertor_entrance_electron_temp / 1000.0
  sep_temp_keV = sol_model.separatrix_electron_temp / 1000.0

  # Calculate integrated radiation terms (L_INT) for seeded impurities.
  # See Eq. 34 in Body et al. 2025.
  Ls_cc_div = (
      collisional_radiative_models.calculate_weighted_L_INT(
          sol_model.params.seed_impurity_weights,
          start_temp=cc_temp_keV,
          stop_temp=div_temp_keV,
          ne_tau=sol_model.params.ne_tau,
      )
      * _LINT_SCALE_FACTOR
  )
  Ls_cc_u = (
      collisional_radiative_models.calculate_weighted_L_INT(
          sol_model.params.seed_impurity_weights,
          start_temp=cc_temp_keV,
          stop_temp=sep_temp_keV,
          ne_tau=sol_model.params.ne_tau,
      )
      * _LINT_SCALE_FACTOR
  )
  Ls_div_u = Ls_cc_u - Ls_cc_div

  # Calculate integrated radiation terms for fixed background impurities.
  Lf_cc_div = (
      collisional_radiative_models.calculate_weighted_L_INT(
          sol_model.params.fixed_impurity_concentrations,
          start_temp=cc_temp_keV,
          stop_temp=div_temp_keV,
          ne_tau=sol_model.params.ne_tau,
      )
      * _LINT_SCALE_FACTOR
  )
  Lf_cc_u = (
      collisional_radiative_models.calculate_weighted_L_INT(
          sol_model.params.fixed_impurity_concentrations,
          start_temp=cc_temp_keV,
          stop_temp=sep_temp_keV,
          ne_tau=sol_model.params.ne_tau,
      )
      * _LINT_SCALE_FACTOR
  )
  Lf_div_u = Lf_cc_u - Lf_cc_div

  # Define shorthand variables for clarity, matching the paper's notation.
  qu = sol_model.state.q_parallel
  qcc = sol_model.parallel_heat_flux_at_cc_interface
  b = sol_model.params.divertor_broadening_factor
  # `k` is a lumped parameter from the Lengyel model derivation.
  # See Eq. 33 in Body et al. 2025.
  # Need log to avoid overflow in fp32 when jitted.
  log_k = (
      jnp.log(2.0)
      + jnp.log(sol_model.state.kappa_e)
      + 2.0
      * jnp.log(
          sol_model.params.separatrix_electron_density * _DENSITY_SCALE_FACTOR
      )
      + 2.0 * jnp.log(sol_model.separatrix_electron_temp)
  )

  k = jnp.exp(log_k)

  # Calculate the squared parallel heat flux at the divertor entrance.
  # This formula is derived by combining the Lengyel equations for the region
  # above and below the divertor entrance. See Eq. 40 in Body et al. 2025.
  q_div_squared = (
      Ls_div_u
      / _LINT_SCALE_FACTOR
      * (qcc**2 + k * Lf_cc_div / _LINT_K_INVERSE_SCALE_FACTOR)
      + (Ls_cc_div / _LINT_SCALE_FACTOR)
      * (qu**2 - k * Lf_div_u / _LINT_K_INVERSE_SCALE_FACTOR)
  ) / (Ls_div_u / _LINT_SCALE_FACTOR / b**2 + Ls_cc_div / _LINT_SCALE_FACTOR)

  # Check for unphysical result.
  status = jnp.where(
      q_div_squared < 0.0,
      SolveStatus.Q_DIV_SQUARED_NEGATIVE,
      SolveStatus.SUCCESS,
  )

  # Calculate the required seeded impurity concentration `c_z`.
  # See Eq. 42 in Body et al. 2025.
  c_z_prefactor = (
      (qu**2 + (1.0 / b**2 - 1.0) * q_div_squared - qcc**2)
      / (k * Ls_cc_u / _LINT_K_INVERSE_SCALE_FACTOR)
  ) - (Lf_cc_u / Ls_cc_u)

  return c_z_prefactor, status


def _solve_for_qcc(
    sol_model: divertor_sol_1d_lib.DivertorSOL1D,
) -> tuple[jax.Array, jax.Array]:
  """Calculates the parallel heat flux at the cc-interface for fixed impurities.

  This function is part of the extended Lengyel model in forward mode,
  calculating the parallel heat flux at the convective-conductive interface
  (`q_cc`) which is needed to calculate the target temperature consistent with
  a given set of plasma parameters.

  See Section 5 of T. Body et al 2025 Nucl. Fusion 65 086002 for the derivation.
  This is equation 38 rearranged for q_cc, and using equation 39 to calculate
  q_div.
  https://doi.org/10.1088/1741-4326/ade4d9

  Args:
    sol_model: A DivertorSOL1D object containing the plasma parameters.

  Returns:
      q_cc: The parallel heat flux at the cc-interface
      status: A SolveStatus enum indicating the outcome of the calculation.
  """
  # Temperatures must be in keV for the L_INT calculation.
  cc_temp_keV = sol_model.electron_temp_at_cc_interface / 1000.0
  div_temp_keV = sol_model.divertor_entrance_electron_temp / 1000.0
  sep_temp_keV = sol_model.separatrix_electron_temp / 1000.0

  # Calculate integrated radiation terms for fixed impurities.
  # See Eq. 34 in Body et al. 2025.
  Lint_cc_div = (
      collisional_radiative_models.calculate_weighted_L_INT(
          sol_model.params.fixed_impurity_concentrations,
          start_temp=cc_temp_keV,
          stop_temp=div_temp_keV,
          ne_tau=sol_model.params.ne_tau,
      )
      * _LINT_SCALE_FACTOR
  )
  Lint_cc_u = (
      collisional_radiative_models.calculate_weighted_L_INT(
          sol_model.params.fixed_impurity_concentrations,
          start_temp=cc_temp_keV,
          stop_temp=sep_temp_keV,
          ne_tau=sol_model.params.ne_tau,
      )
      * _LINT_SCALE_FACTOR
  )
  Lint_div_u = Lint_cc_u - Lint_cc_div

  # Define shorthand variables for clarity, matching the paper's notation.
  qu = sol_model.state.q_parallel
  b = sol_model.params.divertor_broadening_factor
  # `k` is a lumped parameter from the Lengyel model derivation.
  # See Eq. 33 in Body et al. 2025.
  # Need log to avoid overflow in fp32 when jitted.
  log_k = (
      jnp.log(2.0)
      + jnp.log(sol_model.state.kappa_e)
      + 2.0
      * jnp.log(
          sol_model.params.separatrix_electron_density * _DENSITY_SCALE_FACTOR
      )
      + 2.0 * jnp.log(sol_model.separatrix_electron_temp)
  )

  k = jnp.exp(log_k)

  qcc_squared = (
      qu**2 / b**2
      - k * (Lint_div_u / b**2 + Lint_cc_div) / _LINT_K_INVERSE_SCALE_FACTOR
  )

  # Check for unphysical result.
  status = jnp.where(
      qcc_squared < 0.0,
      SolveStatus.Q_CC_SQUARED_NEGATIVE,
      SolveStatus.SUCCESS,
  )

  qcc = jnp.where(status == SolveStatus.SUCCESS, jnp.sqrt(qcc_squared), 0.0)

  return qcc, status

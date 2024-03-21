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

"""Carries out the predictor corrector method for the PDE solution.

Picard iterations to approximate the nonlinear solution. If
static_config_slice.solver.predictor_corrector is False, reverts to a
standard linear solution.
"""

import jax
from jax import numpy as jnp
from torax import calc_coeffs
from torax import config_slice
from torax import fvm
from torax import jax_utils
from torax import state as state_module
from torax.fvm import implicit_solve_block


def predictor_corrector_method(
    init_val: tuple[
        tuple[fvm.cell_variable.CellVariable, ...], calc_coeffs.AuxOutput
    ],
    state_t_plus_dt: state_module.State,
    evolving_names: tuple[str, ...],
    dt: jax.Array,
    coeffs_exp: fvm.block_1d_coeffs.Block1DCoeffs,
    coeffs_callback: fvm.block_1d_coeffs.Block1DCoeffsCallback,
    dynamic_config_slice_t_plus_dt: config_slice.DynamicConfigSlice,
    static_config_slice: config_slice.StaticConfigSlice,
) -> tuple[tuple[fvm.CellVariable, ...], calc_coeffs.AuxOutput]:
  """Predictor-corrector method.

  Args:
    init_val: initial guess for the predictor corrector output.
    state_t_plus_dt: Sim state which contains all available prescribed
      quantities at the end of the time step. This includes evolving boundary
      conditions and prescribed time-dependent profiles that are not being
      evolved by the PDE system.
    evolving_names: The names of variables within the state that should evolve.
    dt: current timestep
    coeffs_exp: Block1DCoeffs PDE coefficients at beginning of timestep
    coeffs_callback: coefficient callback function
    dynamic_config_slice_t_plus_dt: dynamic config parameters corresponding to
      the next time step, needed for the implicit PDE coefficients
    static_config_slice: General input parameters which are fixed through a
      simulation run, and if changed, would trigger a recompile.

  Returns:
    x_new: solution of evolving state variables
    auxiliary_outputs: Block1DCoeffs containing the PDE coefficients
    corresponding to the last guess of x_new
  """

  # predictor-corrector loop. Will only be traversed once if not in
  # predictor-corrector mode
  def loop_body(i, val):  # pylint: disable=unused-argument
    x_new_guess = val[0]
    x_new_vec_guess = jnp.concatenate([var.value for var in x_new_guess])

    x_new, auxiliary_outputs = implicit_solve_block.implicit_solve_block(
        x_old=init_val[0],
        x_new_vec_guess=x_new_vec_guess,
        state_t_plus_dt=state_t_plus_dt,
        evolving_names=evolving_names,
        dt=dt,
        coeffs_old=coeffs_exp,
        coeffs_callback=coeffs_callback,
        dynamic_config_slice_t_plus_dt=dynamic_config_slice_t_plus_dt,
        # theta_imp is not time-dependent. Not all parameters in the
        # dynamic_config_slice need to be time-dependent. They can simply
        # change from simulation run to simulation run without triggering a
        # recompile.
        theta_imp=dynamic_config_slice_t_plus_dt.solver.theta_imp,
        convection_dirichlet_mode=(
            static_config_slice.solver.convection_dirichlet_mode
        ),
        convection_neumann_mode=(
            static_config_slice.solver.convection_neumann_mode
        ),
    )

    return (x_new, auxiliary_outputs)

  # jax.lax.fori_loop jits the function by default. Need to explicitly avoid
  # compilation and revert to a standard for loop if
  # TORAX_COMPILATION_ENABLED=False. This logic is in jax.utils_py_fori_loop.
  # If the static predictor_corrector=False, then compilation is faster, so
  # we maintain this option.
  if static_config_slice.solver.predictor_corrector:
    return jax_utils.py_fori_loop(
        0,
        dynamic_config_slice_t_plus_dt.solver.corrector_steps + 1,
        loop_body,
        init_val,
    )
  else:
    return loop_body(0, init_val)

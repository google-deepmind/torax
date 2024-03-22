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
from torax import calc_coeffs
from torax import config_slice
from torax import fvm
from torax import jax_utils
from torax.fvm import implicit_solve_block


def predictor_corrector_method(
    init_val: tuple[
        tuple[fvm.CellVariable, ...], calc_coeffs.AuxOutput
    ],
    x_old: tuple[fvm.CellVariable, ...],
    dt: jax.Array,
    coeffs_exp: fvm.block_1d_coeffs.Block1DCoeffs,
    coeffs_callback: fvm.block_1d_coeffs.Block1DCoeffsCallback,
    dynamic_config_slice_t_plus_dt: config_slice.DynamicConfigSlice,
    static_config_slice: config_slice.StaticConfigSlice,
) -> tuple[tuple[fvm.CellVariable, ...], calc_coeffs.AuxOutput]:
  """Predictor-corrector method.

  Args:
    init_val: Initial guess for the predictor corrector output.
    x_old: Tuple of CellVariables correspond to the evolving state at time t.
    dt: current timestep
    coeffs_exp: Block1DCoeffs PDE coefficients at beginning of timestep
    coeffs_callback: coefficient callback function
    dynamic_config_slice_t_plus_dt: Dynamic config parameters corresponding to
      the next time step, needed for the implicit PDE coefficients
    static_config_slice: General input parameters which are fixed through a
      simulation run, and if changed, would trigger a recompile.

  Returns:
    x_new: Solution of evolving state variables
    auxiliary_outputs: Block1DCoeffs containing the PDE coefficients
    corresponding to the last guess of x_new
  """

  # predictor-corrector loop. Will only be traversed once if not in
  # predictor-corrector mode
  def loop_body(i, val):  # pylint: disable=unused-argument
    x_new_guess = val[0]

    coeffs_new = coeffs_callback(
        x_new_guess, dynamic_config_slice_t_plus_dt, allow_pereverzev=True,
    )
    aux_output = coeffs_new.auxiliary_outputs

    x_new = implicit_solve_block.implicit_solve_block(
        x_old=x_old,
        x_new_guess=x_new_guess,
        dt=dt,
        coeffs_old=coeffs_exp,
        coeffs_new=coeffs_new,
        theta_imp=static_config_slice.solver.theta_imp,
        convection_dirichlet_mode=(
            static_config_slice.solver.convection_dirichlet_mode
        ),
        convection_neumann_mode=(
            static_config_slice.solver.convection_neumann_mode
        ),
    )

    return (x_new, aux_output)

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

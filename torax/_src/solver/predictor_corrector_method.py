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
static_runtime_params_slice.solver.use_predictor_corrector is False, reverts to
a standard linear solution.
"""
import jax
from torax._src import jax_utils
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.fvm import block_1d_coeffs
from torax._src.fvm import calc_coeffs
from torax._src.fvm import cell_variable
from torax._src.fvm import implicit_solve_block
from torax._src.geometry import geometry


def predictor_corrector_method(
    dt: jax.Array,
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice_t_plus_dt: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo_t_plus_dt: geometry.Geometry,
    x_old: tuple[cell_variable.CellVariable, ...],
    x_new_guess: tuple[cell_variable.CellVariable, ...],
    core_profiles_t_plus_dt: state.CoreProfiles,
    coeffs_exp: block_1d_coeffs.Block1DCoeffs,
    coeffs_callback: calc_coeffs.CoeffsCallback,
) -> tuple[cell_variable.CellVariable, ...]:
  """Predictor-corrector method.

  Args:
    dt: current timestep
    static_runtime_params_slice: General input parameters which are fixed
      through a simulation run, and if changed, would trigger a recompile.
    dynamic_runtime_params_slice_t_plus_dt: Dynamic runtime parameters
      corresponding to the next time step, needed for the implicit PDE
      coefficients.
    geo_t_plus_dt: Geometry at the next time step.
    x_old: Tuple of CellVariables correspond to the evolving core profiles at
      time t.
    x_new_guess: Tuple of CellVariables corresponding to the initial guess for
      the next time step.
    core_profiles_t_plus_dt: Core profiles at the next time step.
    coeffs_exp: Block1DCoeffs PDE coefficients at beginning of timestep
    coeffs_callback: coefficient callback function

  Returns:
    x_new: Solution of evolving core profile state variables
  """

  # predictor-corrector loop. Will only be traversed once if not in
  # predictor-corrector mode
  def loop_body(i, x_new_guess):  # pylint: disable=unused-argument
    coeffs_new = coeffs_callback(
        dynamic_runtime_params_slice_t_plus_dt,
        geo_t_plus_dt,
        core_profiles_t_plus_dt,
        x_new_guess,
        allow_pereverzev=True,
    )

    return implicit_solve_block.implicit_solve_block(
        dt=dt,
        x_old=x_old,
        x_new_guess=x_new_guess,
        coeffs_old=coeffs_exp,
        coeffs_new=coeffs_new,
        theta_implicit=static_runtime_params_slice.solver.theta_implicit,
        convection_dirichlet_mode=(
            static_runtime_params_slice.solver.convection_dirichlet_mode
        ),
        convection_neumann_mode=(
            static_runtime_params_slice.solver.convection_neumann_mode
        ),
    )

  # jax.lax.fori_loop jits the function by default. Need to explicitly avoid
  # compilation and revert to a standard for loop if
  # TORAX_COMPILATION_ENABLED=False. This logic is in jax.utils_py_fori_loop.
  # If the static use_predictor_corrector=False, then compilation is faster, so
  # we maintain this option.
  if static_runtime_params_slice.solver.use_predictor_corrector:
    x_new = jax_utils.py_fori_loop(
        0,
        dynamic_runtime_params_slice_t_plus_dt.solver.n_corrector_steps + 1,
        loop_body,
        x_new_guess,
    )
  else:
    x_new = loop_body(0, x_new_guess)
  return x_new

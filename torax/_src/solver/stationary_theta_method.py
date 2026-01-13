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

"""The StationaryThetaMethod solver class for equilibrium state finding."""
import functools

import jax
from jax import numpy as jnp
from torax._src import jax_utils
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import convertors
from torax._src.fvm import calc_coeffs
from torax._src.fvm import cell_variable
from torax._src.fvm import implicit_solve_block
from torax._src.geometry import geometry
from torax._src.solver import solver as solver_lib
from torax._src.sources import source_profiles


class StationaryThetaMethod(solver_lib.Solver):
  """Solves for the stationary (equilibrium) state where ∂x/∂t = 0.
  
  This solver finds the equilibrium profiles where the time derivatives
  are zero. It uses the theta method with theta=1 (fully implicit) and
  a nominal time step, solving the algebraic equilibrium equations:
  
    C(x_eq) * x_eq + c(x_eq) = 0
  
  This is useful for:
  - Finding initial equilibrium states
  - Computing steady-state solutions
  - Performing equilibrium analysis without time evolution
  """

  @functools.partial(
      jax.jit,
      static_argnames=[
          'self',
          'evolving_names',
      ],
  )
  def _x_new(
      self,
      dt: jax.Array,
      runtime_params_t: runtime_params_lib.RuntimeParams,
      runtime_params_t_plus_dt: runtime_params_lib.RuntimeParams,
      geo_t: geometry.Geometry,
      geo_t_plus_dt: geometry.Geometry,
      core_profiles_t: state.CoreProfiles,
      core_profiles_t_plus_dt: state.CoreProfiles,
      explicit_source_profiles: source_profiles.SourceProfiles,
      evolving_names: tuple[str, ...],
  ) -> tuple[
      tuple[cell_variable.CellVariable, ...],
      state.SolverNumericOutputs,
  ]:
    """Finds the stationary state where ∂x/∂t = 0.
    
    For the stationary state, we solve the implicit equation with theta=1:
      T(x_eq) * x_eq = dt * C(x_eq) * x_eq + dt * c(x_eq)
    
    As dt → 0, this becomes:
      T(x_eq) * x_eq = 0, or equivalently: C(x_eq) * x_eq + c(x_eq) = 0
    
    The solver uses a small time step and fully implicit integration to
    find the equilibrium state.
    
    Args:
      dt: Time step duration (treated as a nominal small step for equilibrium).
      runtime_params_t: Runtime parameters at time t.
      runtime_params_t_plus_dt: Runtime parameters at time t + dt.
      geo_t: Magnetic geometry at time t.
      geo_t_plus_dt: Magnetic geometry at time t + dt.
      core_profiles_t: Core profiles at time t (used as initial guess).
      core_profiles_t_plus_dt: Core profiles at time t + dt.
      explicit_source_profiles: Explicit source profiles.
      evolving_names: Names of variables to evolve.
    
    Returns:
      A tuple containing:
        - x_eq: The equilibrium state.
        - solver_numeric_outputs: Solver statistics.
    """
    x_old = convertors.core_profiles_to_solver_x_tuple(
        core_profiles_t, evolving_names
    )
    x_new_guess = convertors.core_profiles_to_solver_x_tuple(
        core_profiles_t_plus_dt, evolving_names
    )

    coeffs_callback = calc_coeffs.CoeffsCallback(
        physics_models=self.physics_models,
        evolving_names=evolving_names,
    )

    # For stationary state, we compute coefficients at t + dt (equilibrium)
    # since we're solving for equilibrium, not time evolution
    coeffs_imp = coeffs_callback(
        runtime_params_t_plus_dt,
        geo_t_plus_dt,
        core_profiles_t_plus_dt,
        x_new_guess,
        explicit_source_profiles=explicit_source_profiles,
        allow_pereverzev=False,
        explicit_call=False,
    )

    # For stationary state, we use the implicit solve with theta=1 (fully implicit)
    # The time derivative term becomes negligible as dt → 0
    x_new, solver_iterations = implicit_solve_block.implicit_solve_block(
        dt=dt,
        runtime_params_t_plus_dt=runtime_params_t_plus_dt,
        geo_t_plus_dt=geo_t_plus_dt,
        x_old=x_old,
        x_new_guess=x_new_guess,
        core_profiles_t_plus_dt=core_profiles_t_plus_dt,
        coeffs_imp=coeffs_imp,
        coeffs_callback=coeffs_callback,
        explicit_source_profiles=explicit_source_profiles,
        theta_implicit=1.0,  # Fully implicit for stationary state
    )

    solver_numeric_outputs = state.SolverNumericOutputs(
        inner_solver_iterations=jnp.array(
            solver_iterations, jax_utils.get_int_dtype()
        ),
        outer_solver_iterations=jnp.array(1, jax_utils.get_int_dtype()),
        # Stationary solver treats convergence the same as linear solver
        solver_error_state=jnp.array(0, jax_utils.get_int_dtype()),
        sawtooth_crash=False,
    )

    return (
        x_new,
        solver_numeric_outputs,
    )
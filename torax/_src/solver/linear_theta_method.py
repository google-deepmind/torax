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

"""The LinearThetaMethod solver class."""
import functools

import jax
from jax import numpy as jnp
from torax._src import jax_utils
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.core_profiles import convertors
from torax._src.fvm import calc_coeffs
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.solver import common as solver_common
from torax._src.solver import predictor_corrector_method
from torax._src.solver import solver as solver_lib
from torax._src.sources import source_profiles


class LinearThetaMethod(solver_lib.Solver):
  """Time step update using theta method, linearized on coefficients at t."""

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
      runtime_params_t: runtime_params_slice.RuntimeParams,
      runtime_params_t_plus_dt: runtime_params_slice.RuntimeParams,
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
    """See Solver._x_new docstring."""

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

    # Compute the explicit coeffs based on the core profiles at time t and all
    # runtime parameters at time t.
    coeffs_exp = coeffs_callback(
        runtime_params_t,
        geo_t,
        core_profiles_t,
        x_old,
        explicit_source_profiles=explicit_source_profiles,
        allow_pereverzev=True,
        explicit_call=True,
    )

    # Calculate x_new with the predictor corrector method. Reverts to a
    # standard linear solve if
    # runtime_params_slice.predictor_corrector=False.
    # init_val is the initialization for the predictor_corrector loop.
    x_new = predictor_corrector_method.predictor_corrector_method(
        dt=dt,
        runtime_params_t_plus_dt=runtime_params_t_plus_dt,
        geo_t_plus_dt=geo_t_plus_dt,
        x_old=x_old,
        x_new_guess=x_new_guess,
        core_profiles_t_plus_dt=core_profiles_t_plus_dt,
        coeffs_exp=coeffs_exp,
        coeffs_callback=coeffs_callback,
        explicit_source_profiles=explicit_source_profiles,
    )

    if runtime_params_t_plus_dt.solver.use_predictor_corrector:
      inner_solver_iterations = (
          1 + runtime_params_t_plus_dt.solver.n_corrector_steps
      )
    else:
      inner_solver_iterations = 1

    solver_numeric_outputs = state.SolverNumericOutputs(
        inner_solver_iterations=jnp.array(
            inner_solver_iterations, jax_utils.get_int_dtype()
        ),
        outer_solver_iterations=jnp.array(1, jax_utils.get_int_dtype()),
        # linear method always works
        solver_error_state=solver_common.SolverError.converged,
        sawtooth_crash=False,
    )

    return (
        x_new,
        solver_numeric_outputs,
    )

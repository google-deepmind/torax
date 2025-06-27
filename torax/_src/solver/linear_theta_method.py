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
from torax._src import state
from torax._src import xnp
from torax._src.config import runtime_params_slice
from torax._src.core_profiles import convertors
from torax._src.fvm import calc_coeffs
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.solver import predictor_corrector_method
from torax._src.solver import solver as solver_lib
from torax._src.sources import source_profiles


class LinearThetaMethod(solver_lib.Solver):
  """Time step update using theta method, linearized on coefficients at t."""

  @functools.partial(
      xnp.jit,
      static_argnames=[
          'self',
          'static_runtime_params_slice',
          'evolving_names',
      ],
  )
  def _x_new(
      self,
      dt: jax.Array,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice_t: runtime_params_slice.DynamicRuntimeParamsSlice,
      dynamic_runtime_params_slice_t_plus_dt: runtime_params_slice.DynamicRuntimeParamsSlice,
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
        static_runtime_params_slice=static_runtime_params_slice,
        transport_model=self.transport_model,
        source_models=self.source_models,
        pedestal_model=self.pedestal_model,
        neoclassical_models=self.neoclassical_models,
        evolving_names=evolving_names,
    )

    # Compute the explicit coeffs based on the core profiles at time t and all
    # runtime parameters at time t.
    coeffs_exp = coeffs_callback(
        dynamic_runtime_params_slice_t,
        geo_t,
        core_profiles_t,
        x_old,
        explicit_source_profiles=explicit_source_profiles,
        allow_pereverzev=True,
        explicit_call=True,
    )

    # Calculate x_new with the predictor corrector method. Reverts to a
    # standard linear solve if
    # static_runtime_params_slice.predictor_corrector=False.
    # init_val is the initialization for the predictor_corrector loop.
    x_new = predictor_corrector_method.predictor_corrector_method(
        dt=dt,
        static_runtime_params_slice=static_runtime_params_slice,
        dynamic_runtime_params_slice_t_plus_dt=(
            dynamic_runtime_params_slice_t_plus_dt
        ),
        geo_t_plus_dt=geo_t_plus_dt,
        x_old=x_old,
        x_new_guess=x_new_guess,
        core_profiles_t_plus_dt=core_profiles_t_plus_dt,
        coeffs_exp=coeffs_exp,
        coeffs_callback=coeffs_callback,
        explicit_source_profiles=explicit_source_profiles,
    )

    if static_runtime_params_slice.solver.use_predictor_corrector:
      inner_solver_iterations = (
          1 + dynamic_runtime_params_slice_t_plus_dt.solver.n_corrector_steps
      )
    else:
      inner_solver_iterations = 1

    solver_numeric_outputs = state.SolverNumericOutputs(
        inner_solver_iterations=inner_solver_iterations,
        outer_solver_iterations=1,
        solver_error_state=0,  # linear method always works
    )

    return (
        x_new,
        solver_numeric_outputs,
    )

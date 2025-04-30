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
import jax
from torax import state
from torax.config import runtime_params_slice
from torax.fvm import calc_coeffs
from torax.fvm import cell_variable
from torax.geometry import geometry
from torax.sources import source_profiles
from torax.stepper import predictor_corrector_method
from torax.stepper import stepper as stepper_lib


class LinearThetaMethod(stepper_lib.Solver):
  """Time step update using theta method, linearized on coefficients at t."""

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
      source_profiles.SourceProfiles,
      state.CoreTransport,
      state.SolverNumericOutputs,
  ]:
    """See Solver._x_new docstring."""

    x_old = tuple([core_profiles_t[name] for name in evolving_names])
    x_new_guess = tuple(
        [core_profiles_t_plus_dt[name] for name in evolving_names]
    )

    coeffs_callback = calc_coeffs.CoeffsCallback(
        static_runtime_params_slice=static_runtime_params_slice,
        transport_model=self.transport_model,
        explicit_source_profiles=explicit_source_profiles,
        source_models=self.source_models,
        pedestal_model=self.pedestal_model,
        evolving_names=evolving_names,
    )

    # Compute the explicit coeffs based on the core profiles at time t and all
    # runtime parameters at time t.
    coeffs_exp = coeffs_callback(
        dynamic_runtime_params_slice_t,
        geo_t,
        core_profiles_t,
        x_old,
        allow_pereverzev=True,
        explicit_call=True,
    )

    # Calculate x_new with the predictor corrector method. Reverts to a
    # standard linear solve if
    # static_runtime_params_slice.predictor_corrector=False.
    # init_val is the initialization for the predictor_corrector loop.
    x_new, _ = (
        predictor_corrector_method.predictor_corrector_method(
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
        )
    )

    coeffs_final = coeffs_callback(
        dynamic_runtime_params_slice_t_plus_dt,
        geo_t_plus_dt,
        core_profiles_t_plus_dt,
        x_new,
        allow_pereverzev=True,
    )
    core_sources, core_transport = coeffs_final.auxiliary_outputs

    solver_numeric_outputs = state.SolverNumericOutputs(
        inner_solver_iterations=1,
        solver_error_state=0,  # linear method always works
    )

    return x_new, core_sources, core_transport, solver_numeric_outputs

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

"""The LinearThetaMethodStepper class."""

from __future__ import annotations

from typing import Literal

import jax
import pydantic
from torax import state
from torax.config import runtime_params_slice
from torax.fvm import calc_coeffs
from torax.fvm import cell_variable
from torax.geometry import geometry
from torax.sources import source_profiles
from torax.stepper import predictor_corrector_method
from torax.stepper import runtime_params
from torax.stepper import stepper as stepper_lib
from torax.torax_pydantic import torax_pydantic


# pylint: disable=invalid-name
class Linear(torax_pydantic.BaseModelMutable):
  """Model for linear stepper.

  Attributes:
    stepper_type: The type of stepper to use, hardcoded to 'linear'.
    theta_imp: The theta value in the theta method 0 = explicit, 1 = fully
      implicit, 0.5 = Crank-Nicolson.
    predictor_corrector: Enables predictor_corrector iterations with the linear
      solver. If False, compilation is faster.
    corrector_steps: The number of corrector steps for the predictor-corrector
      linear solver. 0 means a pure linear solve with no corrector steps.
    convection_dirichlet_mode: See `fvm.convection_terms` docstring,
      `dirichlet_mode` argument.
    convection_neumann_mode: See `fvm.convection_terms` docstring,
      `neumann_mode` argument.
    use_pereverzev: Use pereverzev terms for linear solver. Is only applied in
      the nonlinear solver for the optional initial guess from the linear solver
    chi_per: (deliberately) large heat conductivity for Pereverzev rule.
    d_per: (deliberately) large particle diffusion for Pereverzev rule.
  """

  stepper_type: Literal['linear'] = 'linear'
  theta_imp: torax_pydantic.UnitInterval = 1.0
  predictor_corrector: bool = True
  corrector_steps: pydantic.NonNegativeInt = 1
  convection_dirichlet_mode: Literal['ghost', 'direct', 'semi-implicit'] = (
      'ghost'
  )
  convection_neumann_mode: Literal['ghost', 'semi-implicit'] = 'ghost'
  use_pereverzev: bool = False
  chi_per: float = 20.0
  d_per: float = 10.0

  def build_dynamic_params(self) -> runtime_params.DynamicRuntimeParams:
    return runtime_params.DynamicRuntimeParams(
        chi_per=self.chi_per,
        d_per=self.d_per,
        corrector_steps=self.corrector_steps,
    )

  def build_stepper(
      self, transport_model, source_models, pedestal_model
  ) -> LinearThetaMethod:
    return LinearThetaMethod(
        transport_model=transport_model,
        source_models=source_models,
        pedestal_model=pedestal_model,
    )


class LinearThetaMethod(stepper_lib.Stepper):
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
      state.StepperNumericOutputs,
  ]:
    """See Stepper._x_new docstring."""

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

    stepper_numeric_outputs = state.StepperNumericOutputs(
        inner_solver_iterations=1,
        stepper_error_state=0,  # linear method always works
    )

    return x_new, core_sources, core_transport, stepper_numeric_outputs

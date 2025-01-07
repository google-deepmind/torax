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

from collections.abc import Callable
import dataclasses
from typing import Type, TypeAlias

import jax
from torax import state
from torax.config import runtime_params_slice
from torax.fvm import calc_coeffs
from torax.fvm import cell_variable
from torax.geometry import geometry
from torax.pedestal_model import pedestal_model as pedestal_model_lib
from torax.sources import source_models as source_models_lib
from torax.sources import source_profiles
from torax.stepper import predictor_corrector_method
from torax.stepper import runtime_params as runtime_params_lib
from torax.stepper import stepper as stepper_lib
from torax.transport_model import transport_model as transport_model_lib


class LinearThetaMethod(stepper_lib.Stepper):
  """Time step update using theta method, linearized on coefficients at t."""

  def __init__(
      self,
      transport_model: transport_model_lib.TransportModel,
      source_models: source_models_lib.SourceModels,
      pedestal_model: pedestal_model_lib.PedestalModel,
      callback_class: Type[
          calc_coeffs.CoeffsCallback
      ] = calc_coeffs.CoeffsCallback,
  ):
    super().__init__(transport_model, source_models, pedestal_model)
    self.callback_class = callback_class

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
    x_new_init = tuple(
        [core_profiles_t_plus_dt[name] for name in evolving_names]
    )

    # Instantiate coeffs_callback class
    coeffs_callback = self.callback_class(
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
    # Neither value impacts the final result, but needs to be the correct
    # type. x_new initialization (index 0) input is x_old for correct typing.
    # auxiliary_outputs (index 1) is a tuple of dataclasses with correct array
    # sizes for tracing.
    init_val = (
        x_new_init,
        (
            source_models_lib.build_all_zero_profiles(
                geo_t,
                self.source_models,
            ),
            state.CoreTransport.zeros(geo_t),
        ),
    )

    x_new, (core_sources, core_transport) = (
        predictor_corrector_method.predictor_corrector_method(
            dt=dt,
            static_runtime_params_slice=static_runtime_params_slice,
            dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
            geo_t_plus_dt=geo_t_plus_dt,
            x_old=x_old,
            core_profiles_t_plus_dt=core_profiles_t_plus_dt,
            init_val=init_val,
            coeffs_exp=coeffs_exp,
            coeffs_callback=coeffs_callback,
        )
    )

    stepper_numeric_outputs = state.StepperNumericOutputs(
        inner_solver_iterations=1,
        stepper_error_state=0,  # linear method always works
    )

    return x_new, core_sources, core_transport, stepper_numeric_outputs


def _default_linear_builder(
    transport_model: transport_model_lib.TransportModel,
    source_models: source_models_lib.SourceModels,
    pedestal_model: pedestal_model_lib.PedestalModel,
) -> LinearThetaMethod:
  return LinearThetaMethod(transport_model, source_models, pedestal_model)


# Type-alias so that users only need to import this file.
LinearRuntimeParams: TypeAlias = runtime_params_lib.RuntimeParams


@dataclasses.dataclass(kw_only=True)
class LinearThetaMethodBuilder(stepper_lib.StepperBuilder):
  """Builds a LinearThetaMethod."""

  builder: Callable[
      [
          transport_model_lib.TransportModel,
          source_models_lib.SourceModels,
          pedestal_model_lib.PedestalModel,
      ],
      LinearThetaMethod,
  ] = _default_linear_builder

  def __call__(
      self,
      transport_model: transport_model_lib.TransportModel,
      source_models: source_models_lib.SourceModels,
      pedestal_model: pedestal_model_lib.PedestalModel,
  ) -> LinearThetaMethod:
    return self.builder(transport_model, source_models, pedestal_model)

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

import copy
from typing import Type

import jax
from torax import calc_coeffs
from torax import config_slice
from torax import fvm
from torax import geometry
from torax import sim
from torax import state as state_module
from torax.sources import source_profiles
from torax.stepper import predictor_corrector_method
from torax.stepper import stepper as stepper_lib
from torax.transport_model import transport_model as transport_model_lib


class LinearThetaMethod(stepper_lib.Stepper):
  """Time step update using theta method, linearized on coefficients at t."""

  def __init__(
      self,
      transport_model: transport_model_lib.TransportModel,
      sources: source_profiles.Sources,
      callback_class: Type[sim.CoeffsCallback] = sim.CoeffsCallback,
  ):
    super().__init__(transport_model, sources)
    self.callback_class = callback_class

  def _x_new(
      self,
      state: state_module.State,
      geo: geometry.Geometry,
      dynamic_config_slice_t: config_slice.DynamicConfigSlice,
      dynamic_config_slice_t_plus_dt: config_slice.DynamicConfigSlice,
      static_config_slice: config_slice.StaticConfigSlice,
      evolving_names: tuple[str, ...],
      x_new_update_fns: tuple[fvm.CellVariableUpdateFn, ...],
      dt: jax.Array,
      mask: jax.Array,
      explicit_source_profiles: source_profiles.SourceProfiles,
  ) -> tuple[tuple[fvm.CellVariable, ...], int, calc_coeffs.AuxOutput]:
    """See Stepper._x_new docstring."""

    orig_evolving = copy.deepcopy([state[name] for name in evolving_names])
    orig_state = state

    # Instantiate coeffs_callback class
    coeffs_callback = self.callback_class(
        orig_state=orig_state,
        evolving_names=evolving_names,
        geo=geo,
        static_config_slice=static_config_slice,
        transport_model=self.transport_model,
        mask=mask,
        explicit_source_profiles=explicit_source_profiles,
        sources=self.sources,
    )

    # Compute the explicit coeffs based on the state at time t and all runtime
    # parameters at time t.
    x_old = tuple(orig_evolving)
    coeffs_exp = coeffs_callback(
        x_old, dynamic_config_slice_t, allow_pereverzev=True
    )

    # Calculate x_new with the predictor corrector method. Reverts to a
    # standard linear solve if static_config_slice.predictor_corrector=False.
    # init_val is the initialization for the predictor_corrector loop.
    # Neither value impacts the final result, but needs to be the correct
    # type. x_new initialization (index 0) input is x_old for correct typing.
    # auxiliary_outputs (index 1) is an AuxOutput dataclass with correct array
    # sizes for tracing
    init_val = (
        x_old,
        calc_coeffs.AuxOutput.build_from_geo(geo),
    )

    x_new, auxiliary_outputs = (
        predictor_corrector_method.predictor_corrector_method(
            init_val=init_val,
            x_new_update_fns=x_new_update_fns,
            dt=dt,
            coeffs_exp=coeffs_exp,
            coeffs_callback=coeffs_callback,
            dynamic_config_slice_t_plus_dt=dynamic_config_slice_t_plus_dt,
            static_config_slice=static_config_slice,
        )
    )

    error = 0  # linear method always works

    return x_new, error, auxiliary_outputs

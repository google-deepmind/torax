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

"""Tests torax.sim for handling time dependent input runtime params."""

import copy
import dataclasses
from typing import Callable, Literal

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from torax import sim as sim_lib
from torax import state
from torax.config import numerics as numerics_lib
from torax.config import profile_conditions as profile_conditions_lib
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.geometry import geometry_provider as geometry_provider_lib
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.pedestal_model import pedestal_model as pedestal_model_lib
from torax.pedestal_model import pydantic_model as pedestal_pydantic_model
from torax.sources import pydantic_model as source_pydantic_model
from torax.sources import source_models as source_models_lib
from torax.sources import source_profile_builders
from torax.sources import source_profiles
from torax.stepper import linear_theta_method
from torax.stepper import pydantic_model as stepper_pydantic_model
from torax.time_step_calculator import fixed_time_step_calculator
from torax.transport_model import transport_model as transport_model_lib


class SimWithTimeDependeceTest(parameterized.TestCase):
  """Integration tests for torax.sim with time-dependent runtime params."""

  @parameterized.named_parameters(
      ('with_adaptive_dt', True, 3, 0, 2.44444444444, [1, 2, 3]),
      ('without_adaptive_dt', False, 1, 1, 3.0, [4]),
  )
  def test_time_dependent_params_update_in_adaptive_dt(
      self,
      adaptive_dt: bool,
      expected_outer_stepper_iterations: int,
      expected_error_state: int,
      expected_combined_value: float,
      inner_solver_iterations: list[int],
  ):
    """Tests the SimulationStepFn's adaptive dt uses time-dependent params."""
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            Ti_bound_right={0.0: 1.0, 1.0: 2.0, 10.0: 11.0},
            ne_bound_right=0.5,
        ),
        numerics=numerics_lib.Numerics(
            adaptive_dt=adaptive_dt,
            fixed_dt=1.0,  # 1 time step in, the Ti_bound_right will be 2.0
            dt_reduction_factor=1.5,
        ),
    )
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    geometry_provider = geometry_provider_lib.ConstantGeometryProvider(geo)
    transport_builder = FakeTransportModelBuilder()
    # max combined value of Ti_bound_right should be 2.5. Higher will make the
    # error state from the stepper be 1.
    time_calculator = fixed_time_step_calculator.FixedTimeStepCalculator()
    sim = sim_lib.Sim.create(
        runtime_params=runtime_params,
        geometry_provider=geometry_provider,
        stepper=FakeStepperModel.from_dict(
            dict(
                param='Ti_bound_right',
                max_value=2.5,
                inner_solver_iterations=inner_solver_iterations,
            )
        ),
        transport_model_builder=transport_builder,
        sources=source_pydantic_model.Sources(),
        pedestal=pedestal_pydantic_model.Pedestal(),
        time_step_calculator=time_calculator,
    )
    sim_step_fn = sim.step_fn
    output_state, _ = sim_step_fn(
        static_runtime_params_slice=sim.static_runtime_params_slice,
        dynamic_runtime_params_slice_provider=sim.dynamic_runtime_params_slice_provider,
        geometry_provider=sim.geometry_provider,
        input_state=sim.initial_state,
    )
    # The initial step will not work, so it should take several adaptive time
    # steps to get under the Ti_bound_right threshold set above if adaptive_dt
    # was set to True.
    self.assertEqual(
        output_state.stepper_numeric_outputs.outer_stepper_iterations,
        expected_outer_stepper_iterations,
    )
    self.assertEqual(
        output_state.stepper_numeric_outputs.inner_solver_iterations,
        np.sum(inner_solver_iterations),
    )
    self.assertEqual(
        output_state.stepper_numeric_outputs.stepper_error_state,
        expected_error_state,
    )
    np.testing.assert_allclose(
        output_state.core_sources.qei.qei_coef, expected_combined_value
    )


class FakeStepperConfig(stepper_pydantic_model.LinearThetaMethod):
  """Fake stepper config that allows us to hook into the error logic."""
  stepper_type: Literal['fake'] = 'fake'
  param: str = 'Ti_bound_right'
  max_value: float = 2.5
  inner_solver_iterations: list[int] | None = None

  def build_stepper(
      self, transport_model, source_models, pedestal_model
  ) -> 'FakeStepper':
    return FakeStepper(
        param=self.param,
        max_value=self.max_value,
        inner_solver_iterations=self.inner_solver_iterations,
        transport_model=transport_model,
        source_models=source_models,
        pedestal_model=pedestal_model,
    )


class FakeStepperModel(stepper_pydantic_model.Stepper):
  """Config for a stepper."""

  stepper_config: FakeStepperConfig


class FakeStepper(linear_theta_method.LinearThetaMethod):
  """Fake stepper that allows us to hook into the error logic.

  Given the name of a time-dependent param in the runtime_params, and a max
  value for
  that param, this stepper returns a successful state if the config values for
  that param in the config at time t and config at time t+dt sum to less than
  max value.

  The number of inner solver iterations can also be specified.

  This stepper returns the input state as is and doesn't actually use the
  transport model or sources provided. They are given just to match the base
  class api.
  """

  def __init__(
      self,
      param: str,
      max_value: float,
      transport_model: transport_model_lib.TransportModel,
      source_models: source_models_lib.SourceModels,
      pedestal_model: pedestal_model_lib.PedestalModel,
      inner_solver_iterations: list[int] | None = None,
  ):
    self.transport_model = transport_model
    self.source_models = source_models
    self.pedestal_model = pedestal_model
    self._param = param
    self._max_value = max_value
    self._inner_solver_iterations = (
        copy.deepcopy(inner_solver_iterations)
        if inner_solver_iterations is not None
        else []
    )

  def __call__(
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
  ) -> tuple[
      state.CoreProfiles,
      source_profiles.SourceProfiles,
      state.CoreTransport,
      state.StepperNumericOutputs,
  ]:
    combined = getattr(
        dynamic_runtime_params_slice_t.profile_conditions, self._param
    ) + getattr(
        dynamic_runtime_params_slice_t_plus_dt.profile_conditions, self._param
    )
    pedestal_model_output = self.pedestal_model(
        dynamic_runtime_params_slice_t,
        geo_t,
        core_profiles_t,
    )
    transport = self.transport_model(
        dynamic_runtime_params_slice_t,
        geo_t,
        core_profiles_t,
        pedestal_model_output,
    )
    # Use Qei as a hacky way to extract what the combined value was.
    core_sources = source_profile_builders.build_all_zero_profiles(
        geo=geo_t,
    )
    core_sources = dataclasses.replace(
        core_sources,
        qei=dataclasses.replace(
            core_sources.qei, qei_coef=jnp.ones_like(geo_t.rho) * combined
        ),
    )

    current_inner_solver_iterations = (
        self._inner_solver_iterations.pop(0)
        if self._inner_solver_iterations
        else 1
    )

    def get_return_value(error_code: int):
      return (
          core_profiles_t,
          core_sources,
          transport,
          state.StepperNumericOutputs(
              outer_stepper_iterations=1,
              stepper_error_state=error_code,
              inner_solver_iterations=current_inner_solver_iterations,
          ),
      )

    return jax.lax.cond(
        combined < self._max_value,
        lambda: get_return_value(error_code=0),
        lambda: get_return_value(error_code=1),
    )


class FakeTransportModel(transport_model_lib.TransportModel):
  """Dummy transport model that always returns zeros."""

  def __init__(self):
    super().__init__()
    self._frozen = True

  def _call_implementation(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
  ) -> state.CoreTransport:
    return state.CoreTransport.zeros(geo)


def _default_fake_builder() -> FakeTransportModel:
  return FakeTransportModel()


@dataclasses.dataclass(kw_only=True)
class FakeTransportModelBuilder(transport_model_lib.TransportModelBuilder):
  """Builds a class FakeTransportModel."""

  builder: Callable[
      [],
      FakeTransportModel,
  ] = _default_fake_builder

  def __call__(
      self,
  ) -> FakeTransportModel:
    return self.builder()


if __name__ == '__main__':
  absltest.main()

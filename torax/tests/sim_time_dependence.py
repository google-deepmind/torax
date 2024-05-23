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

import dataclasses
from typing import Callable

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from torax import geometry
from torax import sim as sim_lib
from torax import state
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.sources import source_models as source_models_lib
from torax.sources import source_profiles
from torax.stepper import runtime_params as stepper_runtime_params
from torax.stepper import stepper as stepper_lib
from torax.time_step_calculator import fixed_time_step_calculator
from torax.transport_model import transport_model as transport_model_lib


class SimWithTimeDependeceTest(parameterized.TestCase):
  """Integration tests for torax.sim with time-dependent runtime params."""

  @parameterized.named_parameters(
      ('with_adaptive_dt', True, 3, 0, 2.44444444444),
      ('without_adaptive_dt', False, 1, 1, 3.0),
  )
  def test_time_dependent_params_update_in_adaptive_dt(
      self,
      adaptive_dt: bool,
      expected_stepper_iterations: int,
      expected_error_state: int,
      expected_combined_value: float,
  ):
    """Tests the SimulationStepFn's adaptive dt uses time-dependent params."""
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=general_runtime_params.ProfileConditions(
            Ti_bound_right={0.0: 1.0, 1.0: 2.0, 10.0: 11.0},
        ),
        numerics=general_runtime_params.Numerics(
            adaptive_dt=adaptive_dt,
            fixed_dt=1.0,  # 1 time step in, the Ti_bound_right will be 2.0
            dt_reduction_factor=1.5,
        ),
    )
    geo = geometry.build_circular_geometry()
    transport_builder = FakeTransportModelBuilder()
    transport = FakeTransportModel()
    source_models = source_models_lib.SourceModels()
    # max combined value of Ti_bound_right should be 2.5. Higher will make the
    # error state from the stepper be 1.
    stepper = FakeStepper(
        param='Ti_bound_right',
        max_value=2.5,
        transport_model=transport,
        source_models=source_models,
    )
    time_calculator = fixed_time_step_calculator.FixedTimeStepCalculator()
    sim_step_fn = sim_lib.SimulationStepFn(
        stepper,
        time_calculator,
        transport_model=transport,
    )
    dynamic_runtime_params_slice_provider = (
        runtime_params_slice.DynamicRuntimeParamsSliceProvider(
            runtime_params=runtime_params,
            transport_getter=lambda: transport_builder.runtime_params,
            sources_getter=lambda: source_models.runtime_params,
            stepper_getter=stepper_runtime_params.RuntimeParams,
        )
    )
    initial_dynamic_runtime_params_slice = (
        dynamic_runtime_params_slice_provider(runtime_params.numerics.t_initial)
    )
    input_state = sim_lib.get_initial_state(
        dynamic_runtime_params_slice=initial_dynamic_runtime_params_slice,
        geo=geo,
        time_step_calculator=time_calculator,
        source_models=source_models,
    )
    output_state = sim_step_fn(
        static_runtime_params_slice=runtime_params_slice.build_static_runtime_params_slice(
            runtime_params
        ),
        dynamic_runtime_params_slice_provider=dynamic_runtime_params_slice_provider,
        geo=geo,
        input_state=input_state,
        explicit_source_profiles=source_models_lib.build_source_profiles(
            source_models=source_models,
            dynamic_runtime_params_slice=initial_dynamic_runtime_params_slice,
            geo=geo,
            core_profiles=input_state.core_profiles,
            explicit=True,
        ),
    )
    # The initial step will not work, so it should take several adaptive time
    # steps to get under the Ti_bound_right threshold set above if adaptive_dt
    # was set to True.
    self.assertEqual(
        output_state.stepper_iterations, expected_stepper_iterations
    )
    self.assertEqual(output_state.stepper_error_state, expected_error_state)
    np.testing.assert_allclose(
        output_state.core_sources.qei.qei_coef, expected_combined_value
    )


class FakeStepper(stepper_lib.Stepper):
  """Fake stepper that allows us to hook into the error logic.

  Given the name of a time-dependent param in the runtime_params, and a max
  value for
  that param, this stepper returns a successful state if the config values for
  that param in the config at time t and config at time t+dt sum to less than
  max value.

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
  ):
    self.transport_model = transport_model
    self.source_models = source_models
    self._param = param
    self._max_value = max_value

  def __call__(
      self,
      dt: jax.Array,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice_t: runtime_params_slice.DynamicRuntimeParamsSlice,
      dynamic_runtime_params_slice_t_plus_dt: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles_t: state.CoreProfiles,
      core_profiles_t_plus_dt: state.CoreProfiles,
      explicit_source_profiles: source_profiles.SourceProfiles,
  ) -> tuple[
      state.CoreProfiles,
      source_profiles.SourceProfiles,
      state.CoreTransport,
      int,
  ]:
    combined = getattr(
        dynamic_runtime_params_slice_t.profile_conditions, self._param
    ) + getattr(
        dynamic_runtime_params_slice_t_plus_dt.profile_conditions, self._param
    )
    transport = self.transport_model(
        dynamic_runtime_params_slice_t, geo, core_profiles_t
    )
    # Use Qei as a hacky way to extract what the combined value was.
    core_sources = source_models_lib.build_all_zero_profiles(
        geo=geo,
        source_models=self.source_models,
    )
    core_sources = dataclasses.replace(
        core_sources,
        qei=dataclasses.replace(
            core_sources.qei, qei_coef=jnp.ones_like(geo.r) * combined
        ),
    )
    return jax.lax.cond(
        combined < self._max_value,
        lambda: (core_profiles_t, core_sources, transport, 0),
        lambda: (core_profiles_t, core_sources, transport, 1),
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

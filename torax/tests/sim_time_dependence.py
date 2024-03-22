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

"""Tests torax.sim for handling time dependent input config params."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from torax import calc_coeffs
from torax import config as config_lib
from torax import config_slice
from torax import geometry
from torax import initial_states
from torax import sim as sim_lib
from torax import state as state_module
from torax.sources import source_profiles
from torax.stepper import stepper as stepper_lib
from torax.time_step_calculator import fixed_time_step_calculator
from torax.transport_model import transport_model as transport_model_lib


class SimWithTimeDependeceTest(parameterized.TestCase):
  """Integration tests for torax.sim with time-dependent config params."""

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
    config = config_lib.Config(
        Ti_bound_right={0.0: 1.0, 1.0: 2.0, 10.0: 11.0},
        adaptive_dt=adaptive_dt,
        fixed_dt=1.0,  # 1 time step in, the Ti_bound_right will be 2.0
        dt_reduction_factor=1.5,
    )
    geo = geometry.build_circular_geometry(config)
    transport = FakeTransportModel()
    sources = source_profiles.Sources()
    # max combined value of Ti_bound_right should be 2.5. Higher will make the
    # error state from the stepper be 1.
    stepper = FakeStepper(
        param='Ti_bound_right',
        max_value=2.5,
        transport_model=transport,
        sources=sources,
    )
    time_calculator = fixed_time_step_calculator.FixedTimeStepCalculator()
    sim_step_fn = sim_lib.SimulationStepFn(
        stepper,
        time_calculator,
        transport_model=FakeTransportModel(),
    )
    input_state = initial_states.get_initial_sim_state(
        config=config,
        geo=geo,
        time_step_calculator=time_calculator,
        sources=sources,
    )
    dynamic_config_slice_provider = (
        config_slice.TimeDependentDynamicConfigSliceProvider(config)
    )
    initial_dynamic_config_slice = dynamic_config_slice_provider(
        config.t_initial
    )
    output_state, _ = sim_step_fn(
        input_state=input_state,
        geo=geo,
        dynamic_config_slice_provider=dynamic_config_slice_provider,
        static_config_slice=config_slice.build_static_config_slice(config),
        explicit_source_profiles=source_profiles.build_source_profiles(
            sources=sources,
            dynamic_config_slice=initial_dynamic_config_slice,
            geo=geo,
            sim_state=input_state,
            explicit=True,
        ),
    )
    # The initial step will not work, so it should take several adaptive time
    # steps to get under the Ti_bound_right threshold set above if adaptive_dt
    # was set to True.
    self.assertEqual(
        output_state.state.stepper_iterations, expected_stepper_iterations
    )
    self.assertEqual(
        output_state.state.stepper_error_state, expected_error_state
    )
    np.testing.assert_allclose(output_state.aux.Qei, expected_combined_value)


class FakeStepper(stepper_lib.Stepper):
  """Fake stepper that allows us to hook into the error logic.

  Given the name of a time-dependent param in the config, and a max value for
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
      sources: source_profiles.Sources,
  ):
    self.transport_model = transport_model
    self.sources = sources
    self._param = param
    self._max_value = max_value

  def __call__(
      self,
      sim_state_t: state_module.ToraxSimState,
      sim_state_t_plus_dt: state_module.ToraxSimState,
      geo: geometry.Geometry,
      dynamic_config_slice_t: config_slice.DynamicConfigSlice,
      dynamic_config_slice_t_plus_dt: config_slice.DynamicConfigSlice,
      static_config_slice: config_slice.StaticConfigSlice,
      dt: jax.Array,
      explicit_source_profiles: source_profiles.SourceProfiles,
  ) -> tuple[state_module.State, int, calc_coeffs.AuxOutput]:
    combined = getattr(dynamic_config_slice_t, self._param) + getattr(
        dynamic_config_slice_t_plus_dt, self._param
    )
    # Use Qei as a hacky way to extract what the combined value was.
    aux = calc_coeffs.AuxOutput.build_from_geo(geo)
    aux.Qei = jnp.ones_like(geo.r) * combined
    return jax.lax.cond(
        combined < self._max_value,
        lambda: (sim_state_t.mesh_state, 0, aux),
        lambda: (sim_state_t.mesh_state, 1, aux),
    )


class FakeTransportModel(transport_model_lib.TransportModel):

  def _call_implementation(
      self,
      dynamic_config_slice: config_slice.DynamicConfigSlice,
      geo: geometry.Geometry,
      state: state_module.State,
  ) -> transport_model_lib.TransportCoeffs:
    return transport_model_lib.TransportCoeffs(
        chi_face_ion=jnp.zeros_like(geo.r_face),
        chi_face_el=jnp.zeros_like(geo.r_face),
        d_face_el=jnp.zeros_like(geo.r_face),
        v_face_el=jnp.zeros_like(geo.r_face),
    )


if __name__ == '__main__':
  absltest.main()

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
from typing import Literal
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from torax import sim
from torax import state
from torax.config import build_runtime_params
from torax.config import runtime_params_slice
from torax.fvm import cell_variable
from torax.geometry import geometry
from torax.geometry import geometry_provider as geometry_provider_lib
from torax.orchestration import run_simulation
from torax.orchestration import sim_state
from torax.orchestration import step_function
from torax.output_tools import post_processing
from torax.pedestal_model import pedestal_model as pedestal_model_lib
from torax.sources import source_models as source_models_lib
from torax.sources import source_profile_builders
from torax.sources import source_profiles
from torax.stepper import linear_theta_method
from torax.stepper import pydantic_model as stepper_pydantic_model
from torax.torax_pydantic import model_config
from torax.transport_model import pydantic_model_base as transport_pydantic_model_base
from torax.transport_model import transport_model as transport_model_lib


class SimWithTimeDependenceTest(parameterized.TestCase):
  """Integration tests for torax.sim with time-dependent runtime params."""

  def setUp(self):
    super().setUp()
    # Register the fake transport config.
    model_config.ToraxConfig.model_fields[
        'transport'
    ].annotation |= FakeTransportConfig
    model_config.ToraxConfig.model_fields[
        'solver'
    ].annotation |= FakeSolverConfig
    model_config.ToraxConfig.model_rebuild(force=True)

  @parameterized.named_parameters(
      ('with_adaptive_dt', True, 3, 0, 2.44444444444, [1, 2, 3]),
      ('without_adaptive_dt', False, 1, 1, 3.0, [4]),
  )
  def test_time_dependent_params_update_in_adaptive_dt(
      self,
      adaptive_dt: bool,
      expected_outer_solver_iterations: int,
      expected_error_state: int,
      expected_combined_value: float,
      inner_solver_iterations: list[int],
  ):
    """Tests the SimulationStepFn's adaptive dt uses time-dependent params."""

    config = {
        'profile_conditions': {
            'T_i_right_bc': {0.0: 1.0, 1.0: 2.0, 10.0: 11.0},
            'n_e_right_bc': 0.5,
        },
        'numerics': {
            'adaptive_dt': adaptive_dt,
            # 1 time step in, the T_i_right_bc will be 2.0
            'fixed_dt': 1.0,
            'dt_reduction_factor': 1.5,
            't_final': 1.0,
        },
        'plasma_composition': {},
        'geometry': {
            'geometry_type': 'circular',
        },
        'transport': {'transport_model': 'fake'},
        'solver': {
            'solver_type': 'fake',
            'inner_solver_iterations': inner_solver_iterations,
        },
        'pedestal': {},
        'time_step_calculator': {'calculator_type': 'fixed'},
        'sources': {},
    }
    torax_config = model_config.ToraxConfig.from_dict(config)

    def _fake_sim_run_simulation(
        static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
        dynamic_runtime_params_slice_provider: build_runtime_params.DynamicRuntimeParamsSliceProvider,
        geometry_provider: geometry_provider_lib.GeometryProvider,
        initial_state: sim_state.ToraxSimState,
        initial_post_processed_outputs: post_processing.PostProcessedOutputs,
        step_fn: step_function.SimulationStepFn,
        restart_case: bool,
        log_timestep_info: bool = False,
        progress_bar: bool = True,
    ) -> tuple[
        tuple[sim_state.ToraxSimState, ...],
        tuple[post_processing.PostProcessedOutputs, ...],
        state.SimError,
    ]:
      del log_timestep_info, progress_bar, restart_case
      output_state, post_processed_outputs, error = step_fn(
          static_runtime_params_slice,
          dynamic_runtime_params_slice_provider,
          geometry_provider,
          initial_state,
          initial_post_processed_outputs,
      )
      self.assertEqual(
          output_state.solver_numeric_outputs.outer_solver_iterations,
          expected_outer_solver_iterations,
      )
      self.assertEqual(
          output_state.solver_numeric_outputs.inner_solver_iterations,
          np.sum(inner_solver_iterations),
      )
      self.assertEqual(
          output_state.solver_numeric_outputs.solver_error_state,
          expected_error_state,
      )
      np.testing.assert_allclose(
          output_state.core_sources.qei.qei_coef, expected_combined_value
      )
      return (output_state,), (post_processed_outputs,), error

    with mock.patch.object(
        sim, '_run_simulation', wraps=_fake_sim_run_simulation
    ) as mock_run_simulation:
      run_simulation.run_simulation(torax_config)
    # The initial step will not work, so it should take several adaptive time
    # steps to get under the T_i_right_bc threshold set above if adaptive_dt
    # was set to True.
    mock_run_simulation.assert_called_once()


class FakeSolverConfig(stepper_pydantic_model.LinearThetaMethod):
  """Fake solver config that allows us to hook into the error logic."""

  solver_type: Literal['fake'] = 'fake'
  param: str = 'T_i_right_bc'
  max_value: float = 2.5
  inner_solver_iterations: list[int] | None = None

  def build_solver(
      self,
      static_runtime_params_slice,
      transport_model,
      source_models,
      pedestal_model,
  ) -> 'FakeSolver':
    return FakeSolver(
        param=self.param,
        max_value=self.max_value,
        static_runtime_params_slice=static_runtime_params_slice,
        transport_model=transport_model,
        source_models=source_models,
        pedestal_model=pedestal_model,
        inner_solver_iterations=self.inner_solver_iterations,
    )


class FakeSolver(linear_theta_method.LinearThetaMethod):
  """Fake solver that allows us to hook into the error logic.

  Given the name of a time-dependent param in the runtime_params, and a max
  value for
  that param, this solver returns a successful state if the config values for
  that param in the config at time t and config at time t+dt sum to less than
  max value.

  The number of inner solver iterations can also be specified.

  This solver returns the input state as is and doesn't actually use the
  transport model or sources provided. They are given just to match the base
  class api.
  """

  def __init__(
      self,
      param: str,
      max_value: float,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      transport_model: transport_model_lib.TransportModel,
      source_models: source_models_lib.SourceModels,
      pedestal_model: pedestal_model_lib.PedestalModel,
      inner_solver_iterations: list[int] | None = None,
  ):
    self.static_runtime_params_slice = static_runtime_params_slice
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
    self.evolving_names = ()

  def __call__(
      self,
      t: jax.Array,
      dt: jax.Array,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice_t: runtime_params_slice.DynamicRuntimeParamsSlice,
      dynamic_runtime_params_slice_t_plus_dt: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo_t: geometry.Geometry,
      geo_t_plus_dt: geometry.Geometry,
      core_profiles_t: state.CoreProfiles,
      core_profiles_t_plus_dt: state.CoreProfiles,
      core_sources_t: source_profiles.SourceProfiles,
      core_transport_t: state.CoreTransport,
      explicit_source_profiles: source_profiles.SourceProfiles,
  ) -> tuple[
      tuple[cell_variable.CellVariable, ...],
      sim_state.ToraxSimState,
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
      intermediate_state = sim_state.ToraxSimState(
          t=t + dt,
          dt=dt,
          core_profiles=core_profiles_t_plus_dt,
          core_transport=transport,
          core_sources=core_sources,
          geometry=geo_t_plus_dt,
          solver_numeric_outputs=state.SolverNumericOutputs(
              outer_solver_iterations=1,
              solver_error_state=error_code,
              inner_solver_iterations=current_inner_solver_iterations,
          ),
      )
      return (), intermediate_state

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

  def __hash__(self) -> int:
    return hash(self.__class__.__name__)

  def __eq__(self, other) -> bool:
    return isinstance(other, type(self))


class FakeTransportConfig(transport_pydantic_model_base.TransportBase):
  """Fake transport config for a model that always returns zeros."""

  transport_model: Literal['fake'] = 'fake'

  def build_transport_model(self) -> FakeTransportModel:
    return FakeTransportModel()


if __name__ == '__main__':
  absltest.main()

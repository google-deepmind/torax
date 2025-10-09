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
import dataclasses
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
from torax._src import jax_utils
from torax._src.orchestration import adaptive_step
from torax._src.orchestration import run_simulation
from torax._src.sources import source_profile_builders
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config


class AdaptiveStepTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    torax_config = model_config.ToraxConfig.from_dict(
        default_configs.get_default_config_dict()
    )
    (
        self.params_provider,
        self.sim_state,
        self.post_processed_outputs,
        self.step_fn,
    ) = run_simulation.prepare_simulation(torax_config)
    self.runtime_params = self.params_provider(torax_config.numerics.t_initial)
    self.geo = self.sim_state.geometry

  def test_create_initial_state_smoke(self):
    initial_dt = 0.1
    initial_state = adaptive_step.create_initial_state(
        input_state=self.sim_state,
        evolving_names=self.runtime_params.numerics.evolving_names,
        initial_dt=initial_dt,
        runtime_params_t=self.runtime_params,
        geo_t=self.geo,
    )
    self.assertIsInstance(initial_state, adaptive_step.AdaptiveStepState)
    self.assertFalse(initial_state.solver_numeric_outputs.sawtooth_crash)
    self.assertEqual(
        initial_state.solver_numeric_outputs.solver_error_state,
        jnp.array(1, jax_utils.get_int_dtype()),
    )
    self.assertEqual(
        initial_state.solver_numeric_outputs.outer_solver_iterations,
        jnp.array(0, jax_utils.get_int_dtype()),
    )
    self.assertEqual(
        initial_state.solver_numeric_outputs.inner_solver_iterations,
        jnp.array(0, jax_utils.get_int_dtype()),
    )
    self.assertLen(
        initial_state.x_new, len(self.runtime_params.numerics.evolving_names)
    )

  def test_compute_state_smoke(self):
    initial_dt = 0.1
    loop_statistics = {
        'inner_solver_iterations': jnp.array(0, jax_utils.get_int_dtype()),
    }
    explicit_source_profiles = source_profile_builders.build_source_profiles(
        runtime_params=self.runtime_params,
        geo=self.geo,
        core_profiles=self.sim_state.core_profiles,
        source_models=self.step_fn.solver.physics_models.source_models,
        neoclassical_models=self.step_fn.solver.physics_models.neoclassical_models,
        explicit=True,
    )
    adaptive_step_state, _ = adaptive_step.compute_state(
        i=0,
        loop_statistics=loop_statistics,
        initial_dt=initial_dt,
        runtime_params_t=self.runtime_params,
        geo_t=self.geo,
        input_state=self.sim_state,
        explicit_source_profiles=explicit_source_profiles,
        runtime_params_provider=self.step_fn.runtime_params_provider,
        geometry_provider=self.step_fn.geometry_provider,
        solver=self.step_fn.solver,
    )
    self.assertIsInstance(adaptive_step_state, adaptive_step.AdaptiveStepState)

  @parameterized.named_parameters(
      dict(
          testcase_name='solver_converged',
          solver_error_state=0,
          dt=0.1,
          runtime_params_numerics_updates={},
          sim_state_updates={},
          expected=False,
      ),
      dict(
          testcase_name='solver_did_not_converge_dt_not_too_small',
          solver_error_state=1,
          dt=0.05,
          runtime_params_numerics_updates={'min_dt': 0.01},
          sim_state_updates={},
          expected=True,
      ),
      dict(
          testcase_name='solver_did_not_converge_dt_too_small',
          solver_error_state=1,
          dt=0.05,
          runtime_params_numerics_updates={'min_dt': 0.1},
          sim_state_updates={},
          expected=False,
      ),
      dict(
          testcase_name='solver_did_not_converge_dt_too_small_at_t_final',
          solver_error_state=1,
          dt=0.1,
          runtime_params_numerics_updates={
              'exact_t_final': True,
              't_final': 0.1,
              'min_dt': 0.2,
          },
          sim_state_updates={'t': 0.0},
          expected=True,
      ),
      dict(
          testcase_name='dt_is_nan',
          solver_error_state=1,
          dt=jnp.nan,
          runtime_params_numerics_updates={},
          sim_state_updates={},
          expected=False,
      ),
  )
  def test_cond_fun(
      self,
      solver_error_state,
      dt,
      runtime_params_numerics_updates,
      sim_state_updates,
      expected,
  ):
    initial_dt = 0.1
    base_adaptive_step_state = adaptive_step.create_initial_state(
        input_state=self.sim_state,
        evolving_names=self.runtime_params.numerics.evolving_names,
        initial_dt=initial_dt,
        runtime_params_t=self.runtime_params,
        geo_t=self.geo,
    )

    runtime_params = self.runtime_params
    if runtime_params_numerics_updates:
      runtime_params = dataclasses.replace(
          self.runtime_params,
          numerics=dataclasses.replace(
              self.runtime_params.numerics, **runtime_params_numerics_updates
          ),
      )

    sim_state = self.sim_state
    if sim_state_updates:
      sim_state = dataclasses.replace(self.sim_state, **sim_state_updates)

    adaptive_step_state = dataclasses.replace(
        base_adaptive_step_state,
        dt=jnp.asarray(dt),
        solver_numeric_outputs=dataclasses.replace(
            base_adaptive_step_state.solver_numeric_outputs,
            solver_error_state=jnp.array(
                solver_error_state, jax_utils.get_int_dtype()
            ),
        ),
    )

    self.assertEqual(
        expected,
        adaptive_step.cond_fun(
            adaptive_step_state,
            initial_dt,
            runtime_params,
            mock.ANY,
            sim_state,
            mock.ANY,
            mock.ANY,
            mock.ANY,
        ),
    )


if __name__ == '__main__':
  absltest.main()

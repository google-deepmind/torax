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
import copy
import dataclasses

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from torax._src import state
from torax._src.orchestration import run_simulation
from torax._src.orchestration import step_function
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config


class StepFunctionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    torax_config = model_config.ToraxConfig.from_dict(
        default_configs.get_default_config_dict()
    )
    (
        self.params_provider,
        self.sim_state,
        self.post_processed_outputs,
        _,
    ) = run_simulation.prepare_simulation(torax_config)
    self.runtime_params = self.params_provider(torax_config.numerics.t_initial)

  def test_no_error(self):
    error = step_function.check_for_errors(
        self.params_provider.numerics,
        self.sim_state,
        self.post_processed_outputs,
    )
    self.assertEqual(error, state.SimError.NO_ERROR)

  def test_nan_in_bc(self):
    core_profiles = dataclasses.replace(
        self.sim_state.core_profiles,
        T_i=dataclasses.replace(
            self.sim_state.core_profiles.T_i,
            right_face_constraint=jnp.array(jnp.nan),
        ),
    )
    new_sim_state_core_profiles = dataclasses.replace(
        self.sim_state, core_profiles=core_profiles
    )
    error = step_function.check_for_errors(
        self.params_provider.numerics,
        new_sim_state_core_profiles,
        self.post_processed_outputs,
    )
    self.assertEqual(error, state.SimError.NAN_DETECTED)

  def test_nan_in_post_processed_outputs(self):
    new_post_processed_outputs = dataclasses.replace(
        self.post_processed_outputs,
        P_aux_total=jnp.array(jnp.nan),
    )
    error = step_function.check_for_errors(
        self.params_provider.numerics,
        self.sim_state,
        new_post_processed_outputs,
    )
    self.assertEqual(error, state.SimError.NAN_DETECTED)

  def test_nan_in_source_array(self):
    nan_array = np.zeros_like(self.sim_state.geometry.rho)
    nan_array[-1] = np.nan
    bootstrap_current = dataclasses.replace(
        self.sim_state.core_sources.bootstrap_current,
        j_bootstrap=nan_array,
    )
    new_core_sources = dataclasses.replace(
        self.sim_state.core_sources, bootstrap_current=bootstrap_current
    )
    new_sim_state_sources = dataclasses.replace(
        self.sim_state, core_sources=new_core_sources
    )
    error = step_function.check_for_errors(
        self.params_provider.numerics,
        new_sim_state_sources,
        self.post_processed_outputs,
    )
    self.assertEqual(error, state.SimError.NAN_DETECTED)

  def test_below_min_dt(self):
    numerics = copy.deepcopy(self.params_provider.numerics)
    numerics._update_fields({'min_dt': 2.0, 'dt_reduction_factor': 2.0})

    new_sim_state = dataclasses.replace(
        self.sim_state,
        dt=jnp.array(3.0),
        solver_numeric_outputs=state.SolverNumericOutputs(
            solver_error_state=1,
            outer_solver_iterations=0,
            inner_solver_iterations=0,
        ),
    )
    error = step_function.check_for_errors(
        numerics, new_sim_state, self.post_processed_outputs,
    )
    self.assertEqual(error, state.SimError.REACHED_MIN_DT)

  def test_no_error_when_below_min_dt_but_solver_converged(self):
    numerics = copy.deepcopy(self.params_provider.numerics)
    numerics._update_fields({
        't_final': 5.0,
        'exact_t_final': True,
        'min_dt': 2.0,
        'dt_reduction_factor': 2.0,
    })
    new_sim_state = dataclasses.replace(
        self.sim_state,
        dt=jnp.array(1.0),
        t=jnp.array(5.0),
        solver_numeric_outputs=state.SolverNumericOutputs(
            solver_error_state=0,
            outer_solver_iterations=0,
            inner_solver_iterations=0,
        ),
    )
    error = step_function.check_for_errors(
        numerics, new_sim_state, self.post_processed_outputs,
    )
    self.assertEqual(error, state.SimError.NO_ERROR)


if __name__ == '__main__':
  absltest.main()

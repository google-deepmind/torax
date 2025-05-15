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

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from torax import state
from torax._src.fvm import cell_variable
from torax._src.geometry import pydantic_model as geometry_pydantic_model
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.orchestration import sim_state
from torax._src.orchestration import step_function
from torax._src.output_tools import post_processing
from torax._src.sources import source_profiles as source_profiles_lib
from torax.tests.test_lib import core_profile_helpers


class StepFunctionTest(absltest.TestCase):

  def test_check_for_errors(self):
    t = jnp.array(0.0)
    dt = jnp.array(0.1)
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    source_profiles = source_profiles_lib.SourceProfiles(
        bootstrap_current=bootstrap_current_base.BootstrapCurrent.zeros(geo),
        qei=source_profiles_lib.QeiInfo.zeros(geo),
    )
    dummy_cell_variable = cell_variable.CellVariable(
        value=jnp.zeros_like(geo.rho),
        dr=geo.drho_norm,
        right_face_constraint=jnp.ones(()),
        right_face_grad_constraint=None,
    )
    core_profiles = core_profile_helpers.make_zero_core_profiles(geo)
    torax_state = sim_state.ToraxSimState(
        core_profiles=core_profiles,
        core_transport=state.CoreTransport.zeros(geo),
        core_sources=source_profiles,
        t=t,
        dt=dt,
        solver_numeric_outputs=state.SolverNumericOutputs(
            outer_solver_iterations=1,
            solver_error_state=1,
            inner_solver_iterations=1,
        ),
        geometry=geo,
    )
    post_processed_outputs = post_processing.PostProcessedOutputs.zeros(geo)

    with self.subTest('no NaN'):
      error = torax_state.check_for_errors()
      self.assertEqual(error, state.SimError.NO_ERROR)
      error = step_function._check_for_errors(
          torax_state, post_processed_outputs)
      self.assertEqual(error, state.SimError.NO_ERROR)

    with self.subTest('NaN in BC'):
      core_profiles = dataclasses.replace(
          core_profiles,
          T_i=dataclasses.replace(
              core_profiles.T_i,
              right_face_constraint=jnp.array(jnp.nan),
          ),
      )
      new_sim_state_core_profiles = dataclasses.replace(
          torax_state, core_profiles=core_profiles
      )
      error = new_sim_state_core_profiles.check_for_errors()
      self.assertEqual(error, state.SimError.NAN_DETECTED)
      error = step_function._check_for_errors(
          new_sim_state_core_profiles, post_processed_outputs
      )
      self.assertEqual(error, state.SimError.NAN_DETECTED)

    with self.subTest('NaN in post processed outputs'):
      new_post_processed_outputs = dataclasses.replace(
          post_processed_outputs,
          P_aux_total=jnp.array(jnp.nan),
      )
      error = new_post_processed_outputs.check_for_errors()
      self.assertEqual(error, state.SimError.NAN_DETECTED)
      error = step_function._check_for_errors(
          torax_state, new_post_processed_outputs
      )
      self.assertEqual(error, state.SimError.NAN_DETECTED)

    with self.subTest('NaN in one element of source array'):
      nan_array = np.zeros_like(geo.rho)
      nan_array[-1] = np.nan
      bootstrap_current = dataclasses.replace(
          torax_state.core_sources.bootstrap_current,
          j_bootstrap=nan_array,
      )
      new_core_sources = dataclasses.replace(
          torax_state.core_sources, bootstrap_current=bootstrap_current
      )
      new_sim_state_sources = dataclasses.replace(
          torax_state, core_sources=new_core_sources
      )
      error = new_sim_state_sources.check_for_errors()
      self.assertEqual(error, state.SimError.NAN_DETECTED)
      error = step_function._check_for_errors(
          new_sim_state_sources, post_processed_outputs
      )
      self.assertEqual(error, state.SimError.NAN_DETECTED)


if __name__ == '__main__':
  absltest.main()

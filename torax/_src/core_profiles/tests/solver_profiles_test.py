# Copyright 2026 DeepMind Technologies Limited
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

"""Tests verifying construction, validation, and slicing of SolverProfiles."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from torax._src import jax_utils
from torax._src.core_profiles import solver_profiles
from torax._src.core_profiles import solver_state as solver_state_lib
from torax._src.physics import psi_calculations
from torax._src.test_utils import core_profile_helpers
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import config


class SolverProfilesTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.torax_config = config.ToraxConfig.from_dict(
        default_configs.get_default_config_dict()
    )
    self.core_profiles_t0, self.runtime_params, self.geo = (
        core_profile_helpers.make_initial_inputs(self.torax_config)
    )
    self.solver_state = solver_state_lib.SolverState.from_core_profiles(
        self.core_profiles_t0
    )

  def test_from_solver_state_structure_and_values(self):
    result = solver_profiles.SolverProfiles.from_solver_state(
        runtime_params=self.runtime_params,
        geo=self.geo,
        solver_state=self.solver_state,
    )

    self.assertIsInstance(result, solver_profiles.SolverProfiles)
    np.testing.assert_allclose(
        result.kinetic.T_i.value,
        self.solver_state.T_i.value,
    )
    np.testing.assert_allclose(
        result.kinetic.T_e.value,
        self.solver_state.T_e.value,
    )
    np.testing.assert_allclose(
        result.kinetic.n_e.value,
        self.solver_state.n_e.value,
    )
    np.testing.assert_allclose(
        result.flux.psi.value,
        self.solver_state.psi.value,
    )
    np.testing.assert_allclose(
        result.toroidal_angular_velocity.value,
        self.solver_state.toroidal_angular_velocity.value,
    )
    np.testing.assert_allclose(
        result.flux.v_loop_lcfs,
        jnp.asarray(0.0, dtype=jax_utils.get_dtype()),
    )

    expected_q_face = psi_calculations.calc_q_face(
        self.geo, self.solver_state.psi
    )
    np.testing.assert_allclose(result.flux.q_face, expected_q_face)
    self.assertTrue(result.quasineutrality_satisfied())

  def test_from_solver_state_custom_v_loop_lcfs(self):
    custom_v_loop = jnp.asarray(1.75, dtype=jax_utils.get_dtype())
    result = solver_profiles.SolverProfiles.from_solver_state(
        runtime_params=self.runtime_params,
        geo=self.geo,
        solver_state=self.solver_state,
        v_loop_lcfs=custom_v_loop,
    )
    np.testing.assert_allclose(result.flux.v_loop_lcfs, custom_v_loop)

  def test_from_solver_state_jittable(self):
    @jax.jit
    def jitted_builder(solver_state):
      return solver_profiles.SolverProfiles.from_solver_state(
          runtime_params=self.runtime_params,
          geo=self.geo,
          solver_state=solver_state,
      )

    result = jitted_builder(self.solver_state)
    self.assertIsInstance(result, solver_profiles.SolverProfiles)
    np.testing.assert_allclose(
        result.kinetic.T_i.value,
        self.solver_state.T_i.value,
    )


if __name__ == "__main__":
  absltest.main()

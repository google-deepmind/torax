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

"""Tests for core_profiles builders."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from torax._src import jax_utils
from torax._src import state
from torax._src.core_profiles import builders
from torax._src.physics import psi_calculations
from torax._src.test_utils import core_profile_helpers
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import config


class BuildersTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    torax_config = config.ToraxConfig.from_dict(
        default_configs.get_default_config_dict()
    )
    core_profiles, self.runtime_params, self.geo = (
        core_profile_helpers.make_initial_inputs(torax_config)
    )
    self.solver_state = state.SolverState.from_core_profiles(
        core_profiles
    )

  def test_build_base_core_profiles_structure_and_values(self):
    result = builders.build_base_core_profiles(
        runtime_params=self.runtime_params,
        geo=self.geo,
        solver_state=self.solver_state,
    )

    self.assertIsInstance(result, state.BaseCoreProfiles)
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

    # Check that safety factor and total current match direct calculations.
    expected_q_face = psi_calculations.calc_q_face(
        self.geo, self.solver_state.psi
    )
    np.testing.assert_allclose(result.flux.q_face, expected_q_face)

    # Check compatibility property access on BaseCoreProfiles.
    np.testing.assert_allclose(
        result.T_i.value,
        self.solver_state.T_i.value,
    )
    np.testing.assert_allclose(
        result.psi.value,
        self.solver_state.psi.value,
    )
    self.assertTrue(result.quasineutrality_satisfied())

  def test_build_base_core_profiles_custom_v_loop_lcfs(self):
    custom_v_loop = jnp.asarray(1.75, dtype=jax_utils.get_dtype())
    result = builders.build_base_core_profiles(
        runtime_params=self.runtime_params,
        geo=self.geo,
        solver_state=self.solver_state,
        v_loop_lcfs=custom_v_loop,
    )
    np.testing.assert_allclose(result.flux.v_loop_lcfs, custom_v_loop)

  def test_build_base_core_profiles_jittable(self):
    @jax.jit
    def jitted_builder(solver_state):
      return builders.build_base_core_profiles(
          runtime_params=self.runtime_params,
          geo=self.geo,
          solver_state=solver_state,
      )

    result = jitted_builder(self.solver_state)
    self.assertIsInstance(result, state.BaseCoreProfiles)
    np.testing.assert_allclose(
        result.kinetic.T_i.value,
        self.solver_state.T_i.value,
    )


if __name__ == '__main__':
  absltest.main()

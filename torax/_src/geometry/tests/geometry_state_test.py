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

"""Unit tests for GeometryState and update_geometry_state."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from torax._src.geometry import circular_geometry
from torax._src.geometry import geometry_state
from torax._src.output_tools import output_grid_context
from torax._src.output_tools import output_keys


class GeometryStateTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.geo_t = (
        circular_geometry.CircularConfig(
            n_rho=4,
            R_major=6.2,
            a_minor=2.0,
            B_0=5.3,
        )
        .build_runtime_params(0.0)
        .geometry
    )
    # Create geo_t_plus_dt with a different magnetic field so Phi_b changes.
    self.geo_t_plus_dt = (
        circular_geometry.CircularConfig(
            n_rho=4,
            R_major=6.2,
            a_minor=2.0,
            B_0=5.5,
        )
        .build_runtime_params(0.0)
        .geometry
    )
    self.initial_state = geometry_state.GeometryState(
        Phi_b=jnp.asarray(self.geo_t.Phi_b, dtype=jnp.float64)
    )

  def test_from_geometry_default_phi_b_dot(self):
    state = geometry_state.GeometryState.from_geometry(self.geo_t)
    chex.assert_trees_all_close(state.Phi_b, self.geo_t.Phi_b)
    chex.assert_trees_all_close(state.Phi_b_dot, 0.0)

  def test_from_geometry_with_phi_b_dot(self):
    state = geometry_state.GeometryState.from_geometry(
        self.geo_t, Phi_b_dot=jnp.asarray(1.5)
    )
    chex.assert_trees_all_close(state.Phi_b, self.geo_t.Phi_b)
    chex.assert_trees_all_close(state.Phi_b_dot, 1.5)

  def test_update_geometry_state_calc_false(self):
    state = geometry_state.update_geometry_state(
        geometry_state=self.initial_state,
        geo=self.geo_t_plus_dt,
        dt=0.1,
        calc_Phi_b_dot=False,
    )
    chex.assert_trees_all_close(state.Phi_b, self.geo_t_plus_dt.Phi_b)
    chex.assert_trees_all_close(state.Phi_b_dot, 0.0)

  def test_update_geometry_state_calc_true(self):
    dt = 0.1
    state = geometry_state.update_geometry_state(
        geometry_state=self.initial_state,
        geo=self.geo_t_plus_dt,
        dt=dt,
        calc_Phi_b_dot=True,
    )
    expected_phi_b_dot = (self.geo_t_plus_dt.Phi_b - self.geo_t.Phi_b) / dt
    chex.assert_trees_all_close(state.Phi_b, self.geo_t_plus_dt.Phi_b)
    chex.assert_trees_all_close(state.Phi_b_dot, expected_phi_b_dot)

  def test_to_output_dict(self):
    state1 = geometry_state.GeometryState(
        Phi_b=jnp.asarray(0.0), Phi_b_dot=jnp.asarray(1.5)
    )
    state2 = geometry_state.GeometryState(
        Phi_b=jnp.asarray(0.0), Phi_b_dot=jnp.asarray(2.5)
    )
    stacked_state = jax.tree.map(lambda *xs: jnp.stack(xs), state1, state2)

    times = np.array([0.0, 0.1])
    rho_face = np.linspace(0, 1, 5)
    rho_cell = 0.5 * (rho_face[:-1] + rho_face[1:])
    context = output_grid_context.OutputGridContext(
        times=times,
        rho_face_norm=rho_face,
        rho_cell_norm=rho_cell,
        rho_cell_plus_boundaries_norm=np.concatenate([[0.0], rho_cell, [1.0]]),
    )

    out = stacked_state.to_output_dict(context)
    self.assertIn(output_keys.PHI_B_DOT, out)
    dims, arr, units = out[output_keys.PHI_B_DOT]
    self.assertEqual(dims, (output_keys.TIME,))
    np.testing.assert_allclose(arr, [1.5, 2.5])
    self.assertEqual(units, output_keys.get_units(output_keys.PHI_B_DOT))


if __name__ == '__main__':
  absltest.main()

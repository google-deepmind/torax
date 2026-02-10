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

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from torax._src.geometry import circular_geometry
from torax._src.internal_boundary_conditions import adaptive_source
from torax._src.internal_boundary_conditions import internal_boundary_conditions
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name


class InternalBoundaryConditionsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('initial_time', 0.0, [1.0, 0.0, 0.0, 2.0]),
      ('intermediate_time', 0.5, [2.0, 0.0, 0.0, 3.0]),
      ('final_time', 1.0, [3.0, 0.0, 0.0, 4.0]),
      ('after_final_time', 1.5, [3.0, 0.0, 0.0, 4.0]),
  )
  def test_internal_boundary_conditions_config_build_runtime_params(
      self, t, expected_T_i
  ):
    ibc_config = internal_boundary_conditions.InternalBoundaryConditionsConfig(
        T_i={
            0.0: {0: 1.0, 1: 2.0},
            1.0: {0: 3.0, 1: 4.0},
        },
    )
    geo = circular_geometry.CircularConfig(n_rho=4).build_geometry()
    torax_pydantic.set_grid(ibc_config, geo.torax_mesh)

    runtime_params = ibc_config.build_runtime_params(t=t)
    np.testing.assert_array_equal(
        runtime_params.T_i,
        np.array(expected_T_i),
    )
    np.testing.assert_array_equal(
        runtime_params.T_e,
        np.array([0.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(
        runtime_params.n_e,
        np.array([0.0, 0.0, 0.0, 0.0]),
    )

  def test_update(self):
    ibc1 = internal_boundary_conditions.InternalBoundaryConditions(
        T_i=jnp.array([1.0, 0.0, 0.0]),
        T_e=jnp.array([0.0, 2.0, 0.0]),
        n_e=jnp.array([0.0, 0.0, 3.0]),
    )
    ibc2 = internal_boundary_conditions.InternalBoundaryConditions(
        T_i=jnp.array([1.1, 1.1, 0.0]),
        T_e=jnp.array([0.0, 0.0, 2.2]),
        n_e=jnp.array([3.3, 0.0, 0.0]),
    )

    updated_ibc = ibc1.update(ibc2)

    np.testing.assert_allclose(updated_ibc.T_i, jnp.array([1.1, 1.1, 0.0]))
    np.testing.assert_allclose(updated_ibc.T_e, jnp.array([0.0, 2.0, 2.2]))
    np.testing.assert_allclose(updated_ibc.n_e, jnp.array([3.3, 0.0, 3.0]))

  def test_apply_adaptive_source(self):
    nx = 4
    ibc = internal_boundary_conditions.InternalBoundaryConditions(
        T_i=jnp.array([10.0, 0.0, 0.0, 0.0]),
        T_e=jnp.array([0.0, 20.0, 0.0, 0.0]),
        n_e=jnp.array([0.0, 0.0, 30.0, 0.0]),
    )

    source_T_i = jnp.zeros(nx)
    source_T_e = jnp.zeros(nx)
    source_n_e = jnp.zeros(nx)
    source_mat_ii = jnp.zeros(nx)
    source_mat_ee = jnp.zeros(nx)
    source_mat_nn = jnp.zeros(nx)

    class MockNumerics:
      adaptive_T_source_prefactor = 2.0
      adaptive_n_source_prefactor = 3.0

    class MockRuntimeParams:
      numerics = MockNumerics()

    (
        source_T_i,
        source_T_e,
        source_n_e,
        source_mat_ii,
        source_mat_ee,
        source_mat_nn,
    ) = adaptive_source.apply_adaptive_source(
        source_T_i=source_T_i,
        source_T_e=source_T_e,
        source_n_e=source_n_e,
        source_mat_ii=source_mat_ii,
        source_mat_ee=source_mat_ee,
        source_mat_nn=source_mat_nn,
        runtime_params=MockRuntimeParams(),  # pytype: disable=wrong-arg-types
        internal_boundary_conditions=ibc,
    )

    np.testing.assert_allclose(source_T_i, jnp.array([20.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(source_mat_ii, jnp.array([-2.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(source_T_e, jnp.array([0.0, 40.0, 0.0, 0.0]))
    np.testing.assert_allclose(source_mat_ee, jnp.array([0.0, -2.0, 0.0, 0.0]))
    np.testing.assert_allclose(source_n_e, jnp.array([0.0, 0.0, 90.0, 0.0]))
    np.testing.assert_allclose(source_mat_nn, jnp.array([0.0, 0.0, -3.0, 0.0]))


if __name__ == '__main__':
  absltest.main()

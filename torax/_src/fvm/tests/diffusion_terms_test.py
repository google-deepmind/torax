# Copyright 2025 DeepMind Technologies Limited
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
from jax import numpy as jnp
import numpy as np
from torax._src.fvm import cell_variable
from torax._src.fvm import diffusion_terms


# TODO(b/469726859): Extend tests to cover non-uniform grid.
class DiffusionTermsTest(absltest.TestCase):

  def test_diffusion_terms_with_dirichlet_boundary_conditions_unit_space(self):
    cell_var = cell_variable.CellVariable(
        value=jnp.zeros(4),
        dr=jnp.array(1.0),
        right_face_grad_constraint=None,
        left_face_grad_constraint=None,
        left_face_constraint=jnp.array(-5.0),
        right_face_constraint=jnp.array(5.0),
    )
    mat, vec = diffusion_terms.make_diffusion_terms(
        d_face=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        var=cell_var,
    )
    np.testing.assert_allclose(mat, np.array([
        [-4, 2, 0, 0,],
        [2, -5, 3, 0,],
        [0, 3, -7, 4,],
        [0, 0, 4, -14,],
    ]))
    np.testing.assert_allclose(vec, np.array([-10.0, 0.0, 0.0, 50.0]))

  def test_diffusion_terms_with_neumann_boundary_conditions_unit_space(self):
    cell_var = cell_variable.CellVariable(
        value=jnp.zeros(4),
        dr=jnp.array(1.0),
        right_face_grad_constraint=jnp.array(10.0),
        left_face_grad_constraint=jnp.array(-10.0),
        left_face_constraint=None,
        right_face_constraint=None,
    )
    mat, vec = diffusion_terms.make_diffusion_terms(
        d_face=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        var=cell_var,
    )
    np.testing.assert_allclose(mat, np.array([
        [-2, 2, 0, 0,],
        [2, -5, 3, 0,],
        [0, 3, -7, 4,],
        [0, 0, 4, -4,],
    ]))
    np.testing.assert_allclose(vec, np.array([10.0, 0.0, 0.0, 50.0]))

  def test_diffusion_terms_with_dirichlet_boundary_conditions(self):
    cell_var = cell_variable.CellVariable(
        value=jnp.zeros(4),
        dr=jnp.array(0.2),
        right_face_grad_constraint=None,
        left_face_grad_constraint=None,
        left_face_constraint=jnp.array(-5.0),
        right_face_constraint=jnp.array(5.0),
    )
    mat, vec = diffusion_terms.make_diffusion_terms(
        d_face=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        var=cell_var,
    )
    np.testing.assert_allclose(mat, np.array([
        [-100, 50, 0, 0,],
        [50, -125, 75, 0,],
        [0, 75, -175, 100,],
        [0, 0, 100, -350,],
    ]))
    np.testing.assert_allclose(vec, np.array([-250.0, 0.0, 0.0, 1250.0]))

  def test_diffusion_terms_with_neumann_boundary_conditions(self):
    cell_var = cell_variable.CellVariable(
        value=jnp.zeros(4),
        dr=jnp.array(0.2),
        right_face_grad_constraint=jnp.array(10.0),
        left_face_grad_constraint=jnp.array(-10.0),
        left_face_constraint=None,
        right_face_constraint=None,
    )
    mat, vec = diffusion_terms.make_diffusion_terms(
        d_face=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        var=cell_var,
    )
    np.testing.assert_allclose(mat, np.array([
        [-50, 50, 0, 0,],
        [50, -125, 75, 0,],
        [0, 75, -175, 100,],
        [0, 0, 100, -100,],
    ]))
    np.testing.assert_allclose(vec, np.array([50.0, 0.0, 0.0, 250.0]))

if __name__ == "__main__":
  absltest.main()

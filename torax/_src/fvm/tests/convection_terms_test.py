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
from absl.testing import parameterized
from jax import numpy as jnp
import numpy as np
from torax._src.fvm import cell_variable
from torax._src.fvm import convection_terms


def _make_face_centers(dr: float, num_cells: int) -> np.ndarray:
  """Creates uniform face centers from dr and number of cells."""
  return np.linspace(0.0, num_cells * dr, num=num_cells + 1)


class ConvectionTermsTest(parameterized.TestCase):
  """Tests for make_convection_terms."""

  def test_zero_convection_returns_zero_mat_and_vec(self):
    """When v_face is zero everywhere, mat and vec should be zero."""
    cell_var = cell_variable.CellVariable(
        value=jnp.array([1.0, 2.0, 3.0, 4.0]),
        face_centers=_make_face_centers(1.0, 4),
        right_face_grad_constraint=None,
        left_face_grad_constraint=None,
        left_face_constraint=jnp.array(0.0),
        right_face_constraint=jnp.array(0.0),
    )
    mat, vec = convection_terms.make_convection_terms(
        v_face=jnp.zeros(5),
        d_face=jnp.ones(5),
        var=cell_var,
    )
    np.testing.assert_allclose(mat, np.zeros((4, 4)), atol=1e-12)
    np.testing.assert_allclose(vec, np.zeros(4), atol=1e-12)

  def test_output_shapes(self):
    """Matrix should be (n, n), vector should be (n,)."""
    n = 6
    cell_var = cell_variable.CellVariable(
        value=jnp.ones(n),
        face_centers=_make_face_centers(0.5, n),
        right_face_grad_constraint=None,
        left_face_grad_constraint=None,
        left_face_constraint=jnp.array(0.0),
        right_face_constraint=jnp.array(0.0),
    )
    mat, vec = convection_terms.make_convection_terms(
        v_face=jnp.ones(n + 1),
        d_face=jnp.ones(n + 1),
        var=cell_var,
    )
    self.assertEqual(mat.shape, (n, n))
    self.assertEqual(vec.shape, (n,))

  def test_matrix_is_tridiagonal(self):
    """Off-tridiagonal elements should be zero."""
    n = 5
    cell_var = cell_variable.CellVariable(
        value=jnp.ones(n),
        face_centers=_make_face_centers(1.0, n),
        right_face_grad_constraint=None,
        left_face_grad_constraint=None,
        left_face_constraint=jnp.array(1.0),
        right_face_constraint=jnp.array(1.0),
    )
    mat, _ = convection_terms.make_convection_terms(
        v_face=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        d_face=jnp.ones(n + 1),
        var=cell_var,
    )
    # Off-tridiagonal entries should be zero.
    for i in range(n):
      for j in range(n):
        if abs(i - j) > 1:
          self.assertAlmostEqual(
              float(mat[i, j]),
              0.0,
              places=12,
              msg=f'mat[{i},{j}] should be 0 (off-tridiagonal)',
          )

  def test_single_cell_raises_not_implemented(self):
    """A single-cell grid should raise NotImplementedError."""
    cell_var = cell_variable.CellVariable(
        value=jnp.array([1.0]),
        face_centers=_make_face_centers(1.0, 1),
        right_face_grad_constraint=None,
        left_face_grad_constraint=None,
        left_face_constraint=jnp.array(0.0),
        right_face_constraint=jnp.array(0.0),
    )
    with self.assertRaises(NotImplementedError):
      convection_terms.make_convection_terms(
          v_face=jnp.array([1.0, 1.0]),
          d_face=jnp.ones(2),
          var=cell_var,
      )

  @parameterized.named_parameters(
      ('ghost', 'ghost'),
      ('direct', 'direct'),
      ('semi_implicit', 'semi-implicit'),
  )
  def test_dirichlet_modes_run_without_error(self, dirichlet_mode):
    """All three dirichlet_mode options should execute without errors."""
    cell_var = cell_variable.CellVariable(
        value=jnp.zeros(4),
        face_centers=_make_face_centers(1.0, 4),
        right_face_grad_constraint=None,
        left_face_grad_constraint=None,
        left_face_constraint=jnp.array(1.0),
        right_face_constraint=jnp.array(2.0),
    )
    mat, vec = convection_terms.make_convection_terms(
        v_face=jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        d_face=jnp.ones(5),
        var=cell_var,
        dirichlet_mode=dirichlet_mode,
    )
    self.assertEqual(mat.shape, (4, 4))
    self.assertEqual(vec.shape, (4,))

  @parameterized.named_parameters(
      ('ghost', 'ghost'),
      ('semi_implicit', 'semi-implicit'),
  )
  def test_neumann_modes_run_without_error(self, neumann_mode):
    """Both neumann_mode options should execute without errors."""
    cell_var = cell_variable.CellVariable(
        value=jnp.zeros(4),
        face_centers=_make_face_centers(1.0, 4),
        right_face_grad_constraint=jnp.array(0.0),
        left_face_grad_constraint=jnp.array(0.0),
        left_face_constraint=None,
        right_face_constraint=None,
    )
    mat, vec = convection_terms.make_convection_terms(
        v_face=jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        d_face=jnp.ones(5),
        var=cell_var,
        neumann_mode=neumann_mode,
    )
    self.assertEqual(mat.shape, (4, 4))
    self.assertEqual(vec.shape, (4,))

  def test_invalid_dirichlet_mode_raises(self):
    """An invalid dirichlet_mode should raise ValueError."""
    cell_var = cell_variable.CellVariable(
        value=jnp.zeros(4),
        face_centers=_make_face_centers(1.0, 4),
        right_face_grad_constraint=None,
        left_face_grad_constraint=None,
        left_face_constraint=jnp.array(0.0),
        right_face_constraint=jnp.array(0.0),
    )
    with self.assertRaises(ValueError):
      convection_terms.make_convection_terms(
          v_face=jnp.ones(5),
          d_face=jnp.ones(5),
          var=cell_var,
          dirichlet_mode='invalid',
      )

  def test_invalid_neumann_mode_raises(self):
    """An invalid neumann_mode should raise ValueError."""
    cell_var = cell_variable.CellVariable(
        value=jnp.zeros(4),
        face_centers=_make_face_centers(1.0, 4),
        right_face_grad_constraint=jnp.array(0.0),
        left_face_grad_constraint=jnp.array(0.0),
        left_face_constraint=None,
        right_face_constraint=None,
    )
    with self.assertRaises(ValueError):
      convection_terms.make_convection_terms(
          v_face=jnp.ones(5),
          d_face=jnp.ones(5),
          var=cell_var,
          neumann_mode='invalid',
      )

  def test_dirichlet_ghost_boundary_contributes_to_vec(self):
    """Non-zero Dirichlet BCs should produce a non-zero vec."""
    cell_var = cell_variable.CellVariable(
        value=jnp.zeros(4),
        face_centers=_make_face_centers(1.0, 4),
        right_face_grad_constraint=None,
        left_face_grad_constraint=None,
        left_face_constraint=jnp.array(5.0),
        right_face_constraint=jnp.array(10.0),
    )
    _, vec = convection_terms.make_convection_terms(
        v_face=jnp.array([2.0, 2.0, 2.0, 2.0, 2.0]),
        d_face=jnp.ones(5),
        var=cell_var,
        dirichlet_mode='ghost',
    )
    # Boundary cells should have non-zero contributions from BCs.
    self.assertNotAlmostEqual(float(vec[0]), 0.0)
    self.assertNotAlmostEqual(float(vec[-1]), 0.0)
    # Interior cells should have zero vec (no source from BCs).
    np.testing.assert_allclose(vec[1:-1], 0.0, atol=1e-12)

  def test_uniform_convection_neumann_zero_grad(self):
    """Uniform v with zero-gradient Neumann BCs: vec boundary from grad=0."""
    cell_var = cell_variable.CellVariable(
        value=jnp.array([1.0, 2.0, 3.0, 4.0]),
        face_centers=_make_face_centers(1.0, 4),
        right_face_grad_constraint=jnp.array(0.0),
        left_face_grad_constraint=jnp.array(0.0),
        left_face_constraint=None,
        right_face_constraint=None,
    )
    _, vec = convection_terms.make_convection_terms(
        v_face=jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        d_face=jnp.ones(5),
        var=cell_var,
        neumann_mode='ghost',
    )
    # With zero gradient constraint, boundary vec contributions are zero.
    np.testing.assert_allclose(vec, 0.0, atol=1e-12)

  def test_negative_d_face_no_crash(self):
    """Negative diffusion should not crash (sign handled internally)."""
    cell_var = cell_variable.CellVariable(
        value=jnp.zeros(4),
        face_centers=_make_face_centers(1.0, 4),
        right_face_grad_constraint=None,
        left_face_grad_constraint=None,
        left_face_constraint=jnp.array(0.0),
        right_face_constraint=jnp.array(0.0),
    )
    mat, vec = convection_terms.make_convection_terms(
        v_face=jnp.ones(5),
        d_face=-jnp.ones(5),
        var=cell_var,
    )
    self.assertEqual(mat.shape, (4, 4))
    self.assertEqual(vec.shape, (4,))
    # Results should still be finite.
    self.assertTrue(np.all(np.isfinite(mat)))
    self.assertTrue(np.all(np.isfinite(vec)))

  def test_symmetry_with_symmetric_inputs(self):
    """Symmetric v_face and BCs produce finite, structured output."""
    cell_var = cell_variable.CellVariable(
        value=jnp.zeros(4),
        face_centers=_make_face_centers(1.0, 4),
        right_face_grad_constraint=None,
        left_face_grad_constraint=None,
        left_face_constraint=jnp.array(1.0),
        right_face_constraint=jnp.array(1.0),
    )
    # Symmetric v_face.
    v = jnp.array([1.0, 2.0, 3.0, 2.0, 1.0])
    mat, vec = convection_terms.make_convection_terms(
        v_face=v,
        d_face=jnp.ones(5),
        var=cell_var,
        dirichlet_mode='ghost',
    )
    self.assertTrue(np.all(np.isfinite(mat)))
    self.assertTrue(np.all(np.isfinite(vec)))
    # Boundary cells have non-zero vec from Dirichlet BCs.
    self.assertNotAlmostEqual(float(vec[0]), 0.0)
    self.assertNotAlmostEqual(float(vec[-1]), 0.0)
    # Interior cells have zero vec.
    np.testing.assert_allclose(vec[1:-1], 0.0, atol=1e-12)

  def test_large_peclet_convection_dominated(self):
    """Very large Péclet number (convection-dominated limit)."""
    cell_var = cell_variable.CellVariable(
        value=jnp.zeros(4),
        face_centers=_make_face_centers(1.0, 4),
        right_face_grad_constraint=None,
        left_face_grad_constraint=None,
        left_face_constraint=jnp.array(0.0),
        right_face_constraint=jnp.array(0.0),
    )
    # Large v relative to d -> large Péclet number.
    mat, vec = convection_terms.make_convection_terms(
        v_face=jnp.ones(5) * 1000.0,
        d_face=jnp.ones(5) * 0.001,
        var=cell_var,
    )
    self.assertTrue(np.all(np.isfinite(mat)))
    self.assertTrue(np.all(np.isfinite(vec)))

  def test_small_peclet_diffusion_dominated(self):
    """Very small Péclet number (diffusion-dominated limit)."""
    cell_var = cell_variable.CellVariable(
        value=jnp.zeros(4),
        face_centers=_make_face_centers(1.0, 4),
        right_face_grad_constraint=None,
        left_face_grad_constraint=None,
        left_face_constraint=jnp.array(0.0),
        right_face_constraint=jnp.array(0.0),
    )
    # Small v relative to d -> small Péclet number.
    mat, vec = convection_terms.make_convection_terms(
        v_face=jnp.ones(5) * 0.001,
        d_face=jnp.ones(5) * 1000.0,
        var=cell_var,
    )
    self.assertTrue(np.all(np.isfinite(mat)))
    self.assertTrue(np.all(np.isfinite(vec)))
    # In diffusion-dominated limit, convection contribution is small.
    np.testing.assert_allclose(mat, np.zeros((4, 4)), atol=0.1)


if __name__ == '__main__':
  absltest.main()

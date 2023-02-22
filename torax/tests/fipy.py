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

"""Unit tests for torax.fvm."""
from typing import Optional, Sequence

from absl.testing import absltest
from absl.testing import parameterized
import fipy
from jax import numpy as jnp
import numpy as np
from torax import config as config_lib
from torax import config_slice
from torax import fvm
from torax.fvm import implicit_solve_block
from torax.tests.test_lib import pint_ref


class FiPyTest(pint_ref.ReferenceValueTest):
  """Tests that torax.fvm matches FiPy."""

  @parameterized.parameters([
      dict(seed=202303151, left_grad=False, right_grad=False, dim=3),
      dict(seed=202303152, left_grad=False, right_grad=True, dim=4),
      dict(seed=202303153, left_grad=True, right_grad=False, dim=5),
      dict(seed=202303154, left_grad=True, right_grad=True, dim=6),
  ])
  def test_grad_and_face_grad(
      self, seed: int, left_grad: bool, right_grad: bool, dim: int
  ):
    """Test that CellVariable.face_grad matches a FiPy equivalent.

    Args:
      seed: Numpy RNG seed
      left_grad: if True, use a gradient constraint on leftmost face, else use a
        value constraint.
      right_grad: if True, use a gradient constraint on rightmost face, else use
        a value constraint.
      dim: Size ofthe CellVariable
    """

    # Define the problem
    rng = np.random.RandomState(seed)
    value = rng.randn(dim)
    eps = 1e-8
    dr = np.abs(rng.randn()) + eps
    left_face_constraint = rng.randn() if not left_grad else None
    left_face_grad_constraint = rng.randn() if left_grad else None
    right_face_constraint = rng.randn() if not right_grad else None
    right_face_grad_constraint = rng.randn() if right_grad else None

    # Torax solution
    convert = lambda x: None if x is None else jnp.array(x)
    cell_var_torax = fvm.CellVariable(
        value=jnp.array(value),
        dr=jnp.array(dr),
        left_face_constraint=convert(left_face_constraint),
        left_face_grad_constraint=convert(left_face_grad_constraint),
        right_face_constraint=convert(right_face_constraint),
        right_face_grad_constraint=convert(right_face_grad_constraint),
    )
    grad_torax = cell_var_torax.grad()
    face_grad_torax = cell_var_torax.face_grad()

    # FiPy solution
    mesh = fipy.Grid1D(
        nx=dim,
        dx=dr,
    )
    cell_var_fipy = fipy.CellVariable(
        mesh=mesh,
        value=value,
    )
    if left_grad:
      cell_var_fipy.faceGrad.constrain(
          left_face_grad_constraint, where=mesh.facesLeft
      )
    else:
      cell_var_fipy.constrain(left_face_constraint, where=mesh.facesLeft)
    if right_grad:
      cell_var_fipy.faceGrad.constrain(
          right_face_grad_constraint, where=mesh.facesRight
      )
    else:
      cell_var_fipy.constrain(right_face_constraint, where=mesh.facesRight)
    grad_fipy = np.squeeze(cell_var_fipy.grad())
    face_grad_fipy = np.squeeze(cell_var_fipy.faceGrad())

    # Check that the two solutions match
    np.testing.assert_allclose(grad_torax, grad_fipy)
    np.testing.assert_allclose(face_grad_torax, face_grad_fipy)

  @parameterized.parameters([
      dict(seed=202303155, left_grad=False, right_grad=False, dim=6),
      dict(seed=202303156, left_grad=False, right_grad=True, dim=5),
      dict(seed=202303157, left_grad=True, right_grad=False, dim=4),
      dict(seed=202303158, left_grad=True, right_grad=True, dim=3),
  ])
  def test_transient_diffusion(
      self,
      seed: int,
      left_grad: bool,
      right_grad: bool,
      dim: int,
  ):
    """Test that implicit method with a transient term and diffusion term matches a FiPy equivalent.

    Args:
      seed: Numpy RNG seed
      left_grad: if True, use a gradient constraint on leftmost face, else use a
        value constraint.
      right_grad: if True, use a gradient constraint on rightmost face, else use
        a value constraint.
      dim: Size ofthe CellVariable
    """

    # Define the problem
    rng = np.random.RandomState(seed)
    init_x = rng.randn(dim)
    eps = 1e-8
    dr = np.abs(rng.randn()) + eps
    dt = np.abs(rng.randn()) + eps
    tc_cell = np.abs(rng.randn(dim)) + eps
    d_face = np.abs(rng.randn(dim + 1)) + eps
    left_face_constraint = rng.randn() if not left_grad else None
    left_face_grad_constraint = rng.randn() if left_grad else None
    right_face_constraint = rng.randn() if not right_grad else None
    right_face_grad_constraint = rng.randn() if right_grad else None

    # Torax solution
    convert = lambda x: None if x is None else jnp.array(x)
    init_x_torax = fvm.CellVariable(
        value=jnp.array(init_x),
        dr=jnp.array(dr),
        left_face_constraint=convert(left_face_constraint),
        left_face_grad_constraint=convert(left_face_grad_constraint),
        right_face_constraint=convert(right_face_constraint),
        right_face_grad_constraint=convert(right_face_grad_constraint),
    )
    coeffs = fvm.Block1DCoeffs(
        transient_out_cell=(tc_cell,),
        transient_in_cell=(jnp.ones_like(tc_cell),),
        d_face=(jnp.array(d_face),),
    )
    config = config_lib.Config(nr=dim)
    dynamic_config_slice = config_slice.build_dynamic_config_slice(config)

    (final_x_torax,), _ = implicit_solve_block.implicit_solve_block(
        x_old=(init_x_torax,),
        # Use the original x as the initial "guess" for x_new. Used when
        # computing the coefficients for time t + dt.
        x_new_vec_guess=init_x_torax.value,
        x_new_update_fns=tuple([lambda cv: cv]),  # no-op
        dt=dt,
        coeffs_old=coeffs,
        # Assume no time-dependent params.
        coeffs_callback=lambda x, dcs, allow_pereverzev=False: coeffs,
        dynamic_config_slice_t_plus_dt=dynamic_config_slice,
        theta_imp=1.0,
    )
    # FiPy solution
    mesh = fipy.Grid1D(
        nx=dim,
        dx=dr,
    )
    assert init_x.ndim == 1
    x_fipy = fipy.CellVariable(
        mesh=mesh,
        value=init_x,
    )
    if left_grad:
      x_fipy.faceGrad.constrain(left_face_grad_constraint, where=mesh.facesLeft)
    else:
      x_fipy.constrain(left_face_constraint, where=mesh.facesLeft)
    if right_grad:
      x_fipy.faceGrad.constrain(
          right_face_grad_constraint, where=mesh.facesRight
      )
    else:
      x_fipy.constrain(right_face_constraint, where=mesh.facesRight)

    transient_coeff = fipy.CellVariable(mesh=mesh, value=tc_cell)
    transient = fipy.TransientTerm(coeff=transient_coeff, var=x_fipy)
    diffusion_coeff = fipy.FaceVariable(mesh=mesh, value=d_face)
    diffusion = fipy.DiffusionTerm(coeff=diffusion_coeff, var=x_fipy)
    eq = transient == diffusion
    eq.solve(dt=dt)

    # Check that the two solutions match
    np.testing.assert_allclose(final_x_torax.value, x_fipy.value)

  @parameterized.parameters([
      dict(seed=202303161, left_grad=False, right_grad=False, dim=4),
      dict(seed=202303162, left_grad=False, right_grad=True, dim=3),
      dict(seed=202303163, left_grad=True, right_grad=False, dim=6),
      dict(seed=202303164, left_grad=True, right_grad=True, dim=5),
      dict(
          seed=202303165,
          left_grad=True,
          right_grad=True,
          dim=2,
      ),
      # Use d_face_mask to make sure we handle the zero diffusion corner
      # case the same as FiPy. Zero diffusion requires some mild numerical
      # hacks to avoid divide by zero.
      dict(
          seed=202304071,
          left_grad=False,
          right_grad=False,
          dim=4,
          d_face_mask=[1.0, 0.0, 0.0, 0.0, 1.0],
      ),
      dict(
          seed=202304072,
          left_grad=False,
          right_grad=True,
          dim=3,
          d_face_mask=[0.0, 1.0, 1.0, 0.0],
      ),
      dict(
          seed=202304073,
          left_grad=True,
          right_grad=False,
          dim=6,
          d_face_mask=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      ),
      dict(
          seed=202304074,
          left_grad=True,
          right_grad=True,
          dim=5,
          d_face_mask=[1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
      ),
  ])
  def test_transient_diffusion_convection(
      self,
      seed: int,
      left_grad: bool,
      right_grad: bool,
      dim: int,
      d_face_mask: Optional[Sequence[float]] = None,
  ):
    """Test that implicit method with transient, diffusion and convection terms matches a FiPy equivalent.

    Args:
      seed: Numpy RNG seed
      left_grad: if True, use a gradient constraint on leftmost face, else use a
        value constraint.
      right_grad: if True, use a gradient constraint on rightmost face, else use
        a value constraint.
      dim: Size ofthe CellVariable
      d_face_mask: Mask applied to `d_face` to test with zero diffusion
    """

    # Define the problem
    rng = np.random.RandomState(seed)
    init_x = rng.randn(dim)
    eps = 1e-8
    dr = np.abs(rng.randn()) + eps
    dt = np.abs(rng.randn()) + eps
    tc_cell = np.abs(rng.randn(dim)) + eps
    d_face = np.abs(rng.randn(dim + 1))
    if d_face_mask is not None:
      d_face = d_face * np.array(d_face_mask)
    v_face = rng.randn(dim + 1)
    left_face_constraint = rng.randn() if not left_grad else None
    left_face_grad_constraint = rng.randn() if left_grad else None
    right_face_constraint = rng.randn() if not right_grad else None
    right_face_grad_constraint = rng.randn() if right_grad else None

    # Torax solution
    convert = lambda x: None if x is None else jnp.array(x)
    init_x_torax = fvm.CellVariable(
        value=jnp.array(init_x),
        dr=jnp.array(dr),
        left_face_constraint=convert(left_face_constraint),
        left_face_grad_constraint=convert(left_face_grad_constraint),
        right_face_constraint=convert(right_face_constraint),
        right_face_grad_constraint=convert(right_face_grad_constraint),
    )
    coeffs = fvm.Block1DCoeffs(
        transient_out_cell=(tc_cell,),
        transient_in_cell=(jnp.ones_like(tc_cell),),
        d_face=(jnp.array(d_face),),
        v_face=(jnp.array(v_face),),
    )
    config = config_lib.Config(nr=dim)
    dynamic_config_slice = config_slice.build_dynamic_config_slice(config)
    (final_x_torax,), _ = implicit_solve_block.implicit_solve_block(
        x_old=(init_x_torax,),
        # Use the original x as the initial "guess" for x_new. Used when
        # computing the coefficients for time t + dt.
        x_new_vec_guess=init_x_torax.value,
        x_new_update_fns=tuple([lambda cv: cv]),  # no-op
        dt=dt,
        coeffs_old=coeffs,
        # Assume no time-dependent params.
        coeffs_callback=lambda x, dcs, allow_pereverzev=False: coeffs,
        dynamic_config_slice_t_plus_dt=dynamic_config_slice,
        theta_imp=1.0,
        # Use FiPy's approach to convection boundary conditions
        convection_dirichlet_mode="semi-implicit",
        convection_neumann_mode="semi-implicit",
    )
    # FiPy solution
    mesh = fipy.Grid1D(
        nx=dim,
        dx=dr,
    )
    assert init_x.ndim == 1
    x_fipy = fipy.CellVariable(
        mesh=mesh,
        value=init_x,
    )
    if left_grad:
      x_fipy.faceGrad.constrain(left_face_grad_constraint, where=mesh.facesLeft)
    else:
      x_fipy.constrain(left_face_constraint, where=mesh.facesLeft)
    if right_grad:
      x_fipy.faceGrad.constrain(
          right_face_grad_constraint, where=mesh.facesRight
      )
    else:
      x_fipy.constrain(right_face_constraint, where=mesh.facesRight)

    transient_coeff = fipy.CellVariable(mesh=mesh, value=tc_cell)
    transient = fipy.TransientTerm(coeff=transient_coeff, var=x_fipy)
    diffusion_coeff = fipy.FaceVariable(mesh=mesh, value=d_face)
    diffusion = fipy.DiffusionTerm(coeff=diffusion_coeff, var=x_fipy)
    convection_coeff = fipy.FaceVariable(
        mesh=mesh, value=np.expand_dims(v_face, 0)
    )
    convection = fipy.ConvectionTerm(coeff=convection_coeff, var=x_fipy)
    eq = transient + convection == diffusion
    eq.solve(dt=dt)

    # Check that the two solutions match
    np.testing.assert_allclose(final_x_torax.value, x_fipy.value)


if __name__ == "__main__":
  absltest.main()

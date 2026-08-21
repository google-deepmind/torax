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

"""The CellVariable class.

A dataclass used to represent variables on meshes for the 1D fvm solver.
Naming conventions and API are similar to those developed in the FiPy fvm solver
[https://www.ctcms.nist.gov/fipy/]
"""
import dataclasses
import functools

import jax
from jax import numpy as jnp
import jaxtyping as jt
from torax._src import array_typing
from torax._src import jax_utils
from torax._src import math_utils


def _zero() -> array_typing.FloatScalar:
  """Returns a scalar zero as a jax Array."""
  return jnp.zeros(())


@jax.jit
def _compute_inner_grad(
    value: array_typing.FloatVectorCell,
    value_right_face: array_typing.FloatScalar,
    face_centers: array_typing.FloatVectorFace,
    cell_centers: array_typing.FloatVectorCell,
) -> jt.Float[array_typing.Array, 'rhon-1']:
  """Computes the gradient on inner faces.

  This gradient is computed using a 3-point stencil and is accurate to second
  order differences. The gradient on the penultimate rightmost face is
  determined by the value on the right face.

  Note: This gradient can not be compared against FiPy as that is not correct
  to second order differences when the grid is not uniform.

  Args:
    value: The value on the cell grid.
    value_right_face: The value on the right face, used to determine the
      gradient on the second-to-last face.
    face_centers: The locations of the face centers.
    cell_centers: The locations of the cell centers.

  """
  @jax.jit
  def gradient(
      d: tuple[
          array_typing.Array,
          array_typing.Array,
          array_typing.Array,
      ],
      v: tuple[
          array_typing.Array,
          array_typing.Array,
          array_typing.Array,
      ],
  ) -> array_typing.Array:
    d1, d2, d3 = d
    v1, v2, v3 = v
    c1 = (d2 + d3) / (-1 * d1**2 + d1 * d2 + d1 * d3 - d2 * d3)
    # c2 = (-d1 - d3) / (-1*d1*d2 + d1*d3 + d2**2 - d2*d3)
    c3 = (-d1 - d2) / (d1 * d2 - d1 * d3 - d2 * d3 + d3**2)
    # We use c1*(v1-v2) + c3*(v3-v2) instead of c1*v1 + c2*v2 + c3*v3
    # because c1+c2+c3 = 0 analytically, but not numerically.
    # By using differences we ensure that if v1=v2=v3, the result is exactly 0.
    return c1 * (v1 - v2) + c3 * (v3 - v2)

  d_left = cell_centers[:-2] - face_centers[1:-2]
  d_right = cell_centers[1:-1] - face_centers[1:-2]
  d_right_right = cell_centers[2:] - face_centers[1:-2]
  left = value[:-2]
  right = value[1:-1]
  right_right = value[2:]
  inner_grads = gradient(
      (d_left, d_right, d_right_right), (left, right, right_right)
  )

  penultimate_left = value[-2]
  penultimate_right = value[-1]
  penultimate_right_right = value_right_face
  d_left = cell_centers[-2] - face_centers[-2]
  d_right = cell_centers[-1] - face_centers[-2]
  d_right_right = face_centers[-1] - face_centers[-2]
  penultimate_grad = gradient(
      (d_left, d_right, d_right_right),
      (penultimate_left, penultimate_right, penultimate_right_right),
  )
  return jnp.concat([inner_grads, jnp.atleast_1d(penultimate_grad)], axis=-1)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class CellVariable:
  """A variable representing values of the cells along the radius.

  This class may be used as a pytree argument to jitted Jax functions.
  Its hash and comparison functions should not be used.

  Attributes:
    value: Value of this variable at each cell grid point.
    face_centers: Locations of the face centers. This array should have length
      len(value) + 1. Supports both uniform and non-uniform grids.
    left_face_constraint: An optional scalar specifying the value of the
      leftmost face. Defaults to None, signifying no constraint. The user can
      modify this field at any time, but when face_grad is called exactly one of
      left_face_constraint and left_face_grad_constraint must be None.
    left_face_grad_constraint: An optional scalar specifying the (otherwise
      underdetermined) value of gradient on the leftmost face. See
      left_face_constraint.
    right_face_constraint: Analogous to left_face_constraint but for the right
      face, see left_face_constraint.
    right_face_grad_constraint: Analogous to left_face_grad_constraint but for
      the right face, see left_face_grad_constraint.
  """
  value: jt.Float[array_typing.Array, 'cell']
  face_centers: jt.Float[array_typing.Array, 'cell+1']
  left_face_constraint: array_typing.FloatScalar | None = None
  right_face_constraint: array_typing.FloatScalar | None = None
  left_face_grad_constraint: array_typing.FloatScalar | None = (
      dataclasses.field(default_factory=_zero)  # pyrefly: ignore[bad-assignment]
  )
  right_face_grad_constraint: array_typing.FloatScalar | None = (
      dataclasses.field(default_factory=_zero)  # pyrefly: ignore[bad-assignment]
  )
  # Can't make the above default values be jax zeros because that would be a
  # call to jax before absl.app.run

  @functools.cached_property
  def cell_centers(self) -> jt.Float[array_typing.Array, 'cell']:
    """Locations of the cell centers."""
    return (self.face_centers[..., 1:] + self.face_centers[..., :-1]) / 2.0

  @property
  def cell_widths(self) -> jt.Float[array_typing.Array, 'cell']:
    """Size of each cell."""
    return jnp.diff(self.face_centers)

  @property
  def cell_spacings(self) -> jt.Float[array_typing.Array, 'cell-1']:
    """Spacing between each cell."""
    return jnp.diff(self.cell_centers)

  def __post_init__(self):
    """Check that the CellVariable is valid."""
    for field in dataclasses.fields(self):
      value = getattr(self, field.name)
      name = field.name
      if isinstance(value, jax.Array):
        jax_dtype = jax_utils.get_dtype()
        if value.dtype != jax_dtype:
          raise TypeError(
              f'Expected dtype {jax_dtype}, got {value.dtype} for `{name}`'
          )
    left_constraints = (
        self.left_face_constraint, self.left_face_grad_constraint
    )
    if sum(constraint is not None for constraint in left_constraints) != 1:
      raise ValueError(
          'Exactly one of left_face_constraint and '
          'left_face_grad_constraint must be set.'
      )
    right_constraints = (
        self.right_face_constraint, self.right_face_grad_constraint,
    )
    if sum(constraint is not None for constraint in right_constraints) != 1:
      raise ValueError(
          'Exactly one of right_face_constraint and '
          'right_face_grad_constraint must be set.'
      )

  def face_grad(
      self,
      *,
      x: jt.Float[array_typing.Array, 'cell'] | None = None,
      x_left: array_typing.FloatScalar | None = None,
      x_right: array_typing.FloatScalar | None = None,
  ) -> jt.Float[array_typing.Array, 'face']:
    """Returns the gradient of this value with respect to the faces.

    Implemented using linear interpolation of 3-points accurate to second order
    differences. Leftmost and rightmost gradient entries are determined by user
    specified constraints.

    In the special case of a uniformly spaced grid, this gradient is identical
    to the gradient computed by forward differencing.

    Args:
      x: (optional) coordinates over which differentiation is carried out. If
        specified we assume this is a function of the cell grid.
      x_left: (optional) value of `x` at the leftmost face. Must be specified if
        `x` is specified.
      x_right: (optional) value of `x` at the rightmost face. Must be specified
        if `x` is specified.

    Returns:
      A jax.Array of shape (num_faces,) containing the gradient.
    """
    inner_grad = _compute_inner_grad(
        self.value,
        self.right_face_value,
        self.face_centers,
        self.cell_centers,
    )
    # If we are differentiating w.r.t another variable x, we apply the chain
    # rule and effectively weight the computed gradient by dx_dcell.
    if x is not None:
      if x_left is None or x_right is None:
        raise ValueError('Must specify both x_left and x_right with x.')
      dx_dcell = _compute_inner_grad(
          x, jnp.atleast_1d(x_right), self.face_centers, self.cell_centers
      )
      # dval_dx = dval_dcell / dx_dcell
      inner_grad = inner_grad / dx_dcell
      d_left = x[0] - x_left
      d_right = x_right - x[-1]
    else:
      d_left = self.cell_widths[0] / 2.0
      d_right = self.cell_widths[-1] / 2.0

    if self.left_face_constraint is not None:
      left_grad = (self.value[0] - self.left_face_constraint) / d_left
    else:
      assert self.left_face_grad_constraint is not None
      left_grad = self.left_face_grad_constraint

    if self.right_face_constraint is not None:
      right_grad = (self.right_face_constraint - self.value[-1]) / d_right
    else:
      assert self.right_face_grad_constraint is not None
      right_grad = self.right_face_grad_constraint

    left = jnp.expand_dims(left_grad, axis=0)
    right = jnp.expand_dims(right_grad, axis=0)
    return jnp.concatenate([left, inner_grad, right])

  @functools.cached_property
  def left_face_value(self) -> jt.Float[array_typing.Array, '']:
    """Calculates the value of the leftmost face."""
    if self.left_face_constraint is not None:
      value = self.left_face_constraint
      # Boundary value has one fewer dim than cell value, expand to concat with.
      value = jnp.expand_dims(value, axis=-1)
    else:
      # When there is no constraint, leftmost face equals
      # leftmost cell
      value = self.value[..., 0:1]
    return value

  @functools.cached_property
  def right_face_value(self) -> jt.Float[array_typing.Array, '']:
    """Calculates the value of the rightmost face."""
    if self.right_face_constraint is not None:
      value = self.right_face_constraint
      # Boundary value has one fewer dim than cell value, expand to concat with.
      value = jnp.expand_dims(value, axis=-1)
    else:
      # Maintain right_face consistent with right_face_grad_constraint
      assert self.right_face_grad_constraint is not None
      dr = self.cell_widths[-1]
      value = (
          self.value[..., -1:]
          + jnp.expand_dims(self.right_face_grad_constraint, axis=-1)
          * jnp.expand_dims(dr, axis=-1)
          / 2
      )
    return value

  def face_value(self) -> jt.Float[array_typing.Array, 'face']:
    """Calculates values of this variable on the face grid."""
    inner = math_utils.inner_face_values_from_cell_values(
        cell_values=self.value,
        face_centers=self.face_centers,
        cell_centers=self.cell_centers,
    )

    return jnp.concatenate(
        [self.left_face_value, inner, self.right_face_value], axis=-1
    )

  def grad(self) -> jt.Float[array_typing.Array, 'cell']:
    """Returns the gradient of this variable wrt cell centers."""
    face = self.face_value()
    return jnp.diff(face) / jnp.diff(self.face_centers)

  def __str__(self) -> str:
    output_string = f'CellVariable(value={self.value}'
    if self.left_face_constraint is not None:
      output_string += f', left_face_constraint={self.left_face_constraint}'
    if self.right_face_constraint is not None:
      output_string += f', right_face_constraint={self.right_face_constraint}'
    if self.left_face_grad_constraint is not None:
      output_string += (
          f', left_face_grad_constraint={self.left_face_grad_constraint}'
      )
    if self.right_face_grad_constraint is not None:
      output_string += (
          f', right_face_grad_constraint={self.right_face_grad_constraint}'
      )
    output_string += ')'
    return output_string

  def cell_plus_boundaries(self) -> jt.Float[array_typing.Array, 'cell+2']:
    """Returns the value of this variable plus left and right boundaries."""
    right_value = self.right_face_value
    left_value = self.left_face_value
    return jnp.concatenate(
        [left_value, self.value, right_value],
        axis=-1,
    )

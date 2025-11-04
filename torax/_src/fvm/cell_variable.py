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

A jax_utils.jax_dataclass used to represent variables on meshes for the 1D fvm
solver.
Naming conventions and API are similar to those developed in the FiPy fvm solver
[https://www.ctcms.nist.gov/fipy/]
"""
import dataclasses

import chex
import jax
from jax import numpy as jnp
import jaxtyping as jt
from torax._src import array_typing


def _zero() -> array_typing.FloatScalar:
  """Returns a scalar zero as a jax Array."""
  return jnp.zeros(())


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class CellVariable:
  """A variable representing values of the cells along the radius.

  This class may be used as a pytree argument to jitted Jax functions.
  Its hash and comparison functions should not be used.

  Attributes:
    value: A jax.Array containing the value of this variable at each cell.
    dr: Distance between cell centers.
    left_face_constraint: An optional jax scalar specifying the value of the
      leftmost face. Defaults to None, signifying no constraint. The user can
      modify this field at any time, but when face_grad is called exactly one of
      left_face_constraint and left_face_grad_constraint must be None.
    left_face_grad_constraint: An optional jax scalar specifying the (otherwise
      underdetermined) value of the leftmost face. See left_face_constraint.
    right_face_constraint: Analogous to left_face_constraint but for the right
      face, see left_face_constraint.
    right_face_grad_constraint: A jax scalar specifying the undetermined value
      of the gradient on the rightmost face variable.
  """

  # t* means match 0 or more leading time dimensions.
  value: jt.Float[chex.Array, 't* cell']
  dr: jt.Float[chex.Array, 't*']
  left_face_constraint: jt.Float[chex.Array, 't*'] | None = None
  right_face_constraint: jt.Float[chex.Array, 't*'] | None = None
  left_face_grad_constraint: jt.Float[chex.Array, 't*'] | None = (
      dataclasses.field(default_factory=_zero)
  )
  right_face_grad_constraint: jt.Float[chex.Array, 't*'] | None = (
      dataclasses.field(default_factory=_zero)
  )
  # Can't make the above default values be jax zeros because that would be a
  # call to jax before absl.app.run

  def __post_init__(self):
    """Check that the CellVariable is valid.

    How is `sanity_check` different from `__post_init__`?
    - `sanity_check` is exposed to the client directly, so the client can
    explicitly check sanity without violating privacy conventions. This is
    useful for checking objects that were created e.g. using jax tree
    transformations.
    - `sanity_check` is guaranteed not to change the object, while
    `__post_init__` could in principle make changes.
    """
    # Automatically check dtypes of all numeric fields

    for field in dataclasses.fields(self):
      value = getattr(self, field.name)
      name = field.name
      if isinstance(value, jax.Array):
        if value.dtype != jnp.float64 and jax.config.read('jax_enable_x64'):
          raise TypeError(
              f'Expected dtype float64, got dtype {value.dtype} for `{name}`'
          )
        if value.dtype != jnp.float32 and not jax.config.read('jax_enable_x64'):
          raise TypeError(
              f'Expected dtype float32, got dtype {value.dtype} for `{name}`'
          )
    left_and = (
        self.left_face_constraint is not None
        and self.left_face_grad_constraint is not None
    )
    left_or = (
        self.left_face_constraint is not None
        or self.left_face_grad_constraint is not None
    )
    if left_and or not left_or:
      raise ValueError(
          'Exactly one of left_face_constraint and '
          'left_face_grad_constraint must be set.'
      )
    right_and = (
        self.right_face_constraint is not None
        and self.right_face_grad_constraint is not None
    )
    right_or = (
        self.right_face_constraint is not None
        or self.right_face_grad_constraint is not None
    )
    if right_and or not right_or:
      raise ValueError(
          'Exactly one of right_face_constraint and '
          'right_face_grad_constraint must be set.'
      )

  def _assert_unbatched(self):
    if len(self.value.shape) != 1:
      raise AssertionError(
          'CellVariable must be unbatched, but has `value` shape '
          f'{self.value.shape}. Consider using vmap to batch the function call.'
      )
    if self.dr.shape:
      raise AssertionError(
          'CellVariable must be unbatched, but has `dr` shape '
          f'{self.dr.shape}. Consider using vmap to batch the function call.'
      )

  def face_grad(
      self, x: jt.Float[chex.Array, 'cell'] | None = None
  ) -> jt.Float[chex.Array, 'face']:
    """Returns the gradient of this value with respect to the faces.

    Implemented using forward differencing of cells. Leftmost and rightmost
    gradient entries are determined by user specify constraints, see
    CellVariable class docstring.

    Args:
      x: (optional) coordinates over which differentiation is carried out

    Returns:
      A jax.Array of shape (num_faces,) containing the gradient.
    """
    self._assert_unbatched()
    if x is None:
      forward_difference = jnp.diff(self.value) / self.dr
    else:
      forward_difference = jnp.diff(self.value) / jnp.diff(x)

    def constrained_grad(
        face: jax.Array | None,
        grad: jax.Array | None,
        cell: jax.Array,
        right: bool,
    ) -> jax.Array:
      """Calculates the constrained gradient entry for an outer face."""

      if face is not None:
        if grad is not None:
          raise ValueError(
              'Cannot constraint both the value and gradient of '
              'a face variable.'
          )
        if x is None:
          dx = self.dr
        else:
          dx = x[-1] - x[-2] if right else x[1] - x[0]
        sign = -1 if right else 1
        return sign * (cell - face) / (0.5 * dx)
      else:
        if grad is None:
          raise ValueError('Must specify one of value or gradient.')
        return grad

    left_grad = constrained_grad(
        self.left_face_constraint,
        self.left_face_grad_constraint,
        self.value[0],
        right=False,
    )
    right_grad = constrained_grad(
        self.right_face_constraint,
        self.right_face_grad_constraint,
        self.value[-1],
        right=True,
    )

    left = jnp.expand_dims(left_grad, axis=0)
    right = jnp.expand_dims(right_grad, axis=0)
    return jnp.concatenate([left, forward_difference, right])

  def _left_face_value(self) -> jt.Float[chex.Array, '#t']:
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

  def _right_face_value(self) -> jt.Float[chex.Array, '#t']:
    """Calculates the value of the rightmost face."""
    if self.right_face_constraint is not None:
      value = self.right_face_constraint
      # Boundary value has one fewer dim than cell value, expand to concat with.
      value = jnp.expand_dims(value, axis=-1)
    else:
      # Maintain right_face consistent with right_face_grad_constraint
      value = (
          self.value[..., -1:]
          + jnp.expand_dims(self.right_face_grad_constraint, axis=-1)
          * jnp.expand_dims(self.dr, axis=-1)
          / 2
      )
    return value

  def face_value(self) -> jt.Float[jax.Array, 't* face']:
    """Calculates values of this variable on the face grid."""
    inner = (self.value[..., :-1] + self.value[..., 1:]) / 2.0
    return jnp.concatenate(
        [self._left_face_value(), inner, self._right_face_value()], axis=-1
    )

  def grad(self) -> jt.Float[jax.Array, 't* face']:
    """Returns the gradient of this variable wrt cell centers."""
    face = self.face_value()
    return jnp.diff(face) / jnp.expand_dims(self.dr, axis=-1)

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

  def cell_plus_boundaries(self) -> jt.Float[jax.Array, 't* cell+2']:
    """Returns the value of this variable plus left and right boundaries."""
    right_value = self._right_face_value()
    left_value = self._left_face_value()
    return jnp.concatenate(
        [left_value, self.value, right_value],
        axis=-1,
    )

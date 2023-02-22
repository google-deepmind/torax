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

"""Optimization-related functionality.

Functionality for differentiation throughout time, constrained optimization,
etc.
"""
import chex
import jax
from jax import numpy as jnp
from jax import tree_util
from torax import fvm
from torax import jax_utils
from torax import state as state_module


def interp(augmented_tree: ..., coords: jax.Array, desired_coord: jax.Array):
  """Interpolates variables in a tree to get their values at `desired_coord`.

  This is useful for gradient-based optimization of functions of the output of
  `scan`, if `scan` reaches the desired coordinate after a variable number
  of steps.

  Uses `stop_gradient` on interpolation weights, so the gradient flows through
  the 1-2 rows of `augmented_tree` used to make the output, but does not flow
  the other arguments.

  This could potentially be made faster by indexing just the 1-2 relevant rows
  of the tree before multiplying by the weights.

  Args:
    augmented_tree: pytree where each leaf has leaf.shape[0] == coords.size.
      Each `leaf` represents a corresponding `var`, with `leaf[i]` containing
      `var` evaluated at `coords[i]`. For example, `augmented_tree` might be a
      history of the state of a system, with `coords` being the time coordinates
      of each history entry, and `leaf[i]` being the value of `var` evaluated at
      time `t = coords[i]`.
    coords: 1-D array of the coordinates representing in `augmented_tree`.
      coords must be strictly increasing.
    desired_coord: Scalar coordinate that we should interpolate to.
      desired_coord must fall within the range represented by `coords`.

  Returns:
    interpolated_tree: pytree where each `interpolated_leaf` has shape
      `leaf.shape[1:]` for the corresponding leaf in `augmented_tree`,
      representing `var` evaluated at `desired_coord`.
  """

  # Validate input
  chex.assert_rank(coords, 1)
  chex.assert_rank(desired_coord, 0)

  def assert_shape(x):
    chex.assert_axis_dimension(x, axis=0, expected=coords.size)
    return x

  tree_util.tree_map(assert_shape, augmented_tree)

  min_coord = coords.min()
  max_coord = coords.max()
  msg = (
      "`desired_coord` must lie in the interval `[coords.min(), coords.max()]`."
  )
  cond = jnp.logical_and(desired_coord >= min_coord, desired_coord <= max_coord)
  coords = jax_utils.error_if(coords, jnp.logical_not(cond), msg)

  diff = jnp.diff(coords)
  min_diff = diff.min()
  coords = jax_utils.error_if_not_positive(
      min_diff, "opt.interp:min_diff", coords
  )

  before_mask = coords <= desired_coord
  after_mask = coords >= desired_coord
  last_before_mask = coords == (before_mask * coords).max()
  first_after_mask = (
      coords == (after_mask * coords + (1 - after_mask) * (2 * max_coord)).min()
  )
  last_before_mask = jax_utils.error_if(
      last_before_mask,
      last_before_mask.sum() != 1,
      "last_before_mask should have exactly one entry.",
  )
  first_after_mask = jax_utils.error_if(
      first_after_mask,
      first_after_mask.sum() != 1,
      "first_after_mask should have exactly one entry.",
  )
  coord_before = jnp.dot(last_before_mask, coords)
  coord_after = jnp.dot(first_after_mask, coords)
  exact_match = coord_before == coord_after
  # In the case of an exact match, the general formula for `alpha_after`
  # gets a divide by zero.
  # In the exact match case, any value in [0, 1] works for `alpha_after`, so
  # we simply add the `exact_match` bool to the denominator to avoid a divide
  # by zero, and arbitrarily set `alpha_after` to 0.
  numer = desired_coord - coord_before
  denom = (coord_after - coord_before) + exact_match
  alpha_after = numer / denom
  weights = (
      1.0 - alpha_after
  ) * last_before_mask + alpha_after * first_after_mask
  weights = jax.lax.stop_gradient(weights)
  interpolated_coord = jnp.dot(weights, coords)
  weights = jax_utils.error_if(
      weights,
      jnp.logical_not(jnp.allclose(interpolated_coord, desired_coord)),
      "interpolation failed",
  )

  if isinstance(augmented_tree, (fvm.CellVariable, state_module.State)):
    # Need to do this to keep track of whether the CellVariables are in
    # "history" mode.
    interpolated_tree = augmented_tree.project(weights)
  else:
    interp_leaf = lambda x: jnp.dot(weights, x)
    interpolated_tree = jax.tree_util.tree_map(interp_leaf, augmented_tree)
  return interpolated_tree

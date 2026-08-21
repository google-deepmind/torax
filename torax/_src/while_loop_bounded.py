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
"""Implementation of a differentiable jax.lax.while_loop with a maximum number of steps."""

from typing import Any, Callable, ParamSpec, TypeAlias, TypeVar
import chex
import jax
from jax import numpy as jnp
from jax.experimental import hijax

T = TypeVar('T')
BooleanNumeric: TypeAlias = Any  # A bool, or a Boolean array.
_State = ParamSpec('_State')
PyTree: TypeAlias = Any

_WHILE_LOOP_COUNT_DTYPE = jnp.int32


def while_loop_bounded(
    cond_fun: Callable[[_State], BooleanNumeric],  # pyrefly: ignore[invalid-annotation]
    body_fun: Callable[[_State], _State],  # pyrefly: ignore[invalid-annotation]
    init_val: _State,  # pyrefly: ignore[invalid-annotation]
    max_steps: int,
) -> tuple[_State, chex.Numeric, _State]:  # pyrefly: ignore[invalid-annotation]
  """A bounded reverse-mode differentiable while_loop."""

  cond_aux, converted_cond_fun = (
      jax.jit(cond_fun).trace(init_val).closure_convert()
  )
  body_aux, converted_body_fun = (
      jax.jit(body_fun).trace(init_val).closure_convert()
  )

  init_val_flat, init_val_tree = jax.tree.flatten(init_val)
  cond_aux_flat, cond_aux_tree = jax.tree.flatten(cond_aux)
  body_aux_flat, body_aux_tree = jax.tree.flatten(body_aux)

  final_state_flat, final_step_idx, history_final_flat = (
      WhileLoopBoundedWhileLoop(
          cond_fun=converted_cond_fun,
          body_fun=converted_body_fun,
          init_val_avals=tuple(jax.typeof(x) for x in init_val_flat),
          max_steps=max_steps,
          cond_aux_avals=tuple(jax.typeof(x) for x in cond_aux_flat),
          body_aux_avals=tuple(jax.typeof(x) for x in body_aux_flat),
          init_val_tree=init_val_tree,
          cond_aux_tree=cond_aux_tree,
          body_aux_tree=body_aux_tree,
      )(init_val_flat, cond_aux_flat, body_aux_flat)
  )

  final_state = jax.tree.unflatten(init_val_tree, final_state_flat)
  history_final = jax.tree.unflatten(init_val_tree, history_final_flat)
  return final_state, final_step_idx, history_final


def _add_axis(x: jax.core.ShapedArray, size: int) -> jax.core.ShapedArray:
  return jax.core.ShapedArray(shape=(size,) + x.shape, dtype=x.dtype)


def _instantiate_zeros(g: PyTree) -> PyTree:
  return jax.tree.map(
      hijax.instantiate_zeros, g, is_leaf=lambda x: isinstance(x, hijax.Zero)
  )


class WhileLoopBoundedWhileLoop(hijax.VJPHiPrimitive):
  """A bounded differentiable while_loop using jax.lax.while_loop."""

  def __init__(
      self,
      cond_fun: Callable[..., BooleanNumeric],  # pyrefly: ignore[invalid-annotation]
      body_fun: Callable[..., _State],  # pyrefly: ignore[invalid-annotation]
      init_val_avals: tuple[jax.core.ShapedArray, ...],
      max_steps: int,
      cond_aux_avals: tuple[jax.core.ShapedArray, ...],
      body_aux_avals: tuple[jax.core.ShapedArray, ...],
      init_val_tree: Any,
      cond_aux_tree: Any,
      body_aux_tree: Any,
  ):
    """Initializes the hijax primitive."""
    self.in_avals = (
        list(init_val_avals),
        list(cond_aux_avals),
        list(body_aux_avals),
    )

    history_state_avals = [_add_axis(x, max_steps) for x in init_val_avals]
    count_type = jax.core.ShapedArray(shape=(), dtype=_WHILE_LOOP_COUNT_DTYPE)
    self.out_aval = (list(init_val_avals), count_type, history_state_avals)
    # Static parameters.
    self.params = dict(
        cond_fun=cond_fun,
        body_fun=body_fun,
        max_steps=max_steps,
        init_val_tree=init_val_tree,
        cond_aux_tree=cond_aux_tree,
        body_aux_tree=body_aux_tree,
    )
    super().__init__()

  # Implementation, used for evaluation and lowering (e.g. under jit).
  def expand(self, *args):
    init_val_flat, cond_aux_flat, body_aux_flat = args
    return _while_loop_bounded_while_loop_fwd(
        self.params['cond_fun'],
        self.params['body_fun'],
        init_val_flat,
        self.params['max_steps'],
        cond_aux_flat,
        body_aux_flat,
        self.params['init_val_tree'],
        self.params['cond_aux_tree'],
        self.params['body_aux_tree'],
    )[0]

  # Reverse-mode: forward pass returns (primal_out, residuals).
  def vjp_fwd(self, nzs_in, /, *args):
    init_val_flat, cond_consts_flat, body_consts_flat = args
    return _while_loop_bounded_while_loop_fwd(
        self.params['cond_fun'],
        self.params['body_fun'],
        init_val_flat,
        self.params['max_steps'],
        cond_consts_flat,
        body_consts_flat,
        self.params['init_val_tree'],
        self.params['cond_aux_tree'],
        self.params['body_aux_tree'],
    )

  # Reverse-mode: backward pass maps (residuals, output cotangent) to a tuple
  # of input cotangents.
  def vjp_bwd_retval(self, res, g):
    return _while_loop_bounded_while_loop_bwd(
        self.params['cond_fun'],
        self.params['body_fun'],
        self.params['max_steps'],
        res,
        _instantiate_zeros(g),
        self.params['init_val_tree'],
        self.params['cond_aux_tree'],
        self.params['body_aux_tree'],
    )

  def jvp(self, primals, tangents):
    tangents = _instantiate_zeros(tangents)
    return jax.jvp(fun=self.expand, primals=primals, tangents=tangents)

  def batch_dim_rule(self, axis_data, in_dims):
    init_val_dims, cond_aux_dims, body_aux_dims = in_dims
    all_dims = list(init_val_dims) + list(cond_aux_dims) + list(body_aux_dims)
    is_batched = any(d is not None for d in all_dims)
    if not is_batched:
      return (list(init_val_dims), None, list(init_val_dims))
    out_state_dims = [d if d is not None else 0 for d in init_val_dims]
    out_count_dim = 0
    out_history_dims = [d if d is not None else 0 for d in init_val_dims]
    return (out_state_dims, out_count_dim, out_history_dims)


# As the history array could be longer than the number of steps executed, we
# initialize it with NaNs for floats and zeros for integers for unused indices.
def _init_history_array(x: jax.Array, max_steps: int) -> jax.Array:
  """Initializes a history array with NaNs or zeros."""
  shape = (max_steps,) + x.shape
  value = jnp.nan if jnp.issubdtype(x.dtype, jnp.floating) else 0
  return jnp.full(shape=shape, fill_value=value, dtype=x.dtype)


def _while_loop_bounded_while_loop_fwd(
    cond_fun,
    body_fun,
    init_val_flat,
    max_steps,
    cond_consts_flat,
    body_consts_flat,
    init_val_tree,
    cond_aux_tree,
    body_aux_tree,
):
  """Forward pass for while_loop_bounded_while_loop."""

  history_init_flat = tuple(
      _init_history_array(x, max_steps) for x in init_val_flat
  )

  init_carry = (
      jnp.array(0, dtype=_WHILE_LOOP_COUNT_DTYPE),
      tuple(init_val_flat),
      history_init_flat,
  )

  cond_consts = jax.tree.unflatten(cond_aux_tree, cond_consts_flat)
  body_consts = jax.tree.unflatten(body_aux_tree, body_consts_flat)

  def cond_tup(carry):
    step_idx, current_state_flat, _ = carry
    current_state = jax.tree.unflatten(init_val_tree, current_state_flat)
    return jnp.logical_and(
        step_idx < max_steps, cond_fun(cond_consts, current_state)
    )

  def body_tup(carry):
    step_idx, current_state_flat, history_flat = carry
    current_state = jax.tree.unflatten(init_val_tree, current_state_flat)
    next_state = body_fun(body_consts, current_state)
    next_state_flat, _ = jax.tree.flatten(next_state)
    next_history_flat = tuple(
        hist.at[step_idx].set(next_x)
        for hist, next_x in zip(history_flat, next_state_flat)
    )
    return step_idx + 1, tuple(next_state_flat), next_history_flat

  final_step_idx, final_state_flat, history_final_flat = jax.lax.while_loop(
      cond_tup, body_tup, init_carry
  )

  # (primal output, residual)
  return (list(final_state_flat), final_step_idx, list(history_final_flat)), (
      list(init_val_flat),
      list(history_final_flat),
      final_step_idx,
      list(cond_consts_flat),
      list(body_consts_flat),
  )


def _sanitize_cotangent_leaf(g_leaf, t_leaf):
  """Deals with issues like symbolic zeros and float0 arrays."""
  if isinstance(g_leaf, jax.Array):
    if (
        g_leaf.dtype == jax.dtypes.float0
        or not jnp.issubdtype(t_leaf.dtype, jnp.floating)
    ):
      return jnp.zeros_like(t_leaf)
    else:
      return g_leaf
  else:
    return jnp.zeros_like(t_leaf)


def _while_loop_bounded_while_loop_bwd(
    cond_fun,
    body_fun,
    max_steps,
    res,
    g,
    init_val_tree,
    cond_aux_tree,
    body_aux_tree,
):
  """Backward pass for while_loop_bounded_while_loop."""

  del cond_fun, max_steps, cond_aux_tree

  init_val_flat, history_flat, num_steps, cond_consts_flat, body_consts_flat = (
      res
  )
  g_final_state_flat, _, g_history_flat = g

  g_final_state_flat = [
      _sanitize_cotangent_leaf(g_leaf, t_leaf)
      for g_leaf, t_leaf in zip(g_final_state_flat, init_val_flat)
  ]
  g_history_flat = [
      _sanitize_cotangent_leaf(g_leaf, t_leaf)
      for g_leaf, t_leaf in zip(g_history_flat, history_flat)
  ]
  g_cond_consts_flat = [jnp.zeros_like(x) for x in cond_consts_flat]
  g_body_consts_init_flat = [jnp.zeros_like(x) for x in body_consts_flat]

  # Build a full history that includes init_val at index 0.
  full_history_flat = [
      jnp.concatenate([iv[None], h], axis=0)
      for iv, h in zip(init_val_flat, history_flat)
  ]
  # Backward from step num_steps-1 down to step 0.
  init_carry = (
      num_steps - 1,
      tuple(g_final_state_flat),
      tuple(g_body_consts_init_flat),
  )

  body_consts = jax.tree.unflatten(body_aux_tree, body_consts_flat)

  def cond_back(carry):
    t, _, _ = carry
    return t >= 0

  def body_back(carry):
    t, g_carry_flat, g_body_consts_flat = carry
    # Get the input to body_fun at forward step t.
    x_input_flat = [fh[t] for fh in full_history_flat]
    x_input = jax.tree.unflatten(init_val_tree, x_input_flat)

    # Get cotangent for the history output at step t.
    g_hist_t_flat = [gh[t] for gh in g_history_flat]

    # Total cotangent for the output of step t.
    g_active_flat = [gc + gh for gc, gh in zip(g_carry_flat, g_hist_t_flat)]
    g_active = jax.tree.unflatten(init_val_tree, g_active_flat)

    # Propagate through body_fun VJP.
    _, body_vjp = jax.vjp(body_fun, body_consts, x_input)
    vjp_outs = body_vjp(g_active)
    g_body_consts_step, g_prev = vjp_outs
    g_body_consts_step_flat, _ = jax.tree.flatten(g_body_consts_step)
    g_prev_flat, _ = jax.tree.flatten(g_prev)

    g_prev_flat = [
        _sanitize_cotangent_leaf(g_leaf, t_leaf)
        for g_leaf, t_leaf in zip(g_prev_flat, x_input_flat)
    ]
    g_body_consts_step_flat = [
        _sanitize_cotangent_leaf(g_leaf, t_leaf)
        for g_leaf, t_leaf in zip(g_body_consts_step_flat, body_consts_flat)
    ]
    next_g_body_consts_flat = tuple(
        (gc + gs) if jnp.issubdtype(t_leaf.dtype, jnp.floating) else gc
        for gc, gs, t_leaf in zip(
            g_body_consts_flat, g_body_consts_step_flat, body_consts_flat
        )
    )

    return t - 1, tuple(g_prev_flat), next_g_body_consts_flat

  _, g_carry_final_flat, g_body_consts_final_flat = jax.lax.while_loop(
      cond_back, body_back, init_carry
  )

  g_cond_consts_flat = [
      _sanitize_cotangent_leaf(g_leaf, t_leaf)
      for g_leaf, t_leaf in zip(g_cond_consts_flat, cond_consts_flat)
  ]
  g_body_consts_final_flat = [
      _sanitize_cotangent_leaf(g_leaf, t_leaf)
      for g_leaf, t_leaf in zip(g_body_consts_final_flat, body_consts_flat)
  ]

  return (
      list(g_carry_final_flat),
      list(g_cond_consts_flat),
      list(g_body_consts_final_flat),
  )

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

from typing import Any, Callable, Literal
import chex
import jax
from jax import numpy as jnp
from jax.experimental import hijax

type BooleanNumeric = Any  # A bool, or a Boolean array.
type PyTree = Any

_WHILE_LOOP_COUNT_DTYPE = jnp.int32

HiPrim = (
    hijax.VJPHiPrimitive  # pyrefly: ignore[missing-attribute]
    if jax.__version_info__ <= (0, 11, 1)
    else hijax.HiPrim
)


@jax.jit(
    static_argnames=['cond_fun', 'body_fun', 'max_steps', 'implementation'],
)
def while_loop_bounded[State](
    cond_fun: Callable[[State], BooleanNumeric],
    body_fun: Callable[[State], State],
    init_val: State,
    max_steps: int,
    implementation: Literal['scan', 'while_loop'] = 'while_loop',
) -> tuple[State, chex.Numeric, State]:
  """A bounded reverse-mode differentiable while_loop.

  `jax.lax.while_loop` is not reverse-mode differentiable. If we make the
  assumption that the number of steps is bounded, then this can be implemented
  using `jax.lax.scan` + `jax.lax.cond` or `jax.lax.while_loop` with a custom
  VJP.

  Args:
    cond_fun: As in jax.lax.while_loop.
    body_fun: As in jax.lax.while_loop.
    init_val: As in jax.lax.while_loop.
    max_steps: An integer, the maximum number of iterations the loop can
      perform.
    implementation: The implementation to use. 'scan' uses `jax.lax.scan` along
      with a `jax.lax.cond`, 'while_loop' uses `jax.lax.while_loop` and a custom
      VJP. The 'scan' implementation should mainly be used for testing.

  Returns:
    A tuple of:
      - The final state after `cond_fun` returns `False` or `max_steps` are
        reached.
      - The number of steps that were actually executed (integer scalar).
      - The output history: a pytree with the same structure as `init_val`
        where each leaf has an additional leading dimension of size
        `max_steps`. Index 0 is the initial state, and subsequent indices
        are the state after each step. The states for steps that are not
        executed are filled with NaNs if floats and 0s otherwise.
  """
  match implementation:
    case 'scan':
      return _while_loop_bounded_scan(
          cond_fun,
          body_fun,
          init_val,
          max_steps=max_steps,
          scan_unroll=1,
      )
    case 'while_loop':
      return _while_loop_bounded_while_loop(
          cond_fun, body_fun, init_val, max_steps
      )


@jax.jit(
    static_argnames=['cond_fun', 'body_fun', 'max_steps', 'scan_unroll'],
)
def _while_loop_bounded_scan[State](
    cond_fun: Callable[[State], BooleanNumeric],
    body_fun: Callable[[State], State],
    init_val: State,
    max_steps: int,
    scan_unroll: int = 1,
) -> tuple[State, chex.Numeric, State]:
  """A reverse-mode differentiable while_loop using jax.lax.scan."""
  # Initial carry for the scan: (current_state, counter,
  # while_loop_condition_met)
  initial_scan_carry = (
      init_val,
      jnp.array(0, dtype=_WHILE_LOOP_COUNT_DTYPE),
      jnp.array(True, dtype=jnp.bool_),
  )

  # NaN state to use for state for steps that are not executed.
  nan_state = jax.tree_util.tree_map(
      lambda x: jnp.full_like(x, fill_value=jnp.nan)
      if jnp.issubdtype(x.dtype, jnp.floating)
      else jnp.zeros_like(x),
      init_val,
  )

  def scan_body(carry, _):
    current_state, counter, cond_prev = carry
    # Only execute cond if the previous cond was True.
    should_execute_body = jax.lax.cond(
        cond_prev, cond_fun, lambda _: False, current_state
    )
    # If the `while_loop` would have terminated, we no-op.
    next_state = jax.lax.cond(
        should_execute_body, body_fun, lambda s: s, current_state
    )
    output_state = jax.lax.cond(
        should_execute_body, lambda: next_state, lambda: nan_state
    )
    next_counter = counter + should_execute_body.astype(jnp.int32)

    return (next_state, next_counter, should_execute_body), output_state

  (final_state, num_steps, _), stacked_outputs = jax.lax.scan(
      scan_body, initial_scan_carry, length=max_steps, unroll=scan_unroll
  )

  return final_state, num_steps, stacked_outputs


def _while_loop_bounded_while_loop[State](
    cond_fun: Callable[[State], BooleanNumeric],
    body_fun: Callable[[State], State],
    init_val: State,
    max_steps: int,
) -> tuple[State, chex.Numeric, State]:
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

  def fwd(init_val_flat, cond_consts_flat, body_consts_flat):
    return _while_loop_bounded_while_loop_fwd(
        converted_cond_fun,
        converted_body_fun,
        init_val_flat,
        max_steps,
        cond_consts_flat,
        body_consts_flat,
        init_val_tree,
        cond_aux_tree,
        body_aux_tree,
    )

  def bwd(res, g):
    return _while_loop_bounded_while_loop_bwd(
        converted_cond_fun,
        converted_body_fun,
        max_steps,
        res,
        g,
        init_val_tree,
        cond_aux_tree,
        body_aux_tree,
    )

  args = (init_val_flat, cond_aux_flat, body_aux_flat)
  in_avals = jax.tree.map(jax.typeof, args)
  init_val_avals = in_avals[0]
  out_aval = (
      init_val_avals,
      jax.core.ShapedArray(shape=(), dtype=_WHILE_LOOP_COUNT_DTYPE),
      [_add_axis(x, max_steps) for x in init_val_avals],
  )
  final_state_flat, final_step_idx, history_final_flat = (
      WhileLoopBoundedWhileLoop(fwd, bwd, in_avals, out_aval)(*args)
  )

  final_state = jax.tree.unflatten(init_val_tree, final_state_flat)
  history_final = jax.tree.unflatten(init_val_tree, history_final_flat)
  return final_state, final_step_idx, history_final


class WhileLoopBoundedWhileLoop(HiPrim):  # pyrefly: ignore[invalid-inheritance]
  """A bounded differentiable while_loop using jax.lax.while_loop."""
  fwd: Any
  bwd: Any

  def __init__(self, fwd, bwd, in_avals, out_aval):
    self.in_avals, self.out_aval = in_avals, out_aval
    self.params = dict(fwd=fwd, bwd=bwd)
    super().__init__()

  def expand(self, *args):
    return self.fwd(*args)[0]

  def vjp_fwd(self, _nzs_in, /, *args):
    return self.fwd(*args)

  def vjp_bwd_retval(self, res, g):
    return self.bwd(res, _instantiate_zeros(g))

  def jvp(self, primals, tangents):
    return jax.jvp(self.expand, primals, _instantiate_zeros(tangents))

  def batch(self, axis_data, args, dims):
    if not jax.tree.leaves(dims):
      return self(*args), jax.tree.map(lambda _: None, self.out_aval)  # pyrefly: ignore[not-callable]
    args = _bdims_at_front(axis_data, args, dims)
    in_avals = jax.tree.map(jax.typeof, args)
    out_dims = jax.tree.map(lambda _: 0, self.out_aval)
    out_aval = _unmap_avals(axis_data, self.out_aval, out_dims)
    fwd = _vmap_rule(axis_data, self.fwd, 0, 0)
    bwd = _vmap_rule(axis_data, self.bwd, 0, 0)
    prim = WhileLoopBoundedWhileLoop(fwd, bwd, in_avals, out_aval)
    return prim(*args), out_dims


def _add_axis(x: jax.core.ShapedArray, size: int) -> jax.core.ShapedArray:
  return jax.core.ShapedArray(shape=(size,) + x.shape, dtype=x.dtype)


if hasattr(hijax, 'vmap_rule'):
  _bdims_at_front = hijax.bdims_at_front
  _unmap_avals = hijax.unmap_avals
  _vmap_rule = hijax.vmap_rule
else:  # TODO(b/401588349): remove after jax 0.12.0 release

  def _bdims_at_front(axis_data, args, dims):
    return jax.tree.map(
        lambda x, d: jnp.moveaxis(x, d, 0)
        if d is not None
        else jnp.broadcast_to(x, (axis_data.size, *x.shape)),
        args,
        dims,
    )

  def _unmap_avals(axis_data, avals, dims):
    del dims
    return jax.tree.map(lambda a: _add_axis(a, axis_data.size), avals)

  def _vmap_rule(axis_data, f, in_axes, out_axes):
    return jax.vmap(
        f,
        in_axes=in_axes,
        out_axes=out_axes,
        axis_name=axis_data.name,
        axis_size=axis_data.size,
        spmd_axis_name=axis_data.spmd_name or axis_data.explicit_mesh_axis,
    )


def _instantiate_zeros(g: PyTree) -> PyTree:
  return jax.tree.map(
      hijax.instantiate_zeros, g, is_leaf=lambda x: isinstance(x, hijax.Zero)
  )


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
    if g_leaf.dtype == jax.dtypes.float0 or not jnp.issubdtype(
        t_leaf.dtype, jnp.floating
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

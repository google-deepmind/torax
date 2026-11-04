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
"""Custom JVP decorator for point-wise functions with narrow interfaces."""

from collections.abc import Callable, Sequence
import functools
import math
from typing import Any, Literal, overload

import jax
from jax import numpy as jnp
import numpy as np

type DifferentiationMode = Literal['auto', 'fwd', 'rev']


def _leaf_shape_dtype(leaf: Any) -> tuple[tuple[int, ...], Any]:
  """Returns `(shape, dtype)` for any JAX output leaf or `ShapeDtypeStruct`."""
  if isinstance(leaf, jax.ShapeDtypeStruct):
    return leaf.shape, leaf.dtype
  aval = jax.typeof(leaf)
  return aval.shape, aval.dtype


def _validate_and_count_outputs(
    leaves: Sequence[Any],
    batch_shape: tuple[int, ...],
    input_shape: tuple[int, ...],
) -> int:
  """Validates that every leaf shares `batch_shape` and counts float outputs."""
  b_ndim = len(batch_shape)
  n_out = 0
  for leaf in leaves:
    shape, dtype = _leaf_shape_dtype(leaf)
    if len(shape) < b_ndim or shape[:b_ndim] != batch_shape:
      raise ValueError(
          f'pointwise_jvp output leaf shape {shape} does not share '
          f'leading batch dimensions {batch_shape} of input shape '
          f'{input_shape}.'
      )
    if jnp.issubdtype(dtype, jnp.complexfloating):
      raise TypeError(
          f'pointwise_jvp does not support complex output leaves, got {dtype}.'
      )
    if jnp.issubdtype(dtype, jnp.floating):
      n_out += math.prod(shape[b_ndim:])
  return n_out


def _zero_tangent_like(leaf: Any) -> jax.Array | np.ndarray:
  """Returns a zero tangent matching JAX's tangent aval for `leaf`."""
  shape, dtype = _leaf_shape_dtype(leaf)
  if jnp.issubdtype(dtype, jnp.floating):
    return jnp.zeros(shape, dtype=dtype)
  return np.zeros(shape, dtype=jax.dtypes.float0)


@overload
def pointwise_jvp[T: Callable[..., Any]](
    fun: T,
    /,
    *,
    nondiff_argnums: Sequence[int] = (),
    nondiff_argnames: Sequence[str] = (),
    mode: DifferentiationMode = 'auto',
) -> T:
  ...


@overload
def pointwise_jvp[T: Callable[..., Any]](
    fun: None = None,
    /,
    *,
    nondiff_argnums: Sequence[int] = (),
    nondiff_argnames: Sequence[str] = (),
    mode: DifferentiationMode = 'auto',
) -> Callable[[T], T]:
  ...


def pointwise_jvp[T: Callable[..., Any]](
    fun: T | None = None,
    /,
    *,
    nondiff_argnums: Sequence[int] = (),
    nondiff_argnames: Sequence[str] = (),
    mode: DifferentiationMode = 'auto',
) -> T | Callable[[T], T]:
  """Precomputes local point-wise Jacobians for narrow-interface functions.

  When differentiating a composite function `R: R^D -> R^D` (e.g. a PDE residual
  with `D ~ 100` state variables) that internally evaluates a wide point-wise
  neural network `f: (*batch, n_in) -> OutPyTree[(*batch, ...)]` with small
  `n_in` or `n_out`, standard `jax.jacfwd` and `jax.jacrev` both propagate `D`
  tangent/cotangent vectors through the wide hidden layers of `f`.

  `pointwise_jvp` attaches a `@jax.custom_jvp` rule that exploits batch
  independence across `*batch` to precompute the local `(*batch, n_out, n_in)`
  derivative tensor in `min(n_in, n_out)` passes (using `jax.vjp` when
  `n_out <= n_in`, or `jax.linearize` when `n_in < n_out`), and contracts it
  with incoming tangents via a single linear `jnp.einsum`. Because the rule is
  linear in the tangent input, JAX automatically batches and transposes it for
  `jax.jacfwd`, `jax.jacrev`, `jax.grad`, `jax.vmap`, and `jax.lax.custom_root`.

  Args:
    fun: Callable taking static arguments (specified via `nondiff_argnums` or
      `nondiff_argnames`) and a single differentiable real floating-point array
      `x` of shape `(*batch, n_in)` (`x.ndim >= 1`), returning an array or
      PyTree of arrays whose leaves each have leading dimensions `*batch` and
      independent evaluations across `*batch`.
    nondiff_argnums: Non-negative positional indices of non-differentiable
      arguments (such as `0` for `self` on methods or a model wrapper instance).
    nondiff_argnames: Names of non-differentiable arguments.
    mode: Local differentiation mode used to form the point-wise Jacobian:
      `'rev'` uses `jax.vjp` (`n_out` backward passes, or 1 un-vmapped pass when
      `n_out == 1`), `'fwd'` uses `jax.linearize` (`n_in` tangent passes), and
      `'auto'` selects `'fwd'` when `n_in < n_out` and `'rev'` otherwise.

  Returns:
    A wrapped callable with identical primal behavior and an optimized JVP rule.
  """
  if mode not in ('auto', 'fwd', 'rev'):
    raise ValueError(
        f"mode must be one of 'auto', 'fwd', or 'rev', got {mode!r}."
    )
  if any(i < 0 for i in nondiff_argnums):
    raise ValueError(
        f'nondiff_argnums must be non-negative, got {nondiff_argnums!r}.'
    )

  def _decorate(fn: T) -> T:
    custom_fn = jax.custom_jvp(
        fn,
        nondiff_argnums=nondiff_argnums,
        nondiff_argnames=nondiff_argnames,
    )
    nondiff_set = frozenset(custom_fn.nondiff_argnums)

    def _jvp(*jvp_args: Any) -> tuple[Any, Any]:
      *static_args, primals, tangents = jvp_args
      if len(primals) != 1:
        raise ValueError(
            'pointwise_jvp expects exactly 1 differentiable array argument, '
            f'got {len(primals)}. Mark extra arguments via nondiff_argnums or '
            'nondiff_argnames.'
        )
      (x,) = primals
      (x_dot,) = tangents
      if not isinstance(x, (jax.Array, np.ndarray)):
        raise TypeError(
            'pointwise_jvp expects a single array argument, got '
            f'{type(x).__name__}.'
        )
      if x.ndim < 1:
        raise ValueError(
            'pointwise_jvp expects differentiable input with ndim >= 1 '
            f'(shape (*batch, n_in)), got shape {x.shape}.'
        )
      if not jnp.issubdtype(x.dtype, jnp.floating):
        raise TypeError(
            f'pointwise_jvp expects real floating-point input, got {x.dtype}.'
        )

      num_args = len(static_args) + 1

      def f_x(z: jax.Array | np.ndarray) -> Any:
        static_iter = iter(static_args)
        full_args = [
            next(static_iter) if idx in nondiff_set else z
            for idx in range(num_args)
        ]
        return fn(*full_args)

      batch_shape = x.shape[:-1]
      b_ndim = len(batch_shape)
      n_in = x.shape[-1]

      if n_in == 0:
        y = f_x(x)
        y_leaves, out_tree = jax.tree.flatten(y)
        _validate_and_count_outputs(y_leaves, batch_shape, x.shape)
        return y, jax.tree.unflatten(
            out_tree, [_zero_tangent_like(leaf) for leaf in y_leaves]
        )

      if mode == 'auto':
        out_avals = jax.eval_shape(f_x, x)
        n_out_est = _validate_and_count_outputs(
            jax.tree.leaves(out_avals), batch_shape, x.shape
        )
        resolved_mode = 'fwd' if n_in < n_out_est else 'rev'
      else:
        resolved_mode = mode

      if resolved_mode == 'fwd':
        y, jvp_lin = jax.linearize(f_x, x)
        y_leaves, out_tree = jax.tree.flatten(y)
        n_out = _validate_and_count_outputs(y_leaves, batch_shape, x.shape)
        if n_out == 0:
          return y, jax.tree.unflatten(
              out_tree, [_zero_tangent_like(leaf) for leaf in y_leaves]
          )

        leaf_metas = [_leaf_shape_dtype(leaf) for leaf in y_leaves]
        eye_in = jnp.eye(n_in, dtype=x.dtype)
        basis_in = jnp.broadcast_to(
            eye_in.reshape((n_in,) + (1,) * b_ndim + (n_in,)),
            (n_in, *batch_shape, n_in),
        )

        def _jvp_floating(dx: jax.Array) -> list[jax.Array]:
          dy_all = jax.tree.leaves(jvp_lin(dx))
          return [
              dy_leaf
              for (_, dtype), dy_leaf in zip(leaf_metas, dy_all, strict=True)
              if jnp.issubdtype(dtype, jnp.floating)
          ]

        dy_floating_iter = iter(jax.vmap(_jvp_floating)(basis_in))
        y_dot_leaves = []
        for shape, dtype in leaf_metas:
          if not jnp.issubdtype(dtype, jnp.floating):
            y_dot_leaves.append(np.zeros(shape, dtype=jax.dtypes.float0))
            continue
          dy_leaf = next(dy_floating_iter)
          k_leaf = math.prod(shape[b_ndim:])
          if k_leaf == 0:
            y_dot_leaves.append(jnp.zeros(shape, dtype=dtype))
            continue
          dy_flat = dy_leaf.reshape((n_in, *batch_shape, k_leaf))
          y_dot_flat = jnp.einsum('i...k,...i->...k', dy_flat, x_dot)
          y_dot_leaves.append(y_dot_flat.reshape(shape).astype(dtype))
        return y, jax.tree.unflatten(out_tree, y_dot_leaves)

      # Reverse-mode local Jacobian (`resolved_mode == 'rev'`).
      y, vjp_fn = jax.vjp(f_x, x)
      y_leaves, out_tree = jax.tree.flatten(y)
      n_out = _validate_and_count_outputs(y_leaves, batch_shape, x.shape)
      if n_out == 0:
        return y, jax.tree.unflatten(
            out_tree, [_zero_tangent_like(leaf) for leaf in y_leaves]
        )

      leaf_metas = [_leaf_shape_dtype(leaf) for leaf in y_leaves]

      # Single-output fast path: 1 backward pass with no inner jax.vmap.
      if n_out == 1:
        ones_leaves = []
        for shape, dtype in leaf_metas:
          if not jnp.issubdtype(dtype, jnp.floating):
            ones_leaves.append(np.zeros(shape, dtype=jax.dtypes.float0))
          elif math.prod(shape[b_ndim:]) == 1:
            ones_leaves.append(jnp.ones(shape, dtype=dtype))
          else:
            ones_leaves.append(jnp.zeros(shape, dtype=dtype))
        (dy_dx,) = vjp_fn(jax.tree.unflatten(out_tree, ones_leaves))
        y_dot_scalar = jnp.einsum('...i,...i->...', dy_dx, x_dot)
        y_dot_leaves = []
        for shape, dtype in leaf_metas:
          if not jnp.issubdtype(dtype, jnp.floating):
            y_dot_leaves.append(np.zeros(shape, dtype=jax.dtypes.float0))
          elif math.prod(shape[b_ndim:]) == 1:
            y_dot_leaves.append(y_dot_scalar.reshape(shape).astype(dtype))
          else:
            y_dot_leaves.append(jnp.zeros(shape, dtype=dtype))
        return y, jax.tree.unflatten(out_tree, y_dot_leaves)

      eye_dtype = jnp.result_type(*[
          dtype
          for _, dtype in leaf_metas
          if jnp.issubdtype(dtype, jnp.floating)
      ])
      eye_out = jnp.eye(n_out, dtype=eye_dtype)
      basis_leaves = []
      in_axes_leaves = []
      offset = 0
      for shape, dtype in leaf_metas:
        if not jnp.issubdtype(dtype, jnp.floating):
          basis_leaves.append(np.zeros(shape, dtype=jax.dtypes.float0))
          in_axes_leaves.append(None)
          continue
        k_leaf = math.prod(shape[b_ndim:])
        if k_leaf == 0:
          basis_leaves.append(jnp.zeros(shape, dtype=dtype))
          in_axes_leaves.append(None)
          continue
        eye_slice = eye_out[:, offset : offset + k_leaf].astype(dtype)
        offset += k_leaf
        trailing_shape = shape[b_ndim:]
        basis_leaves.append(
            jnp.broadcast_to(
                eye_slice.reshape((n_out,) + (1,) * b_ndim + trailing_shape),
                (n_out, *shape),
            )
        )
        in_axes_leaves.append(0)

      in_axes_tree = jax.tree.unflatten(out_tree, in_axes_leaves)
      basis_tree = jax.tree.unflatten(out_tree, basis_leaves)
      (dy_dx,) = jax.vmap(vjp_fn, in_axes=(in_axes_tree,))(basis_tree)
      y_dot_flat = jnp.einsum('o...i,...i->...o', dy_dx, x_dot)

      y_dot_leaves = []
      offset = 0
      for shape, dtype in leaf_metas:
        if not jnp.issubdtype(dtype, jnp.floating):
          y_dot_leaves.append(np.zeros(shape, dtype=jax.dtypes.float0))
          continue
        k_leaf = math.prod(shape[b_ndim:])
        if k_leaf == 0:
          y_dot_leaves.append(jnp.zeros(shape, dtype=dtype))
          continue
        leaf_dot = y_dot_flat[..., offset : offset + k_leaf]
        offset += k_leaf
        y_dot_leaves.append(leaf_dot.reshape(shape).astype(dtype))
      return y, jax.tree.unflatten(out_tree, y_dot_leaves)

    custom_fn.defjvp(_jvp)

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
      return custom_fn(*args, **kwargs)

    return wrapper  # pyrefly: ignore[bad-return]

  if fun is not None:
    return _decorate(fun)
  return _decorate

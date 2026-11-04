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

from typing import Any, Literal, cast
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax import numpy as jnp
import numpy as np
from torax._src import jax_utils


def _make_mlp(
    key: jax.Array,
    n_in: int,
    hidden_width: int,
    num_hidden: int,
    n_out: int,
    dtype: Any = jnp.float64,
):
  keys = jax.random.split(key, num_hidden + 1)
  layers = []
  in_dim = n_in
  for i in range(num_hidden):
    w_key, b_key = jax.random.split(keys[i])
    scale = jnp.sqrt(in_dim)
    w = jax.random.normal(w_key, (in_dim, hidden_width), dtype=dtype) / scale
    b = 0.1 * jax.random.normal(b_key, (hidden_width,), dtype=dtype)
    layers.append((w, b))
    in_dim = hidden_width
  w_key, b_key = jax.random.split(keys[-1])
  w_out = jax.random.normal(w_key, (in_dim, n_out), dtype=dtype) / jnp.sqrt(
      in_dim
  )
  b_out = 0.1 * jax.random.normal(b_key, (n_out,), dtype=dtype)

  def apply_mlp(x: jax.Array) -> jax.Array:
    h = x
    for w, b in layers:
      h = jnp.tanh(h @ w + b)
    return h @ w_out + b_out

  return apply_mlp


class _WrapperModel:

  def __init__(self, mlp_fn, scale: float):
    self._mlp_fn = mlp_fn
    self._scale = scale

  @jax_utils.pointwise_jvp(nondiff_argnums=(0,))
  def predict(self, x: jax.Array) -> dict[str, jax.Array]:
    raw = self._mlp_fn(x) * self._scale
    return {
        'flux_a': raw[..., :1],
        'flux_b': jax.nn.softplus(raw[..., 1:3]),
    }


class PointwiseJvpTest(parameterized.TestCase):

  @parameterized.product(
      batch_shape=[(), (7,), (3, 5)],
      n_out=[1, 4, 12],
      mode=['auto', 'fwd', 'rev'],
  )
  def test_jvp_jacfwd_jacrev_grad_parity(
      self,
      batch_shape: tuple[int, ...],
      n_out: int,
      mode: Literal['auto', 'fwd', 'rev'],
  ):
    n_in = 6
    mlp = _make_mlp(
        jax.random.key(10 + n_out),
        n_in=n_in,
        hidden_width=16,
        num_hidden=2,
        n_out=n_out,
    )
    fast_mlp = jax_utils.pointwise_jvp(mlp, mode=mode)

    x = jax.random.normal(
        jax.random.key(1), (*batch_shape, n_in), dtype=jnp.float64
    )
    v = jax.random.normal(
        jax.random.key(2), (*batch_shape, n_in), dtype=jnp.float64
    )

    y_ref, dy_ref = jax.jvp(mlp, (x,), (v,))
    y_fast, dy_fast = jax.jvp(fast_mlp, (x,), (v,))
    np.testing.assert_allclose(y_fast, y_ref, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(dy_fast, dy_ref, atol=1e-12, rtol=1e-12)

    jac_fwd_ref = jax.jit(jax.jacfwd(mlp))(x)
    jac_fwd_fast = jax.jit(jax.jacfwd(fast_mlp))(x)
    jac_rev_fast = jax.jit(jax.jacrev(fast_mlp))(x)
    np.testing.assert_allclose(
        jac_fwd_fast, jac_fwd_ref, atol=1e-12, rtol=1e-12
    )
    np.testing.assert_allclose(
        jac_rev_fast, jac_fwd_ref, atol=1e-12, rtol=1e-12
    )

    grad_ref = jax.jit(jax.grad(lambda z: jnp.sum(jnp.sin(mlp(z)))))(x)
    grad_fast = jax.jit(jax.grad(lambda z: jnp.sum(jnp.sin(fast_mlp(z)))))(x)
    np.testing.assert_allclose(grad_fast, grad_ref, atol=1e-12, rtol=1e-12)

  def test_pytree_and_scalar_per_point_outputs(self):
    mlp = _make_mlp(
        jax.random.key(20), n_in=5, hidden_width=12, num_hidden=2, n_out=4
    )

    def raw_fn(x: jax.Array):
      out = mlp(x)
      mask = (out[..., 0] > 0.0).astype(jnp.int32)
      return {
          'scalar_per_point': out[..., 0],
          'single_col': out[..., 1:2],
          'pair': out[..., 2:4],
          'ood_mask': mask,
      }

    fast_fn = jax_utils.pointwise_jvp(raw_fn)
    x = jax.random.normal(jax.random.key(21), (8, 5), dtype=jnp.float64)

    def loss_fn(fn):
      def _loss(u):
        res = fn(u)
        return (
            res['scalar_per_point']
            + res['single_col'][..., 0] * 2.0
            + jnp.sum(res['pair'] ** 2, axis=-1)
        )

      return _loss

    chex.assert_trees_all_equal(fast_fn(x), raw_fn(x))
    jac_direct_fast = jax.jit(jax.jacfwd(fast_fn))(x)
    jac_direct_ref = jax.jit(jax.jacfwd(raw_fn))(x)
    for key in ('scalar_per_point', 'single_col', 'pair'):
      np.testing.assert_allclose(
          jac_direct_fast[key], jac_direct_ref[key], atol=1e-12, rtol=1e-12
      )
    np.testing.assert_array_equal(
        jac_direct_fast['ood_mask'], jac_direct_ref['ood_mask']
    )

    batched_x = jax.random.normal(
        jax.random.key(22), (3, 8, 5), dtype=jnp.float64
    )
    np.testing.assert_allclose(
        jax.jit(jax.vmap(jax.jacfwd(loss_fn(fast_fn))))(batched_x),
        jax.jit(jax.vmap(jax.jacfwd(loss_fn(raw_fn))))(batched_x),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        jax.jit(jax.jacrev(loss_fn(fast_fn)))(x),
        jax.jit(jax.jacrev(loss_fn(raw_fn)))(x),
        atol=1e-12,
        rtol=1e-12,
    )

  def test_bound_method_and_nondiff_argnames(self):
    mlp = _make_mlp(
        jax.random.key(30), n_in=6, hidden_width=12, num_hidden=2, n_out=3
    )
    wrapper = _WrapperModel(mlp, scale=2.5)
    x = jax.random.normal(jax.random.key(31), (5, 6), dtype=jnp.float64)

    def ref_predict(z: jax.Array) -> dict[str, jax.Array]:
      raw = mlp(z) * 2.5
      return {
          'flux_a': raw[..., :1],
          'flux_b': jax.nn.softplus(raw[..., 1:3]),
      }

    chex.assert_trees_all_close(
        jax.jit(jax.jacfwd(wrapper.predict))(x),
        jax.jit(jax.jacfwd(ref_predict))(x),
        atol=1e-12,
        rtol=1e-12,
    )

    @jax_utils.pointwise_jvp(nondiff_argnames=('scale',))
    def scaled_fn(z: jax.Array, scale: float) -> jax.Array:
      return mlp(z) * scale

    np.testing.assert_allclose(
        jax.jit(jax.jacfwd(lambda z: scaled_fn(z, scale=1.7)))(x),
        jax.jit(jax.jacfwd(lambda z: mlp(z) * 1.7))(x),
        atol=1e-12,
        rtol=1e-12,
    )

  @parameterized.parameters('fwd', 'rev')
  def test_mixed_input_output_dtypes(self, mode: Literal['fwd', 'rev']):
    w64 = jax.random.normal(jax.random.key(40), (4, 2), dtype=jnp.float64)

    @jax_utils.pointwise_jvp(mode=mode)
    def f32_to_f64(x: jax.Array) -> jax.Array:
      return jnp.tanh(x.astype(jnp.float64) @ w64)

    @jax_utils.pointwise_jvp(mode=mode)
    def f64_to_f32(x: jax.Array) -> jax.Array:
      return jnp.tanh(x @ w64).astype(jnp.float32)

    x32 = jax.random.normal(jax.random.key(41), (6, 4), dtype=jnp.float32)
    x64 = jax.random.normal(jax.random.key(42), (6, 4), dtype=jnp.float64)

    jac_32_to_64 = jax.jit(jax.jacfwd(f32_to_f64))(x32)
    self.assertEqual(jac_32_to_64.dtype, jnp.float64)
    ref_32_to_64 = jax.jit(
        jax.jacfwd(lambda z: jnp.tanh(z.astype(jnp.float64) @ w64))
    )(x32)
    np.testing.assert_allclose(jac_32_to_64, ref_32_to_64, atol=1e-5, rtol=1e-5)

    jac_64_to_32 = jax.jit(jax.jacfwd(f64_to_f32))(x64)
    self.assertEqual(jac_64_to_32.dtype, jnp.float32)
    ref_64_to_32 = jax.jit(
        jax.jacfwd(lambda z: jnp.tanh(z @ w64).astype(jnp.float32))
    )(x64)
    np.testing.assert_allclose(jac_64_to_32, ref_64_to_32, atol=1e-6, rtol=1e-6)

  def test_outer_vmap_and_hessian_and_custom_root(self):
    mlp = _make_mlp(
        jax.random.key(50), n_in=4, hidden_width=10, num_hidden=2, n_out=2
    )
    fast_mlp = jax_utils.pointwise_jvp(mlp)

    # Outer vmap over batch of PDE state vectors.
    batched_u = jax.random.normal(jax.random.key(51), (3, 8), dtype=jnp.float64)
    proj_in = jax.random.normal(
        jax.random.key(52), (8, 5, 4), dtype=jnp.float64
    )
    proj_out = jax.random.normal(
        jax.random.key(53), (5, 2, 8), dtype=jnp.float64
    )

    def make_residual(fn):
      def res(u: jax.Array, scale: jax.Array) -> jax.Array:
        feats = jnp.einsum('d,dfi->fi', u, proj_in)
        flux = fn(feats) * scale
        return u + 0.1 * jnp.einsum('fo,fod->d', flux, proj_out) - 1.0

      return res

    res_ref = make_residual(mlp)
    res_fast = make_residual(fast_mlp)
    scale = jnp.array(0.5, dtype=jnp.float64)

    np.testing.assert_allclose(
        jax.jit(jax.vmap(jax.jacfwd(lambda u: res_fast(u, scale))))(batched_u),
        jax.jit(jax.vmap(jax.jacfwd(lambda u: res_ref(u, scale))))(batched_u),
        atol=1e-12,
        rtol=1e-12,
    )

    # Second-order derivatives (Hessian).
    x_small = jax.random.normal(jax.random.key(54), (3, 4), dtype=jnp.float64)
    np.testing.assert_allclose(
        jax.jit(jax.hessian(lambda z: jnp.sum(fast_mlp(z) ** 2)))(x_small),
        jax.jit(jax.hessian(lambda z: jnp.sum(mlp(z) ** 2)))(x_small),
        atol=1e-11,
        rtol=1e-11,
    )

    # Implicit differentiation through jax.lax.custom_root (TORAX solver).
    def solve_root(res_fn, s: jax.Array) -> jax.Array:
      u0 = jnp.ones(8, dtype=jnp.float64)

      def newton_solve(f, init):
        u = init
        for _ in range(8):
          u = u - jnp.linalg.solve(jax.jacfwd(f)(u), f(u))
        return u

      return jax.lax.custom_root(
          f=lambda u: res_fn(u, s),
          initial_guess=u0,
          solve=newton_solve,
          tangent_solve=lambda g, y: jnp.linalg.solve(jax.jacfwd(g)(y), y),
      )

    grad_root_ref = jax.jit(
        jax.grad(lambda s: jnp.sum(solve_root(res_ref, s) ** 2))
    )(scale)
    grad_root_fast = jax.jit(
        jax.grad(lambda s: jnp.sum(solve_root(res_fast, s) ** 2))
    )(scale)
    np.testing.assert_allclose(
        grad_root_fast, grad_root_ref, atol=1e-11, rtol=1e-11
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='qlknn_like_rev',
          n_in=10,
          n_out=8,
          batch=(26,),
          min_speedup=8.0,
      ),
      dict(
          testcase_name='single_out_rev',
          n_in=9,
          n_out=1,
          batch=(26,),
          min_speedup=30.0,
      ),
      dict(
          testcase_name='toric_like_fwd',
          n_in=10,
          n_out=297,
          batch=(),
          min_speedup=6.0,
      ),
  )
  def test_reduces_compiled_jacobian_flops(
      self,
      n_in: int,
      n_out: int,
      batch: tuple[int, ...],
      min_speedup: float,
  ):
    state_dim = 100
    mlp = _make_mlp(
        jax.random.key(60),
        n_in=n_in,
        hidden_width=128,
        num_hidden=4,
        n_out=n_out,
    )
    fast_mlp = jax_utils.pointwise_jvp(mlp)

    def make_composite(fn):
      def composite(u: jax.Array) -> jax.Array:
        if batch:
          # Sparse FVM-like local stencil mapping R^100 -> (26, n_in).
          n_pts = batch[0]
          cols = [
              u[i : i + n_pts] - 0.5 * u[i + 1 : i + n_pts + 1]
              for i in range(n_in)
          ]
          x = jnp.stack(cols, axis=-1)
          y = fn(x)
          flux = jnp.sum(y, axis=-1)
          return u.at[:n_pts].add(flux)
        else:
          # Global scalar features mapping R^100 -> (n_in,) -> (297,) -> R^100.
          x = u[:n_in] + 0.1 * u[n_in : 2 * n_in]
          y = fn(x)
          return u + y[:state_dim]

      return composite

    u0 = jnp.linspace(0.5, 1.5, state_dim, dtype=jnp.float64)
    comp_ref = jax.jit(jax.jacfwd(make_composite(mlp))).lower(u0).compile()
    comp_fast = (
        jax.jit(jax.jacfwd(make_composite(fast_mlp))).lower(u0).compile()
    )

    cost_ref = comp_ref.cost_analysis()
    cost_fast = comp_fast.cost_analysis()
    self.assertIsNotNone(cost_ref)
    self.assertIsNotNone(cost_fast)
    assert cost_ref is not None and cost_fast is not None
    flops_ref = cost_ref['flops']
    flops_fast = cost_fast['flops']
    self.assertLess(flops_fast * min_speedup, flops_ref)
    np.testing.assert_allclose(
        comp_fast(u0), comp_ref(u0), atol=1e-11, rtol=1e-11
    )

  def test_validation_errors(self):
    with self.assertRaisesRegex(ValueError, 'mode must be one of'):
      jax_utils.pointwise_jvp(
          lambda x: x,
          mode=cast(Literal['auto', 'fwd', 'rev'], 'invalid'),
      )

    with self.assertRaisesRegex(
        ValueError, 'nondiff_argnums must be non-negative'
    ):
      jax_utils.pointwise_jvp(lambda a, x: x, nondiff_argnums=(-1,))

    @jax_utils.pointwise_jvp
    def two_diff_args(x: jax.Array, y: jax.Array) -> jax.Array:
      return x + y

    with self.assertRaisesRegex(
        ValueError, 'exactly 1 differentiable array argument'
    ):
      ones = jnp.ones((2,))
      jax.jvp(two_diff_args, (ones, ones), (ones, ones))

    @jax_utils.pointwise_jvp
    def bad_cross_batch(x: jax.Array) -> jax.Array:
      return jnp.sum(x, axis=0)

    with self.assertRaisesRegex(ValueError, 'does not share leading batch'):
      jax.jacfwd(bad_cross_batch)(jnp.ones((4, 3)))

    @jax_utils.pointwise_jvp
    def scalar_in(x: jax.Array) -> jax.Array:
      return x * 2.0

    with self.assertRaisesRegex(ValueError, 'ndim >= 1'):
      jax.jvp(scalar_in, (jnp.array(1.0),), (jnp.array(1.0),))

    @jax_utils.pointwise_jvp
    def identity_fn(x: jax.Array) -> jax.Array:
      return x

    c = jnp.ones((3,), dtype=jnp.complex128)
    with self.assertRaisesRegex(TypeError, 'real floating-point input'):
      jax.jvp(identity_fn, (c,), (c,))

    @jax_utils.pointwise_jvp
    def complex_out_fn(x: jax.Array) -> jax.Array:
      return x.astype(jnp.complex128)

    with self.assertRaisesRegex(TypeError, 'complex output leaves'):
      jax.jacfwd(complex_out_fn)(jnp.ones((3,)))

    @jax_utils.pointwise_jvp
    def dict_in_fn(d: dict[str, jax.Array]) -> jax.Array:
      return d['a']

    with self.assertRaisesRegex(TypeError, 'single array argument'):
      jax.jacfwd(dict_in_fn)({'a': jnp.ones((3,))})

  @parameterized.parameters('auto', 'fwd', 'rev')
  def test_numpy_inputs_zero_sized_shapes_and_hessian_modes(
      self, mode: Literal['auto', 'fwd', 'rev']
  ):
    mlp_wide_out = _make_mlp(
        jax.random.key(70), n_in=3, hidden_width=8, num_hidden=2, n_out=5
    )
    mlp_single_out = _make_mlp(
        jax.random.key(71), n_in=4, hidden_width=8, num_hidden=2, n_out=1
    )
    fast_wide = jax_utils.pointwise_jvp(mlp_wide_out, mode=mode)
    fast_single = jax_utils.pointwise_jvp(mlp_single_out, mode=mode)

    # Eager numpy.ndarray inputs (not yet converted to jax.Array).
    x_np = np.linspace(-0.5, 0.5, 12, dtype=np.float64).reshape(4, 3)
    v_np = np.ones_like(x_np) * 0.25
    y_fast, dy_fast = jax.jvp(fast_wide, (x_np,), (v_np,))
    y_ref, dy_ref = jax.jvp(mlp_wide_out, (x_np,), (v_np,))
    np.testing.assert_allclose(y_fast, y_ref, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(dy_fast, dy_ref, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        jax.jacfwd(fast_wide)(x_np),
        jax.jacfwd(mlp_wide_out)(x_np),
        atol=1e-12,
        rtol=1e-12,
    )

    # Zero-sized batch dimension: shape (0, 3).
    x_empty_batch = jnp.zeros((0, 3), dtype=jnp.float64)
    self.assertEqual(
        jax.jit(jax.jacfwd(fast_wide))(x_empty_batch).shape, (0, 5, 0, 3)
    )
    x_empty_batch_single = jnp.zeros((0, 4), dtype=jnp.float64)
    self.assertEqual(
        jax.jit(jax.jacfwd(fast_single))(x_empty_batch_single).shape,
        (0, 1, 0, 4),
    )

    # Zero-sized input feature dimension: shape (4, 0).
    @jax_utils.pointwise_jvp(mode=mode)
    def zero_in_fn(x: jax.Array) -> jax.Array:
      return jnp.ones((x.shape[0], 2), dtype=x.dtype)

    x_zero_in = jnp.zeros((4, 0), dtype=jnp.float64)
    jac_zero_in = jax.jit(jax.jacfwd(zero_in_fn))(x_zero_in)
    self.assertEqual(jac_zero_in.shape, (4, 2, 4, 0))

    # Zero-sized float output leaf and mixed empty + non-empty leaves.
    def mixed_empty_raw(x: jax.Array) -> dict[str, Any]:
      return {
          'empty': jnp.zeros((*x.shape[:-1], 0), dtype=x.dtype),
          'single': jnp.sum(x**2, axis=-1, keepdims=True),
          'flag': jnp.zeros(x.shape[:-1], dtype=jnp.int32),
      }

    mixed_empty_fast = jax_utils.pointwise_jvp(mixed_empty_raw, mode=mode)
    x_mix = jax.random.normal(jax.random.key(72), (3, 4), dtype=jnp.float64)
    jac_mix_fast = jax.jit(jax.jacfwd(mixed_empty_fast))(x_mix)
    jac_mix_ref = jax.jit(jax.jacfwd(mixed_empty_raw))(x_mix)
    self.assertEqual(jac_mix_fast['empty'].shape, (3, 0, 3, 4))
    np.testing.assert_allclose(
        jac_mix_fast['single'], jac_mix_ref['single'], atol=1e-12, rtol=1e-12
    )

    # Zero total floating outputs (n_out == 0) and Python scalar leaves at
    # unbatched batch_shape == ().
    @jax_utils.pointwise_jvp(mode=mode)
    def int_only_fn(x: jax.Array) -> jax.Array:
      return (x[..., 0] > 0.0).astype(jnp.int32)

    y_int, dy_int = jax.jvp(int_only_fn, (x_mix,), (jnp.ones_like(x_mix),))
    self.assertEqual(y_int.shape, (3,))
    self.assertEqual(dy_int.dtype, jax.dtypes.float0)

    def unbatched_with_py_scalars(x: jax.Array) -> dict[str, Any]:
      return {'vec': jnp.sin(x), 'const_float': 2.5, 'const_int': 3}

    fast_py_scalars = jax_utils.pointwise_jvp(
        unbatched_with_py_scalars, mode=mode
    )
    x_1d = jnp.array([0.2, -0.4, 0.7], dtype=jnp.float64)
    np.testing.assert_allclose(
        jax.jit(jax.grad(lambda z: jnp.sum(fast_py_scalars(z)['vec'])))(x_1d),
        jnp.cos(x_1d),
        atol=1e-12,
        rtol=1e-12,
    )

    # Second-order derivatives (Hessian) across n_in < n_out and n_out == 1.
    x_hess = jax.random.normal(jax.random.key(73), (2, 3), dtype=jnp.float64)
    np.testing.assert_allclose(
        jax.jit(jax.hessian(lambda z: jnp.sum(fast_wide(z) ** 2)))(x_hess),
        jax.jit(jax.hessian(lambda z: jnp.sum(mlp_wide_out(z) ** 2)))(x_hess),
        atol=1e-11,
        rtol=1e-11,
    )
    x_hess_single = jax.random.normal(
        jax.random.key(74), (2, 4), dtype=jnp.float64
    )
    np.testing.assert_allclose(
        jax.jit(jax.hessian(lambda z: jnp.sum(fast_single(z) ** 2)))(
            x_hess_single
        ),
        jax.jit(jax.hessian(lambda z: jnp.sum(mlp_single_out(z) ** 2)))(
            x_hess_single
        ),
        atol=1e-11,
        rtol=1e-11,
    )


if __name__ == '__main__':
  absltest.main()

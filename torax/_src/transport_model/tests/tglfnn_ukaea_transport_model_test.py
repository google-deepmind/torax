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
import inspect

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from torax._src.transport_model import tglfnn_ukaea_transport_model


class TglfnnUkaeaTransportModelTest(parameterized.TestCase):

  def test_hash_and_eq_same(self):

    model1 = tglfnn_ukaea_transport_model.TGLFNNukaeaTransportModel(
        machine='multimachine'
    )
    model2 = tglfnn_ukaea_transport_model.TGLFNNukaeaTransportModel(
        machine='multimachine'
    )

    self.assertEqual(hash(model1), hash(model2))
    self.assertEqual(model1, model2)

  def test_predict_tglfnn_fluxes_jacobian_and_grad_parity(self):
    transport_model = tglfnn_ukaea_transport_model.TGLFNNukaeaTransportModel(
        machine='multimachine'
    )
    inputs = jax.random.normal(
        jax.random.key(0), (6, 13), dtype=jnp.float64
    ) * 0.1
    tangents = jax.random.normal(
        jax.random.key(1), (6, 13), dtype=jnp.float64
    )
    unwrapped_predict = inspect.unwrap(
        tglfnn_ukaea_transport_model._predict_tglfnn_fluxes
    )

    def fast_fn(x: jax.Array):
      return tglfnn_ukaea_transport_model._predict_tglfnn_fluxes(
          transport_model.model, x
      )

    def ref_fn(x: jax.Array):
      return unwrapped_predict(transport_model.model, x)

    y_fast, dy_fast = jax.jvp(fast_fn, (inputs,), (tangents,))
    y_ref, dy_ref = jax.jvp(ref_fn, (inputs,), (tangents,))
    for yf, yr, dyf, dyr in zip(y_fast, y_ref, dy_fast, dy_ref, strict=True):
      np.testing.assert_allclose(yf, yr, atol=1e-5, rtol=1e-5)
      np.testing.assert_allclose(dyf, dyr, atol=1e-5, rtol=1e-5)

    jac_fwd_fast = jax.jit(jax.jacfwd(fast_fn))(inputs)
    jac_fwd_ref = jax.jit(jax.jacfwd(ref_fn))(inputs)
    jac_rev_fast = jax.jit(jax.jacrev(fast_fn))(inputs)
    for jf, jr, jrev in zip(
        jac_fwd_fast, jac_fwd_ref, jac_rev_fast, strict=True
    ):
      np.testing.assert_allclose(jf, jr, atol=1e-5, rtol=1e-5)
      np.testing.assert_allclose(jrev, jr, atol=1e-5, rtol=1e-5)

    def loss_fn(fn):
      return lambda x: sum(jnp.sum(out**2) for out in fn(x))

    np.testing.assert_allclose(
        jax.jit(jax.grad(loss_fn(fast_fn)))(inputs),
        jax.jit(jax.grad(loss_fn(ref_fn)))(inputs),
        atol=1e-5,
        rtol=1e-5,
    )

    batched_inputs = jnp.stack([inputs, inputs * 0.9], axis=0)
    np.testing.assert_allclose(
        jax.jit(jax.vmap(jax.jacfwd(loss_fn(fast_fn))))(batched_inputs),
        jax.jit(jax.vmap(jax.jacfwd(loss_fn(ref_fn))))(batched_inputs),
        atol=1e-5,
        rtol=1e-5,
    )


if __name__ == '__main__':
  absltest.main()

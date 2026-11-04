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
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from fusion_surrogates.qlknn import qlknn_model
from fusion_surrogates.qlknn import qlknn_model_test_utils
import jax
import jax.numpy as jnp
import numpy as np
from torax._src.transport_model import qlknn_model_wrapper
from torax._src.transport_model import qualikiz_based_transport_model


class QlknnModelWrapperTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if qlknn_model_test_utils is None:
      self.skipTest('fusion_surrogates is not available.')
    # Create a test model on disk to be loaded by the wrapper.
    self._config = qlknn_model_test_utils.get_test_model_config()
    self._batch_dim = 10
    batch_dims = (1, self._batch_dim)
    model = qlknn_model_test_utils.init_model(self._config, batch_dims)
    self._model_file = tempfile.NamedTemporaryFile(
        'wb', suffix='.pkl', delete=False
    )
    self._flux_name_map = dict(
        (flux_name, f'torax_{flux_name}')
        for flux_name in qlknn_model_test_utils.get_test_flux_map().keys()
    )
    model.export_model(self._model_file.name)
    self._qlknn_model_wrapper = qlknn_model_wrapper.QLKNNModelWrapper(
        path=self._model_file.name, name='', flux_name_map=self._flux_name_map
    )

  def test_predict_shape(self):
    """Tests model output shape."""
    inputs = jnp.empty((self._batch_dim, len(self._config.input_names)))
    outputs = self._qlknn_model_wrapper.predict(inputs)
    self.assertLen(outputs, len(self._flux_name_map))
    for output in outputs.values():
      self.assertEqual(output.shape, (self._batch_dim, 1))

  def test_predict_names(self):
    """Tests model output names are the TORAX flux names."""
    inputs = jnp.empty((self._batch_dim, len(self._config.input_names)))
    outputs = self._qlknn_model_wrapper.predict(inputs)
    for flux_name in self._flux_name_map.values():
      self.assertIn(flux_name, outputs)

  def test_predict_jacobian_and_grad_parity(self):
    n_in = len(self._config.input_names)
    inputs = jax.random.uniform(
        jax.random.key(0),
        (self._batch_dim, n_in),
        minval=0.2,
        maxval=0.8,
    )
    tangents = jax.random.normal(
        jax.random.key(1), (self._batch_dim, n_in), dtype=inputs.dtype
    )
    unwrapped_predict = inspect.unwrap(
        qlknn_model_wrapper.QLKNNModelWrapper.predict
    )

    def ref_predict(x: jax.Array) -> dict[str, jax.Array]:
      return unwrapped_predict(self._qlknn_model_wrapper, x)

    y_fast, dy_fast = jax.jvp(
        self._qlknn_model_wrapper.predict, (inputs,), (tangents,)
    )
    y_ref, dy_ref = jax.jvp(ref_predict, (inputs,), (tangents,))
    for k in y_ref:
      np.testing.assert_allclose(y_fast[k], y_ref[k], atol=1e-6, rtol=1e-6)
      np.testing.assert_allclose(dy_fast[k], dy_ref[k], atol=1e-5, rtol=1e-5)

    jac_fwd_fast = jax.jit(jax.jacfwd(self._qlknn_model_wrapper.predict))(
        inputs
    )
    jac_fwd_ref = jax.jit(jax.jacfwd(ref_predict))(inputs)
    jac_rev_fast = jax.jit(jax.jacrev(self._qlknn_model_wrapper.predict))(
        inputs
    )
    for k in jac_fwd_ref:
      np.testing.assert_allclose(
          jac_fwd_fast[k], jac_fwd_ref[k], atol=1e-5, rtol=1e-5
      )
      np.testing.assert_allclose(
          jac_rev_fast[k], jac_fwd_ref[k], atol=1e-5, rtol=1e-5
      )

    def scalar_loss(pred_fn):
      return lambda x: sum(jnp.sum(v**2) for v in pred_fn(x).values())

    grad_fast_fn = jax.jit(
        jax.grad(scalar_loss(self._qlknn_model_wrapper.predict))
    )
    np.testing.assert_allclose(
        grad_fast_fn(inputs),
        jax.jit(jax.grad(scalar_loss(ref_predict)))(inputs),
        atol=1e-5,
        rtol=1e-5,
    )

    batched_inputs = jnp.stack([inputs, inputs * 1.05], axis=0)
    jac_vmap_fast = jax.jit(
        jax.vmap(jax.jacfwd(scalar_loss(self._qlknn_model_wrapper.predict)))
    )(batched_inputs)
    jac_vmap_ref = jax.jit(
        jax.vmap(jax.jacfwd(scalar_loss(ref_predict)))
    )(batched_inputs)
    np.testing.assert_allclose(
        jac_vmap_fast, jac_vmap_ref, atol=1e-5, rtol=1e-5
    )

  @mock.patch.object(
      qlknn_model.QLKNNModel, 'load_model_from_path', autospec=True
  )
  def test_load_model_from_path(self, mock_load_model_from_path):
    """Tests that the model is loaded from the path."""
    qlknn_model_wrapper.QLKNNModelWrapper(path='/my/foo.qlknn', name='bar')
    mock_load_model_from_path.assert_called_once_with('/my/foo.qlknn', 'bar')

  @mock.patch.object(
      qlknn_model.QLKNNModel, 'load_model_from_name', autospec=True
  )
  def test_load_model_from_name(self, mock_load_model_from_name):
    """Tests that the model is loaded from the name."""
    qlknn_model_wrapper.QLKNNModelWrapper(path='', name='bar')
    mock_load_model_from_name.assert_called_once_with('bar')

  @mock.patch.object(
      qlknn_model.QLKNNModel, 'load_default_model', autospec=True
  )
  def test_load_default_model(self, mock_load_default_model):
    """Tests that the default model is loaded."""
    qlknn_model_wrapper.QLKNNModelWrapper(path='', name='')
    mock_load_default_model.assert_called_once()

  def test_inputs_and_ranges(self):
    """Tests inputs_and_ranges property."""
    inputs_and_ranges = self._qlknn_model_wrapper.inputs_and_ranges
    self.assertEqual(list(inputs_and_ranges.keys()), self._config.input_names)

  def test_get_model_inputs_from_qualikiz_inputs(self):
    """Tests get_model_inputs_from_qualikiz_inputs method."""
    input_names = self._config.input_names

    mock_inputs = mock.MagicMock(
        spec=qualikiz_based_transport_model.QualikizInputs
    )

    for i, name in enumerate(input_names):
      setattr(mock_inputs, name, jnp.array([float(i)]))

    if 'Ani' in input_names:
      mock_inputs.Ani0 = jnp.array([77.0])
    if 'LogNuStar' in input_names:
      mock_inputs.log_nu_star_face = jnp.array([88.0])

    model_inputs = (
        self._qlknn_model_wrapper.get_model_inputs_from_qualikiz_inputs(
            mock_inputs
        )
    )

    self.assertEqual(model_inputs.shape, (1, len(input_names)))

    for i, name in enumerate(input_names):
      if name == 'Ani':
        expected_val = 77.0
      elif name == 'LogNuStar':
        expected_val = 88.0
      else:
        expected_val = float(i)
      self.assertEqual(model_inputs[0, i], expected_val)


if __name__ == '__main__':
  absltest.main()

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

import tempfile
from unittest import mock

import jax.numpy as jnp
from absl.testing import absltest
from absl.testing import parameterized
from fusion_surrogates.qlknn import qlknn_model
from fusion_surrogates.qlknn import qlknn_model_test_utils

from torax._src.transport_model import qlknn_model_wrapper


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

  # TODO(b/381134347): Add tests for get_model_inputs_from_qualikiz_inputs
  # and inputs_and_ranges.


if __name__ == '__main__':
  absltest.main()

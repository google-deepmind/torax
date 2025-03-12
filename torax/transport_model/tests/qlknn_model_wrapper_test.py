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

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from torax.transport_model import qlknn_model_wrapper
# pylint: disable=g-import-not-at-top
try:
  from fusion_transport_surrogates import qlknn_model_test_utils
except ImportError:
  qlknn_model_test_utils = None
# pylint: enable=g-import-not-at-top


def _get_test_flux_name_map():
  if qlknn_model_test_utils is None:
    return {}
  return dict(
      (flux_name, f'torax_{flux_name}')
      for flux_name in qlknn_model_test_utils.get_test_flux_map().keys()
  )


class QlknnModelWrapperTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if qlknn_model_test_utils is None:
      self.skipTest('fusion_transport_surrogates is not available.')
    # Create a test model on disk to be loaded by the wrapper.
    self._config = qlknn_model_test_utils.get_test_model_config()
    self._batch_dim = 10
    batch_dims = (1, self._batch_dim)
    model = qlknn_model_test_utils.init_model(self._config, batch_dims)
    self._model_file = tempfile.NamedTemporaryFile(
        'wb', suffix='.pkl', delete=False
    )
    self._flux_name_map = _get_test_flux_name_map()
    model.export_model(self._model_file.name)
    self._qlknn_model_wrapper = qlknn_model_wrapper.QLKNNModelWrapper(
        path=self._model_file.name,
        flux_name_map=self._flux_name_map
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

  # TODO(b/381134347): Add tests for get_model_inputs_from_qualikiz_inputs
  # and inputs_and_ranges.


if __name__ == '__main__':
  absltest.main()

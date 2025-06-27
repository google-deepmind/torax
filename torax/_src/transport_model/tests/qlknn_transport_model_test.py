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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy.testing as npt
from torax._src.transport_model import qlknn_10d
from torax._src.transport_model import qlknn_model_wrapper
from torax._src.transport_model import qlknn_transport_model


class QlknnTransportModelTest(parameterized.TestCase):

  def test_hash_and_eq(self):
    # Test that hash and eq are invariant to copying, so that they will work
    # correctly with jax's persistent cache
    qlknn_1 = qlknn_transport_model.QLKNNTransportModel('foo', 'bar')
    qlknn_2 = qlknn_transport_model.QLKNNTransportModel('foo', 'bar')
    self.assertEqual(qlknn_1, qlknn_2)
    self.assertEqual(hash(qlknn_1), hash(qlknn_2))
    mock_persistent_jax_cache = set([qlknn_1])
    self.assertIn(qlknn_2, mock_persistent_jax_cache)

  def test_hash_and_eq_different(self):
    qlknn_1 = qlknn_transport_model.QLKNNTransportModel('foo', 'bar')
    qlknn_2 = qlknn_transport_model.QLKNNTransportModel('baz', 'bar')
    self.assertNotEqual(qlknn_1, qlknn_2)
    self.assertNotEqual(hash(qlknn_1), hash(qlknn_2))
    mock_persistent_jax_cache = set([qlknn_1])
    self.assertNotIn(qlknn_2, mock_persistent_jax_cache)

  def test_hash_and_eq_different_by_name(self):
    qlknn_1 = qlknn_transport_model.QLKNNTransportModel('foo', 'bar')
    qlknn_2 = qlknn_transport_model.QLKNNTransportModel('foo', 'baz')
    self.assertNotEqual(qlknn_1, qlknn_2)
    self.assertNotEqual(hash(qlknn_1), hash(qlknn_2))
    mock_persistent_jax_cache = set([qlknn_1])
    self.assertNotIn(qlknn_2, mock_persistent_jax_cache)

  @parameterized.named_parameters(
      ('itg', {'itg': False}),
      ('tem', {'tem': False}),
      ('etg', {'etg': False}),
      ('etg_and_itg', {'etg': False, 'itg': False}),
  )
  def test_filter_model_output(self, include_dict):
    """Tests that the model output is properly filtered."""

    shape = (26,)
    itg_keys = ['qi_itg', 'qe_itg', 'pfe_itg']
    tem_keys = ['qe_tem', 'qi_tem', 'pfe_tem']
    etg_keys = ['qe_etg']
    model_output = dict(
        [(k, jnp.ones(shape)) for k in itg_keys + tem_keys + etg_keys]
    )
    filtered_model_output = qlknn_transport_model._filter_model_output(
        model_output=model_output,
        include_ITG=include_dict.get('itg', True),
        include_TEM=include_dict.get('tem', True),
        include_ETG=include_dict.get('etg', True),
    )
    for key in itg_keys:
      expected = (
          jnp.ones(shape) if include_dict.get('itg', True) else jnp.zeros(shape)
      )
      npt.assert_array_equal(filtered_model_output[key], expected)
    for key in tem_keys:
      expected = (
          jnp.ones(shape) if include_dict.get('tem', True) else jnp.zeros(shape)
      )
      npt.assert_array_equal(filtered_model_output[key], expected)
    for key in etg_keys:
      expected = (
          jnp.ones(shape) if include_dict.get('etg', True) else jnp.zeros(shape)
      )
      npt.assert_array_equal(filtered_model_output[key], expected)

  def test_clip_inputs(self):
    """Tests that the inputs are properly clipped."""
    feature_scan = jnp.array([
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        [1.0, 2.8, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    ])
    inputs_and_ranges = {
        'a': {'min': 0.0, 'max': 10.0},
        'b': {'min': 2.5, 'max': 10.0},
        'c': {'min': 0.0, 'max': 2.5},
        'd': {'min': 12.0, 'max': 15.0},
        'e': {'min': 0.0, 'max': 3.0},
    }
    clip_margin = 0.95
    expected = jnp.array([
        [1.0, 2.625, 2.375, 12.6, 2.85, 6.0, 7.0, 8.0, 9.0],
        [1.0, 2.8, 2.0, 12.6, 2.85, 6.0, 7.0, 8.0, 9.0],
    ])
    clipped_feature_scan = qlknn_transport_model.clip_inputs(
        feature_scan=feature_scan,
        inputs_and_ranges=inputs_and_ranges,
        clip_margin=clip_margin,
    )
    npt.assert_allclose(clipped_feature_scan, expected)

  @mock.patch.object(qlknn_model_wrapper, 'QLKNNModelWrapper', autospec=True)
  def test_get_model_with_path(self, mock_qlknn_model_wrapper):
    """Tests that the model is loaded from the path."""
    qlknn_transport_model.get_model(path='/my/foo.qlknn', name='bar')
    mock_qlknn_model_wrapper.assert_called_once_with('/my/foo.qlknn', 'bar')

  @mock.patch.object(qlknn_model_wrapper, 'QLKNNModelWrapper', autospec=True)
  def test_get_model_with_name_only(self, mock_qlknn_model_wrapper):
    """Tests that the model is loaded from the path."""
    qlknn_transport_model.get_model(path='', name='bar')
    mock_qlknn_model_wrapper.assert_called_once_with('', 'bar')

  @mock.patch.object(qlknn_model_wrapper, 'QLKNNModelWrapper', autospec=True)
  def test_get_model_without_path_or_name(self, mock_qlknn_model_wrapper):
    """Tests that the model is loaded from the path."""
    qlknn_transport_model.get_model(path='', name='')
    mock_qlknn_model_wrapper.assert_called_once_with('', '')

  @mock.patch.object(qlknn_10d, 'QLKNN10D', autospec=True)
  def test_get_model_from_path_qlknn10d(self, mock_qlknn_qlknn10d):
    """Tests that the model is loaded from the path."""
    qlknn_transport_model.get_model(path='/foo/qlknn_hyper', name='bar')
    mock_qlknn_qlknn10d.assert_called_once_with('/foo/qlknn_hyper', 'bar')

  @mock.patch.object(qlknn_10d, 'QLKNN10D', autospec=True)
  def test_get_model_from_name_qlknn10d_fails(self, mock_qlknn_qlknn10d):
    """Tests that the model is loaded from the path."""
    with self.assertRaises(ValueError):
      qlknn_transport_model.get_model(path='', name='qlknn10D')
    mock_qlknn_qlknn10d.assert_not_called()

  @mock.patch.object(qlknn_model_wrapper, 'QLKNNModelWrapper', autospec=True)
  def test_get_model_caching(self, mock_qlknn_model_wrapper):
    """Tests that the model is loaded only once."""
    qlknn_transport_model.get_model(path='', name='bar')
    qlknn_transport_model.get_model(path='', name='bar')
    qlknn_transport_model.get_model(path='', name='bar')
    qlknn_transport_model.get_model(path='', name='bar')
    mock_qlknn_model_wrapper.assert_called_once_with('', 'bar')


if __name__ == '__main__':
  absltest.main()

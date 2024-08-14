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

"""Unit tests for torax.transport_model.qlknn_wrapper."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy.testing as npt
from torax import core_profile_setters
from torax import geometry
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.sources import source_models as source_models_lib
from torax.transport_model import qlknn_wrapper


class QlknnWrapperTest(parameterized.TestCase):
  """Unit tests for the `torax.transport_model.qlknn_wrapper` module."""

  def test_qlknn_wrapper_cache_works(self):
    """Tests that QLKNN calls are properly cached."""
    # This test can uncover and changes to the data structures which break the
    # caching.
    qlknn = qlknn_wrapper.QLKNNTransportModel(
        qlknn_wrapper.get_default_model_path()
    )
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = geometry.build_circular_geometry()
    source_models_builder = source_models_lib.SourceModelsBuilder()
    source_models = source_models_builder()
    runtime_params_provider = runtime_params.make_provider(geo.torax_mesh)
    dynamic_runtime_params_slice = (
        runtime_params_slice.build_dynamic_runtime_params_slice(
            runtime_params=runtime_params_provider,
            transport=qlknn_wrapper.RuntimeParams(),
            sources=source_models_builder.runtime_params,
            geo=geo,
        )
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        source_models=source_models,
    )

    # Executing once should lead to a cache entry being created.
    qlknn(dynamic_runtime_params_slice, geo, core_profiles)
    cache_size = qlknn._combined._cache_size()  # pylint: disable=protected-access
    self.assertGreaterEqual(cache_size, 1)

    # Executing again should lead to the same cache entry being used.
    qlknn(dynamic_runtime_params_slice, geo, core_profiles)
    self.assertEqual(
        qlknn._combined._cache_size(),  # pylint: disable=protected-access
        cache_size)

  def test_hash_and_eq(self):
    # Test that hash and eq are invariant to copying, so that they will work
    # correctly with jax's persistent cache
    qlknn_1 = qlknn_wrapper.QLKNNTransportModel('foo')
    qlknn_2 = qlknn_wrapper.QLKNNTransportModel('foo')
    self.assertEqual(qlknn_1, qlknn_2)
    self.assertEqual(hash(qlknn_1), hash(qlknn_2))
    mock_persistent_jax_cache = set([qlknn_1])
    self.assertIn(qlknn_2, mock_persistent_jax_cache)

  def test_hash_and_eq_different(self):
    qlknn_1 = qlknn_wrapper.QLKNNTransportModel('foo')
    qlknn_2 = qlknn_wrapper.QLKNNTransportModel('bar')
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
    filtered_model_output = qlknn_wrapper.filter_model_output(
        model_output=model_output,
        include_ITG=include_dict.get('itg', True),
        include_TEM=include_dict.get('tem', True),
        include_ETG=include_dict.get('etg', True),
        zeros_shape=shape,
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


if __name__ == '__main__':
  absltest.main()

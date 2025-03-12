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

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy.testing as npt
from torax.config import build_runtime_params
from torax.config import runtime_params as general_runtime_params
from torax.core_profiles import initialization
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.pedestal_model import pydantic_model as pedestal_pydantic_model
from torax.sources import pydantic_model as sources_pydantic_model
from torax.sources import source_models as source_models_lib
from torax.transport_model import qlknn_transport_model


class QlknnTransportModelTest(parameterized.TestCase):

  def test_qlknn_transport_model_cache_works(self):
    """Tests that QLKNN calls are properly cached."""
    # This test can uncover and changes to the data structures which break the
    # caching.
    qlknn = qlknn_transport_model.QLKNNTransportModel(
        qlknn_transport_model.get_default_model_path()
    )
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    sources = sources_pydantic_model.Sources()
    pedestal = pedestal_pydantic_model.Pedestal()
    dynamic_runtime_params_slice = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider(
            runtime_params=runtime_params,
            transport=qlknn_transport_model.RuntimeParams(),
            sources=sources,
            torax_mesh=geo.torax_mesh,
            pedestal=pedestal,
        )(
            t=runtime_params.numerics.t_initial,
        )
    )
    static_slice = build_runtime_params.build_static_runtime_params_slice(
        runtime_params=runtime_params,
        sources=sources,
        torax_mesh=geo.torax_mesh,
    )
    source_models = source_models_lib.SourceModels(
        sources=sources.source_model_config
    )
    core_profiles = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
    )
    pedestal_model = pedestal.build_pedestal_model()
    pedestal_model_outputs = pedestal_model(
        dynamic_runtime_params_slice, geo, core_profiles
    )

    # Executing once should lead to a cache entry being created.
    qlknn(
        dynamic_runtime_params_slice, geo, core_profiles, pedestal_model_outputs
    )
    cache_size = qlknn._combined._cache_size()  # pylint: disable=protected-access
    self.assertGreaterEqual(cache_size, 1)

    # Executing again should lead to the same cache entry being used.
    qlknn(
        dynamic_runtime_params_slice, geo, core_profiles, pedestal_model_outputs
    )
    self.assertEqual(
        qlknn._combined._cache_size(),  # pylint: disable=protected-access
        cache_size,
    )

  def test_hash_and_eq(self):
    # Test that hash and eq are invariant to copying, so that they will work
    # correctly with jax's persistent cache
    qlknn_1 = qlknn_transport_model.QLKNNTransportModel('foo')
    qlknn_2 = qlknn_transport_model.QLKNNTransportModel('foo')
    self.assertEqual(qlknn_1, qlknn_2)
    self.assertEqual(hash(qlknn_1), hash(qlknn_2))
    mock_persistent_jax_cache = set([qlknn_1])
    self.assertIn(qlknn_2, mock_persistent_jax_cache)

  def test_hash_and_eq_different(self):
    qlknn_1 = qlknn_transport_model.QLKNNTransportModel('foo')
    qlknn_2 = qlknn_transport_model.QLKNNTransportModel('bar')
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
    filtered_model_output = qlknn_transport_model.filter_model_output(
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

  def test_runtime_params_builds_dynamic_params(self):
    runtime_params = qlknn_transport_model.RuntimeParams()
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    provider = runtime_params.make_provider(geo.torax_mesh)
    provider.build_dynamic_params(t=0.0)


if __name__ == '__main__':
  absltest.main()

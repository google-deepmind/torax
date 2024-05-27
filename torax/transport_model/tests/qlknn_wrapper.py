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
import jax
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
    qlknn = qlknn_wrapper.QLKNNTransportModel()
    # Caching only works when compiled.
    qlknn_jitted = jax.jit(qlknn)
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = geometry.build_circular_geometry()
    source_models = source_models_lib.SourceModels()
    dynamic_runtime_params_slice = (
        runtime_params_slice.build_dynamic_runtime_params_slice(
            runtime_params=runtime_params,
            transport=qlknn_wrapper.RuntimeParams(),
            sources=source_models.runtime_params,
        )
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        source_models=source_models,
    )
    qlknn_jitted(dynamic_runtime_params_slice, geo, core_profiles)
    # The call should be cached. If there was an error, the cache size would be
    # 0.
    self.assertGreaterEqual(
        qlknn._cached_combined.cache_info().currsize,  # pylint: disable=protected-access
        1,
    )

  def test_hash_and_eq(self):
    # Test that hash and eq are invariant to copying, so that they will work
    # correctly with jax's persistent cache
    qlknn_1 = qlknn_wrapper.QLKNNTransportModel()
    qlknn_2 = qlknn_wrapper.QLKNNTransportModel()
    self.assertEqual(qlknn_1, qlknn_2)
    self.assertEqual(hash(qlknn_1), hash(qlknn_2))
    mock_persistent_jax_cache = set([qlknn_1])
    self.assertIn(qlknn_2, mock_persistent_jax_cache)

  def test_prepare_qualikiz_inputs(self):
    """Tests that the Qualikiz inputs are properly prepared."""
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = geometry.build_circular_geometry()
    source_models = source_models_lib.SourceModels()
    dynamic_runtime_params_slice = (
        runtime_params_slice.build_dynamic_runtime_params_slice(
            runtime_params=runtime_params,
            transport=qlknn_wrapper.RuntimeParams(),
            sources=source_models.runtime_params,
        )
    )
    runtime_config_inputs = (
        qlknn_wrapper.QLKNNRuntimeConfigInputs.from_runtime_params_slice(
            dynamic_runtime_params_slice
        )
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        source_models=source_models,
    )

    model_inputs = qlknn_wrapper.prepare_qualikiz_inputs(
        runtime_config_inputs=runtime_config_inputs,
        geo=geo,
        core_profiles=core_profiles,
    )
    vector_keys = [
        'Zeff',
        'Ati',
        'Ate',
        'Ane',
        'Ani',
        'q',
        'smag',
        'x',
        'Ti_Te',
        'log_nu_star_face',
        'normni',
        'chiGB',
    ]
    scalar_keys = ['Rmaj', 'Rmin']
    expected_vector_length = 26
    for key in vector_keys:
      self.assertEqual(model_inputs[key].shape, (expected_vector_length,))
    for key in scalar_keys:
      self.assertEqual(model_inputs[key].shape, ())

  def test_make_core_transport(self):
    """Tests that the model output is properly converted to core transport."""
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = geometry.build_circular_geometry()
    source_models = source_models_lib.SourceModels()
    dynamic_runtime_params_slice = (
        runtime_params_slice.build_dynamic_runtime_params_slice(
            runtime_params=runtime_params,
            transport=qlknn_wrapper.RuntimeParams(),
            sources=source_models.runtime_params,
        )
    )
    runtime_config_inputs = (
        qlknn_wrapper.QLKNNRuntimeConfigInputs.from_runtime_params_slice(
            dynamic_runtime_params_slice
        )
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        source_models=source_models,
    )
    prepared_data = qlknn_wrapper.prepare_qualikiz_inputs(
        runtime_config_inputs=runtime_config_inputs,
        geo=geo,
        core_profiles=core_profiles,
    )
    expected_shape = (26,)
    qi = jnp.ones(expected_shape)
    qe = jnp.zeros(expected_shape)
    pfe = jnp.zeros(expected_shape)
    core_transport = qlknn_wrapper.make_core_transport(
        qi=qi,
        qe=qe,
        pfe=pfe,
        prepared_data=prepared_data,
        runtime_config_inputs=runtime_config_inputs,
        geo=geo,
        core_profiles=core_profiles,
    )
    self.assertEqual(core_transport.chi_face_ion.shape, expected_shape)
    self.assertEqual(core_transport.chi_face_el.shape, expected_shape)
    self.assertEqual(core_transport.d_face_el.shape, expected_shape)
    self.assertEqual(core_transport.v_face_el.shape, expected_shape)

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

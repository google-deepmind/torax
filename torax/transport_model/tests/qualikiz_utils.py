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

"""Unit tests for torax.transport_model.qualikiz_utils."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from torax import core_profile_setters
from torax import geometry
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.sources import source_models as source_models_lib
from torax.transport_model import qlknn_wrapper
from torax.transport_model import qualikiz_utils


class QualikizUtilsTest(parameterized.TestCase):
  """Unit tests for the `torax.transport_model.qualikiz_utils` module."""

  def test_prepare_qualikiz_inputs(self):
    """Tests that the Qualikiz inputs are properly prepared."""
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

    qualikiz_inputs = qualikiz_utils.prepare_qualikiz_inputs(
        zeff=runtime_config_inputs.Zeff,
        nref=runtime_config_inputs.nref,
        Ai=runtime_config_inputs.Ai,
        q_correction_factor=runtime_config_inputs.q_correction_factor,
        transport=runtime_config_inputs.transport,
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
      self.assertEqual(
          getattr(qualikiz_inputs, key).shape, (expected_vector_length,)
      )
    for key in scalar_keys:
      self.assertEqual(getattr(qualikiz_inputs, key).shape, ())

  def test_make_core_transport(self):
    """Tests that the model output is properly converted to core transport."""
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
    qualikiz_inputs = qualikiz_utils.prepare_qualikiz_inputs(
        zeff=runtime_config_inputs.Zeff,
        nref=runtime_config_inputs.nref,
        Ai=runtime_config_inputs.Ai,
        q_correction_factor=runtime_config_inputs.q_correction_factor,
        transport=runtime_config_inputs.transport,
        geo=geo,
        core_profiles=core_profiles,
    )
    expected_shape = (26,)
    qi = jnp.ones(expected_shape)
    qe = jnp.zeros(expected_shape)
    pfe = jnp.zeros(expected_shape)
    core_transport = qualikiz_utils.make_core_transport(
        qi=qi,
        qe=qe,
        pfe=pfe,
        qualikiz_inputs=qualikiz_inputs,
        transport=runtime_config_inputs.transport,
        geo=geo,
        core_profiles=core_profiles,
    )
    self.assertEqual(core_transport.chi_face_ion.shape, expected_shape)
    self.assertEqual(core_transport.chi_face_el.shape, expected_shape)
    self.assertEqual(core_transport.d_face_el.shape, expected_shape)
    self.assertEqual(core_transport.v_face_el.shape, expected_shape)


if __name__ == '__main__':
  absltest.main()

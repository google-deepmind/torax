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
import jax.numpy as jnp
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.pedestal_model import pedestal_model
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config


# pylint: disable=invalid-name
class CombinedTransportModelTest(absltest.TestCase):

  def test_call_implementation(self):
    config = default_configs.get_default_config_dict()
    config['transport'] = {
        'model_name': 'combined',
        'transport_models': [
            {'model_name': 'constant', 'rho_max': 0.2, 'chi_i': 1.0},
            {
                'model_name': 'constant',
                'rho_min': 0.2,
                'rho_max': 0.8,
                'chi_i': 2.0,
            },
            {'model_name': 'constant', 'rho_min': 0.5, 'chi_i': 3.0},
        ],
        'pedestal_transport_models': [{'model_name': 'constant', 'chi_i': 0.1}],
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    model = torax_config.transport.build_transport_model()
    geo = torax_config.geometry.build_provider(
        t=torax_config.numerics.t_initial
    )
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(
        t=torax_config.numerics.t_initial,
    )
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    core_profiles = initialization.initial_core_profiles(
        runtime_params,
        geo,
        source_models,
        neoclassical_models,
    )
    mock_pedestal_outputs = mock.create_autospec(
        pedestal_model.PedestalModelOutput,
        instance=True,
        rho_norm_ped_top=0.91,
    )

    transport_coeffs = model._call_implementation(
        runtime_params.transport,
        runtime_params,
        geo,
        core_profiles,
        mock_pedestal_outputs,
    )
    # Target:
    # - 0.1 for rho = [rho_ped_top, rho_max]
    # - 3 for rho = (0.8, rho_ped_top), to check pedestal overrides it
    # - 5 for rho = (0.5, 0.8], to check case where models overlap
    # - 2 for rho = (0.2, 0.5], to check case rho_min_1 == rho_max_2
    # - 1 for rho = [0, 0.2], to check case where rho_min = 0
    target = jnp.where(geo.rho_face_norm <= 0.91, 3.0, 0.1)
    target = jnp.where(geo.rho_face_norm <= 0.8, 5.0, target)
    target = jnp.where(geo.rho_face_norm <= 0.5, 2.0, target)
    target = jnp.where(geo.rho_face_norm <= 0.2, 1.0, target)
    np.testing.assert_allclose(transport_coeffs.chi_face_ion, target)

  def test_chi_min(self):
    config = default_configs.get_default_config_dict()
    config['transport'] = {
        'model_name': 'combined',
        'transport_models': [
            {'model_name': 'constant', 'rho_min': 0.5, 'chi_i': 2.0},
        ],
        'chi_min': 1.0,
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    model = torax_config.transport.build_transport_model()
    geo = torax_config.geometry.build_provider(
        t=torax_config.numerics.t_initial
    )
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(
        t=torax_config.numerics.t_initial,
    )
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    core_profiles = initialization.initial_core_profiles(
        runtime_params,
        geo,
        source_models,
        neoclassical_models,
    )
    mock_pedestal_outputs = mock.create_autospec(
        pedestal_model.PedestalModelOutput,
        instance=True,
        rho_norm_ped_top=0.91,
    )

    transport_coeffs = model(
        runtime_params,
        geo,
        core_profiles,
        mock_pedestal_outputs,
    )
    # Target:
    # - 1.0 for rho = [rho_ped_top, rho_max], set by chi_min
    # - 2.0 for rho = (0.5, rho_ped_top), set by the model
    # - 1.0 for rho = [0.0, 0.5], set by chi_min
    target = jnp.where(geo.rho_face_norm <= 0.91, 2.0, 1.0)
    target = jnp.where(geo.rho_face_norm <= 0.5, 1.0, target)
    np.testing.assert_allclose(transport_coeffs.chi_face_ion, target)

  def test_error_if_patches_set_on_children(self):
    config = default_configs.get_default_config_dict()
    config['transport'] = {
        'model_name': 'combined',
        'transport_models': [
            {'model_name': 'constant', 'apply_inner_patch': True},
            {'model_name': 'constant'},
        ],
        'pedestal_transport_models': [{'model_name': 'constant'}],
    }
    with self.assertRaisesRegex(
        ValueError, '(?=.*patch)(?=.*CombinedTransportModel)'
    ):
      model_config.ToraxConfig.from_dict(config)

  def test_error_if_patches_set_on_self(self):
    config = default_configs.get_default_config_dict()
    config['transport'] = {
        'model_name': 'combined',
        'transport_models': [
            {'model_name': 'constant'},
            {'model_name': 'constant'},
        ],
        'pedestal_transport_models': [{'model_name': 'constant'}],
        'apply_inner_patch': True,
    }
    with self.assertRaisesRegex(
        ValueError, '(?=.*patch)(?=.*CombinedTransportModel)'
    ):
      model_config.ToraxConfig.from_dict(config)

  def test_error_if_rho_min_or_rho_max_set(self):
    config = default_configs.get_default_config_dict()
    config['transport'] = {
        'model_name': 'combined',
        'transport_models': [
            {'model_name': 'constant'},
            {'model_name': 'constant'},
        ],
        'pedestal_transport_models': [{'model_name': 'constant'}],
        'rho_min': 0.1,
    }
    with self.assertRaisesRegex(
        ValueError, '(?=.*rho)(?=.*CombinedTransportModel)'
    ):
      model_config.ToraxConfig.from_dict(config)

    config['transport'] = {
        'model_name': 'combined',
        'transport_models': [
            {'model_name': 'constant'},
            {'model_name': 'constant'},
        ],
        'pedestal_transport_models': [{'model_name': 'constant'}],
        'rho_max': 0.9,
    }
    with self.assertRaisesRegex(
        ValueError, '(?=.*rho)(?=.*CombinedTransportModel)'
    ):
      model_config.ToraxConfig.from_dict(config)


if __name__ == '__main__':
  absltest.main()

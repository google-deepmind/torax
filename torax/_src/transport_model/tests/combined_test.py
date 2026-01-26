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
from torax._src.transport_model import combined
from torax._src.transport_model import enums
from torax._src.transport_model import transport_model as transport_model_lib


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

    transport_coeffs = model.call_implementation(
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

  def _build_model_and_params(self, transport_config):
    config = default_configs.get_default_config_dict()
    config['transport'] = transport_config
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
    return model, runtime_params, geo

  def test_merge_mode_overwrite(self):
    """Tests that OVERWRITE mode wipes previous values in active region."""
    # Model 1: Value 1.0 everywhere.
    # Model 2: Value 2.0 in rho > 0.5. MergeMode = OVERWRITE.
    config = {
        'model_name': 'combined',
        'transport_models': [
            {'model_name': 'constant', 'chi_i': 1.0},
            {
                'model_name': 'constant',
                'rho_min': 0.5,
                'chi_i': 2.0,
                'merge_mode': 'overwrite',
            },
        ],
        'chi_min': 0.0,
    }
    model, runtime_params, geo = self._build_model_and_params(config)
    mock_pedestal_outputs = mock.create_autospec(
        pedestal_model.PedestalModelOutput,
        instance=True,
        rho_norm_ped_top=1.0,  # No pedestal restriction for this test
    )

    coeffs = model.call_implementation(
        runtime_params.transport,
        runtime_params,
        geo,
        mock.ANY,
        mock_pedestal_outputs,
    )

    # Expected: 1.0 for rho <= 0.5, 2.0 for rho > 0.5
    target = jnp.where(geo.rho_face_norm <= 0.5, 1.0, 2.0)
    np.testing.assert_allclose(coeffs.chi_face_ion, target)

  def test_overwrite_locks_subsequent(self):
    """Tests that OVERWRITE mode prevents subsequent ADD models from modifying the region."""
    # Model 1: Overwrite, Value 1.0 in rho > 0.5.
    # Model 2: Add, Value 2.0 everywhere.
    config = {
        'model_name': 'combined',
        'transport_models': [
            {
                'model_name': 'constant',
                'rho_min': 0.5,
                'chi_i': 1.0,
                'merge_mode': 'overwrite',
            },
            {'model_name': 'constant', 'chi_i': 2.0, 'merge_mode': 'add'},
        ],
        'chi_min': 0.0,
    }
    model, runtime_params, geo = self._build_model_and_params(config)
    mock_pedestal_outputs = mock.create_autospec(
        pedestal_model.PedestalModelOutput,
        instance=True,
        rho_norm_ped_top=1.0,
    )

    coeffs = model.call_implementation(
        runtime_params.transport,
        runtime_params,
        geo,
        mock.ANY,
        mock_pedestal_outputs,
    )

    expected = jnp.where(geo.rho_face_norm <= 0.5, 2.0, 1.0)
    # Model 2 is locked out of rho > 0.5 by Model 1's OVERWRITE.
    np.testing.assert_allclose(coeffs.chi_face_ion, expected)

  def test_disable_channel_transparency(self):
    """Tests that disabling a channel makes the overwrite transparent for that channel."""
    # Model 1: Value 1.0 for chi_i and chi_e.
    # Model 2: Overwrite, Value 2.0 in rho > 0.5. BUT disable_chi_e = True.
    # Result: chi_i should be 1.0 then 2.0. chi_e should be 1.0 everywhere
    # (transparent overwrite).
    config = {
        'model_name': 'combined',
        'transport_models': [
            {'model_name': 'constant', 'chi_i': 1.0, 'chi_e': 1.0},
            {
                'model_name': 'constant',
                'rho_min': 0.5,
                'chi_i': 2.0,
                'chi_e': 2.0,
                'merge_mode': 'overwrite',
                'disable_chi_e': True,
            },
        ],
        'chi_min': 0.0,
    }
    model, runtime_params, geo = self._build_model_and_params(config)
    mock_pedestal_outputs = mock.create_autospec(
        pedestal_model.PedestalModelOutput,
        instance=True,
        rho_norm_ped_top=1.0,
    )

    coeffs = model.call_implementation(
        runtime_params.transport,
        runtime_params,
        geo,
        mock.ANY,
        mock_pedestal_outputs,
    )

    # chi_i: Overwritten in outer half.
    target_i = jnp.where(geo.rho_face_norm <= 0.5, 1.0, 2.0)
    np.testing.assert_allclose(coeffs.chi_face_ion, target_i)

    # chi_e: Not overwritten (transparent), so Model 1 value remains.
    # Model 2 contributes 0 because disable_chi_e=True, and it doesn't wipe
    # because it's disabled.
    target_e = jnp.ones_like(geo.rho_face_norm) * 1.0
    np.testing.assert_allclose(coeffs.chi_face_el, target_e)

  def test_none_handling_in_combine(self):
    """Tests that None values are preserved as None if no model writes to them."""
    # We use a mock model to return None for clear isolation.
    mock_model = mock.create_autospec(
        transport_model_lib.TransportModel, instance=True
    )
    # Return a structure with some None fields
    mock_coeffs = transport_model_lib.TurbulentTransport(
        chi_face_ion=jnp.array([1.0]),
        chi_face_el=jnp.array([1.0]),
        d_face_el=jnp.array([1.0]),
        v_face_el=jnp.array([1.0]),
        # Optional fields as None
        chi_face_el_bohm=None,
        chi_face_el_gyrobohm=None,
        chi_face_ion_bohm=None,
        chi_face_ion_gyrobohm=None,
    )
    mock_model.call_implementation.return_value = mock_coeffs
    mock_model.zero_out_disabled_channels.return_value = mock_coeffs

    # Manually instantiate CombinedTransportModel with our mock
    combined_model = combined.CombinedTransportModel(
        transport_models=(mock_model,),
        pedestal_transport_models=(),
    )

    # We need dummy params for the mock model
    mock_params = mock.Mock()
    mock_params.disable_chi_i = False
    mock_params.disable_chi_e = False
    mock_params.disable_D_e = False
    mock_params.disable_V_e = False
    mock_params.merge_mode = enums.MergeMode.ADD
    mock_params.rho_min = 0.0
    mock_params.rho_max = 1.0

    # We need a RuntimeParams for combined model
    combined_params = mock.create_autospec(
        combined.RuntimeParams, instance=True
    )
    combined_params.transport_model_params = [mock_params]
    combined_params.pedestal_transport_model_params = []

    geo = mock.Mock(spec=transport_model_lib.geometry.Geometry)
    geo.rho_face_norm = jnp.linspace(0, 1, 10)

    pedestal_output = mock.Mock(
        spec=transport_model_lib.pedestal_model_lib.PedestalModelOutput
    )
    pedestal_output.rho_norm_ped_top = 1.0

    runtime_params = mock.Mock()
    core_profiles = mock.Mock()

    coeffs = combined_model.call_implementation(
        combined_params, runtime_params, geo, core_profiles, pedestal_output
    )

    # Check that output has None for optional fields
    self.assertIsNone(coeffs.chi_face_ion_bohm)
    self.assertIsNone(coeffs.chi_face_el_bohm)
    # Check that main fields are arrays
    self.assertIsNotNone(coeffs.chi_face_ion)


if __name__ == '__main__':
  absltest.main()

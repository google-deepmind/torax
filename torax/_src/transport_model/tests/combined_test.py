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
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model_output as pedestal_model_output_lib
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config
from torax._src.transport_model import combined
from torax._src.transport_model import component
from torax._src.transport_model import enums
from torax._src.transport_model import runtime_params as transport_runtime_params_lib


# pylint: disable=invalid-name
class CombinedTransportModelTest(absltest.TestCase):

  def test_combining(self):
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
    config['pedestal'] = {'set_pedestal': True}
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
        pedestal_model_output_lib.PedestalModelOutput,
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
    config['pedestal'] = {'set_pedestal': True}
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
        pedestal_model_output_lib.PedestalModelOutput,
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

  def test_build_smoothing_matrix_zero_width_is_identity(self):
    """Tests that a zero smoothing width produces an identity matrix."""
    config = {
        'model_name': 'combined',
        'smoothing_width': 0.0,
        'transport_models': [{'model_name': 'constant', 'chi_i': 1.0}],
    }
    _, runtime_params, geo = self._build_model_and_params(config)
    mock_pedestal_outputs = mock.create_autospec(
        pedestal_model_output_lib.PedestalModelOutput,
        instance=True,
        rho_norm_ped_top=0.91,
    )
    matrix = combined._build_smoothing_matrix(
        runtime_params.transport,
        runtime_params,
        geo,
        mock_pedestal_outputs,
    )
    np.testing.assert_allclose(
        matrix, np.eye(len(geo.rho_face_norm)), atol=1e-7
    )

  def test_build_smoothing_matrix_row_sums_and_constant_invariance(self):
    """Tests that matrix rows sum to 1 and preserve constant profiles."""
    config = {
        'model_name': 'combined',
        'smoothing_width': 0.08,
        'transport_models': [{'model_name': 'constant', 'chi_i': 1.0}],
    }
    _, runtime_params, geo = self._build_model_and_params(config)
    mock_pedestal_outputs = mock.create_autospec(
        pedestal_model_output_lib.PedestalModelOutput,
        instance=True,
        rho_norm_ped_top=0.91,
    )
    matrix = combined._build_smoothing_matrix(
        runtime_params.transport,
        runtime_params,
        geo,
        mock_pedestal_outputs,
    )
    row_sums = np.sum(matrix, axis=1)
    np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-6)

    constant_profile = jnp.full_like(geo.rho_face_norm, 3.5)
    smoothed = jnp.dot(matrix, constant_profile)
    np.testing.assert_allclose(smoothed, constant_profile, atol=1e-6)

  def test_build_smoothing_matrix_zone_isolation(self):
    """Tests that matrix is identity for points outside defined smoothing zones."""
    config = {
        'model_name': 'combined',
        'smoothing_zones': [
            {'rho_min': 0.3, 'rho_max': 0.7, 'smoothing_width': 0.05},
        ],
        'transport_models': [{'model_name': 'constant', 'chi_i': 1.0}],
    }
    _, runtime_params, geo = self._build_model_and_params(config)
    mock_pedestal_outputs = mock.create_autospec(
        pedestal_model_output_lib.PedestalModelOutput,
        instance=True,
        rho_norm_ped_top=0.91,
    )
    matrix = combined._build_smoothing_matrix(
        runtime_params.transport,
        runtime_params,
        geo,
        mock_pedestal_outputs,
    )

    outside_mask = (geo.rho_face_norm < 0.3) | (geo.rho_face_norm > 0.7)
    outside_indices = np.where(outside_mask)[0]
    for idx in outside_indices:
      expected_row = np.zeros(len(geo.rho_face_norm))
      expected_row[idx] = 1.0
      np.testing.assert_allclose(matrix[idx], expected_row, atol=1e-7)

  def test_build_smoothing_matrix_pedestal_boundary_isolation(self):
    """Tests that smoothing matrix is identity at/above pedestal top in IBC mode."""
    config = default_configs.get_default_config_dict()
    config['transport'] = {
        'model_name': 'combined',
        'smoothing_width': 0.08,
        'transport_models': [{'model_name': 'constant', 'chi_i': 1.0}],
    }
    config['pedestal'] = {
        'set_pedestal': True,
        'mode': 'INTERNAL_BOUNDARY_CONDITION',
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(t=torax_config.numerics.t_initial)
    geo = torax_config.geometry.build_provider(
        t=torax_config.numerics.t_initial
    )
    mock_pedestal_outputs = mock.create_autospec(
        pedestal_model_output_lib.PedestalModelOutput,
        instance=True,
        rho_norm_ped_top=0.8,
    )
    matrix = combined._build_smoothing_matrix(
        runtime_params.transport,
        runtime_params,
        geo,
        mock_pedestal_outputs,
    )

    pedestal_mask = geo.rho_face_norm >= 0.8
    pedestal_indices = np.where(pedestal_mask)[0]
    for idx in pedestal_indices:
      expected_row = np.zeros(len(geo.rho_face_norm))
      expected_row[idx] = 1.0
      np.testing.assert_allclose(matrix[idx], expected_row, atol=1e-7)

  def test_smoothing_zones(self):
    """Tests that smoothing_zones smoothes transport coefficients in the specified region."""
    config = default_configs.get_default_config_dict()
    config['transport'] = {
        'model_name': 'combined',
        'smoothing_zones': [
            {'rho_min': 0.3, 'rho_max': 0.7, 'smoothing_width': 0.08},
        ],
        'transport_models': [
            {'model_name': 'constant', 'rho_max': 0.5, 'chi_i': 1.0},
            {'model_name': 'constant', 'rho_min': 0.5, 'chi_i': 5.0},
        ],
        'chi_min': 0.0,
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    model = torax_config.transport.build_transport_model()
    geo = torax_config.geometry.build_provider(
        t=torax_config.numerics.t_initial
    )
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(t=torax_config.numerics.t_initial)
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    core_profiles = initialization.initial_core_profiles(
        runtime_params, geo, source_models, neoclassical_models
    )
    mock_pedestal_outputs = mock.create_autospec(
        pedestal_model_output_lib.PedestalModelOutput,
        instance=True,
        rho_norm_ped_top=0.95,
    )
    coeffs = model(
        runtime_params,
        geo,
        core_profiles,
        mock_pedestal_outputs,
    )
    self.assertEqual(coeffs.chi_face_ion.shape, geo.rho_face_norm.shape)

    # 1. Unsmoothed regions: rho < 0.25 should equal 1.0,
    # rho > 0.75 should equal 5.0
    unsmoothed_left = geo.rho_face_norm < 0.25
    unsmoothed_right = geo.rho_face_norm > 0.75
    np.testing.assert_allclose(
        coeffs.chi_face_ion[unsmoothed_left], 1.0, atol=1e-5
    )
    np.testing.assert_allclose(
        coeffs.chi_face_ion[unsmoothed_right], 5.0, atol=1e-5
    )

    # 2. Inside smoothing zone: left of step (rho in (0.4, 0.5))
    # smoothed upwards (> 1.0), right of step (rho in (0.5, 0.6))
    # smoothed downwards (< 5.0).
    near_step_left = np.where(
        (geo.rho_face_norm > 0.4) & (geo.rho_face_norm < 0.5)
    )[0]
    near_step_right = np.where(
        (geo.rho_face_norm > 0.5) & (geo.rho_face_norm < 0.6)
    )[0]
    self.assertTrue(np.all(coeffs.chi_face_ion[near_step_left] > 1.0))
    self.assertTrue(np.all(coeffs.chi_face_ion[near_step_right] < 5.0))

  def test_smoothing_width_shortcut(self):
    """Tests that setting smoothing_width applies full-domain smoothing."""
    config_small = {
        'model_name': 'combined',
        'smoothing_width': 0.02,
        'transport_models': [
            {'model_name': 'constant', 'rho_max': 0.5, 'chi_i': 1.0},
            {'model_name': 'constant', 'rho_min': 0.5, 'chi_i': 5.0},
        ],
        'chi_min': 0.0,
    }
    config_large = {
        'model_name': 'combined',
        'smoothing_width': 0.12,
        'transport_models': [
            {'model_name': 'constant', 'rho_max': 0.5, 'chi_i': 1.0},
            {'model_name': 'constant', 'rho_min': 0.5, 'chi_i': 5.0},
        ],
        'chi_min': 0.0,
    }

    model_small, params_small, geo = self._build_model_and_params(config_small)
    model_large, params_large, _ = self._build_model_and_params(config_large)

    mock_pedestal_outputs = mock.create_autospec(
        pedestal_model_output_lib.PedestalModelOutput,
        instance=True,
        rho_norm_ped_top=1.0,
    )

    coeffs_small = model_small(
        params_small,
        geo,
        mock.ANY,
        mock_pedestal_outputs,
    )
    coeffs_large = model_large(
        params_large,
        geo,
        mock.ANY,
        mock_pedestal_outputs,
    )

    idx_left = np.where(
        (geo.rho_face_norm > 0.35) & (geo.rho_face_norm < 0.45)
    )[0]
    self.assertTrue(
        np.all(
            coeffs_large.chi_face_ion[idx_left]
            > coeffs_small.chi_face_ion[idx_left]
        )
    )

  def test_error_if_pedestal_model_defines_rho_min(self):
    config = default_configs.get_default_config_dict()
    config['transport'] = {
        'model_name': 'combined',
        'transport_models': [{'model_name': 'constant'}],
        'pedestal_transport_models': [
            {'model_name': 'constant', 'rho_min': 0.1}
        ],
    }
    with self.assertRaisesRegex(
        ValueError, 'rho_min and rho_max not supported'
    ):
      model_config.ToraxConfig.from_dict(config)

  def test_error_if_pedestal_model_defines_rho_max(self):
    config = default_configs.get_default_config_dict()
    config['transport'] = {
        'model_name': 'combined',
        'transport_models': [{'model_name': 'constant'}],
        'pedestal_transport_models': [
            {'model_name': 'constant', 'rho_max': 0.9}
        ],
    }
    with self.assertRaisesRegex(
        ValueError, 'rho_min and rho_max not supported'
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
        pedestal_model_output_lib.PedestalModelOutput,
        instance=True,
        rho_norm_ped_top=1.0,  # No pedestal restriction for this test
    )

    coeffs = model(
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
        pedestal_model_output_lib.PedestalModelOutput,
        instance=True,
        rho_norm_ped_top=1.0,
    )

    coeffs = model(
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
        pedestal_model_output_lib.PedestalModelOutput,
        instance=True,
        rho_norm_ped_top=1.0,
    )

    coeffs = model(
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
        component.ComponentTransportModel, instance=True
    )
    # Return a structure with some None fields
    mock_coeffs = component.TurbulentTransport(
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
    mock_model.return_value = mock_coeffs

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
        transport_runtime_params_lib.CombinedRuntimeParams, instance=True
    )
    combined_params.core_transport_model_params = [mock_params]
    combined_params.pedestal_transport_model_params = []
    # Set clipping and smoothing params so __call__ doesn't crash on mocks.
    combined_params.chi_min = 0.0
    combined_params.chi_max = 100.0
    combined_params.D_e_min = 0.0
    combined_params.D_e_max = 100.0
    combined_params.V_e_min = -100.0
    combined_params.V_e_max = 100.0
    combined_params.smoothing_width = 0.0
    combined_params.smoothing_zones = ()

    geo = mock.Mock(spec=geometry.Geometry)
    geo.rho_face_norm = jnp.linspace(0, 1, 10)

    pedestal_output = mock.Mock(
        spec=pedestal_model_output_lib.PedestalModelOutput
    )
    pedestal_output.rho_norm_ped_top = 1.0

    runtime_params = mock.Mock()
    runtime_params.transport = combined_params
    core_profiles = mock.Mock()

    coeffs = combined_model(
        runtime_params, geo, core_profiles, pedestal_output
    )

    # Check that output has None for optional fields
    self.assertIsNone(coeffs.chi_face_ion_bohm)
    self.assertIsNone(coeffs.chi_face_el_bohm)
    # Check that main fields are arrays
    self.assertIsNotNone(coeffs.chi_face_ion)


if __name__ == '__main__':
  absltest.main()

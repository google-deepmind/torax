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
import subprocess
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib
from torax._src.test_utils import core_profile_helpers
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config

# pylint: disable=g-import-not-at-top
try:
  from torax._src.transport_model import qualikiz_transport_model  # pylint: disable=unused-import

  _QUALIKIZ_TRANSPORT_MODEL_AVAILABLE = True
except ImportError:
  _QUALIKIZ_TRANSPORT_MODEL_AVAILABLE = False
# pylint: enable=g-import-not-at-top


class QualikizTransportModelTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('with_jit', True),
      ('without_jit', False),
  )
  def test_call(self, jit: bool):
    """Tests that the model can be called."""
    # Test prerequisites
    if not _QUALIKIZ_TRANSPORT_MODEL_AVAILABLE:
      self.skipTest('Qualikiz transport model is not available.')

    # Building the model inputs.
    config = default_configs.get_default_config_dict()
    config['transport'] = {
        'core_transport_models': {
            'qualikiz': {
                'model_name': 'qualikiz',
                'rotation_mode': 'full_radius',
                'rotation_multiplier': 2.0,
            }
        },
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    core_profiles, runtime_params, geo = (
        core_profile_helpers.make_initial_inputs(torax_config)
    )

    # Mocking the actual call to QuaLiKiz and its results.
    mock_process = mock.Mock()
    mock_process.communicate.return_value = (b'stdout', b'stderr')
    # The first call is expecting a 2D array, the others should be 1D arrays.
    num_data = core_profiles.n_e.face_value().shape[0]
    fake_qualikiz_results = [
        np.ones((num_data, 2)),
        np.ones(num_data),
        np.ones(num_data),
    ]
    with (
        mock.patch.object(subprocess, 'Popen', return_value=mock_process),
        mock.patch.object(np, 'loadtxt', side_effect=fake_qualikiz_results),
        mock.patch.object(
            qualikiz_transport_model,
            '_extract_qualikiz_plan',
            wraps=qualikiz_transport_model._extract_qualikiz_plan,
        ) as extract_plan_spy,
    ):
      # Calling the model
      test_model = torax_config.transport.build_transport_model()
      model_call = jax.jit(test_model.__call__) if jit else test_model.__call__
      two_point_mask = np.zeros_like(geo.rho_face_norm, dtype=bool)
      coeffs = model_call(
          runtime_params,
          geo,
          core_profiles,
          pedestal_transition_state_lib.PedestalTransitionState.empty(),
          two_point_mask,
      )
      expected_shape = geo.rho_face_norm.shape
      self.assertEqual(coeffs.total.chi_face_ion.shape, expected_shape)
      self.assertEqual(coeffs.total.chi_face_el.shape, expected_shape)
      self.assertEqual(coeffs.total.d_face_el.shape, expected_shape)
      self.assertEqual(coeffs.total.v_face_el.shape, expected_shape)
      self.assertTrue(np.all(np.isfinite(coeffs.total.chi_face_ion)))
      self.assertTrue(np.all(np.isfinite(coeffs.total.chi_face_el)))
      self.assertTrue(np.all(np.isfinite(coeffs.total.d_face_el)))
      self.assertTrue(np.all(np.isfinite(coeffs.total.v_face_el)))

    # rotation_multiplier is already applied in gamma_E_QLK, so gammaE passed
    # to QuaLiKiz must not be scaled again.
    plan = qualikiz_transport_model._extract_qualikiz_plan(
        **extract_plan_spy.call_args.kwargs
    )
    np.testing.assert_allclose(
        plan['scan_dict']['gammaE'],
        extract_plan_spy.call_args.kwargs['qualikiz_inputs'].gamma_E_QLK,
    )


if __name__ == '__main__':
  absltest.main()

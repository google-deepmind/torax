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

import os
import subprocess
from unittest import mock

from absl.testing import absltest
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.pedestal_model import pedestal_model
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config

# pylint: disable=g-import-not-at-top
try:
  from torax._src.transport_model import qualikiz_transport_model

  _QUALIKIZ_TRANSPORT_MODEL_AVAILABLE = True
except ImportError:
  _QUALIKIZ_TRANSPORT_MODEL_AVAILABLE = False
# pylint: enable=g-import-not-at-top


class QualikizTransportModelTest(absltest.TestCase):

  def setUp(self):
    os.environ['TORAX_COMPILATION_ENABLED'] = '0'
    super().setUp()

  def tearDown(self):
    os.environ['TORAX_COMPILATION_ENABLED'] = '1'
    super().tearDown()

  def test_call(self):
    """Tests that the model can be called."""
    # Test prerequisites
    if not _QUALIKIZ_TRANSPORT_MODEL_AVAILABLE:
      self.skipTest('Qualikiz transport model is not available.')

    # Building the model inputs.
    config = default_configs.get_default_config_dict()
    config['transport'] = {'model_name': 'qualikiz'}
    torax_config = model_config.ToraxConfig.from_dict(config)
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    dynamic_runtime_params_slice = (
        build_runtime_params.RuntimeParamsProvider.from_config(
            torax_config
        )(
            t=torax_config.numerics.t_initial,
        )
    )
    geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
    core_profiles = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
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
    with mock.patch.object(subprocess, 'Popen', return_value=mock_process):
      with mock.patch.object(np, 'loadtxt', side_effect=fake_qualikiz_results):

        # Calling the model
        test_model = qualikiz_transport_model.QualikizTransportModel()
        test_model(
            dynamic_runtime_params_slice,
            geo,
            core_profiles,
            pedestal_model.PedestalModelOutput(
                rho_norm_ped_top=np.inf,
                T_i_ped=0.0,
                T_e_ped=0.0,
                n_e_ped=0.0,
                rho_norm_ped_top_idx=geo.torax_mesh.nx,
            ),
        )


if __name__ == '__main__':
  absltest.main()

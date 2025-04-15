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
from torax.config import build_runtime_params
from torax.core_profiles import initialization
from torax.fvm import cell_variable
from torax.pedestal_model import pedestal_model
from torax.sources import source_models as source_models_lib
from torax.torax_pydantic import model_config


# pylint: disable=g-import-not-at-top
try:
  from torax.transport_model import qualikiz_transport_model

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
    torax_config = model_config.ToraxConfig.from_dict(
        dict(
            runtime_params=dict(),
            geometry=dict(geometry_type='circular'),
            pedestal=dict(),
            sources=dict(),
            stepper=dict(),
            transport=dict(transport_model='qualikiz'),
            time_step_calculator=dict(),
        )
    )
    source_models = source_models_lib.SourceModels(
        sources=torax_config.sources.source_model_config
    )
    dynamic_runtime_params_slice = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider(
            torax_config.runtime_params,
            torax_mesh=torax_config.geometry.build_provider.torax_mesh,
            transport=torax_config.transport,
            sources=torax_config.sources,
            stepper=torax_config.stepper,
            pedestal=torax_config.pedestal,
        )(
            t=torax_config.numerics.t_initial,
        )
    )
    static_runtime_params_slice = (
        build_runtime_params.build_static_params_from_config(torax_config)
    )
    geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
    core_profiles = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_runtime_params_slice,
        geo=geo,
        source_models=source_models,
    )

    # Mocking the actual call to QuaLiKiz and its results.
    mock_process = mock.Mock()
    mock_process.communicate.return_value = (b'stdout', b'stderr')
    # The first call is expecting a 2D array, the others should be 1D arrays.
    num_data = cell_variable.face_value(core_profiles.ne).shape[0]
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
                rho_norm_ped_top=np.inf, Tiped=0.0, Teped=0.0, neped=0.0,
                rho_norm_ped_top_idx=geo.torax_mesh.nx,
            ),
        )


if __name__ == '__main__':
  absltest.main()

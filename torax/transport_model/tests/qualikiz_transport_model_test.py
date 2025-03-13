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
from torax.config import runtime_params as general_runtime_params
from torax.core_profiles import initialization
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.pedestal_model import pedestal_model
from torax.sources import pydantic_model as sources_pydantic_model
from torax.sources import source_models as source_models_lib
from torax.stepper import pydantic_model as stepper_pydantic_model


# pylint: disable=g-import-not-at-top
try:
  from torax.transport_model import qualikiz_transport_model

  _QUALIKIZ_TRANSPORT_MODEL_AVAILABLE = True
except ImportError:
  _QUALIKIZ_TRANSPORT_MODEL_AVAILABLE = False
# pylint: enable=g-import-not-at-top


class RuntimeParamsTest(absltest.TestCase):

  def test_runtime_params_builds_dynamic_params(self):
    if not _QUALIKIZ_TRANSPORT_MODEL_AVAILABLE:
      self.skipTest('Qualikiz transport model is not available.')
    runtime_params = qualikiz_transport_model.RuntimeParams()
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    provider = runtime_params.make_provider(geo.torax_mesh)
    provider.build_dynamic_params(t=0.0)


class QualikizTransportModelTest(absltest.TestCase):

  def test_call(self):
    """Tests that the model can be called."""
    # Test prerequisites
    if not _QUALIKIZ_TRANSPORT_MODEL_AVAILABLE:
      self.skipTest('Qualikiz transport model is not available.')
    os.environ['TORAX_COMPILATION_ENABLED'] = '0'

    # Building the model inputs.
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    sources = sources_pydantic_model.Sources()
    source_models = source_models_lib.SourceModels(
        sources=sources.source_model_config
    )
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    dynamic_runtime_params_slice = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider(
            runtime_params,
            torax_mesh=geo.torax_mesh,
            transport=qualikiz_transport_model.RuntimeParams(),
            sources=sources,
        )(
            t=runtime_params.numerics.t_initial,
        )
    )
    static_runtime_params_slice = (
        build_runtime_params.build_static_runtime_params_slice(
            runtime_params=runtime_params,
            sources=sources,
            torax_mesh=geo.torax_mesh,
            stepper=stepper_pydantic_model.Stepper(),
        )
    )
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
    num_data = core_profiles.ne.face_value().shape[0]
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
                rho_norm_ped_top=0.0, Tiped=0.0, Teped=0.0, neped=0.0
            ),
        )


if __name__ == '__main__':
  absltest.main()

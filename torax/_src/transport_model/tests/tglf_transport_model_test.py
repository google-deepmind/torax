# Copyright 2026 DeepMind Technologies Limited
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
from absl.testing import parameterized
import jax
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.pedestal_model import pedestal_model_output as pedestal_model_output_lib
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config


class TGLFTransportModelTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('with_jit', True),
      ('without_jit', False),
  )
  def test_call(self, jit: bool):
    """Tests that the model can be called (with entirely mocked TGLF)."""
    config = default_configs.get_default_config_dict()
    config['transport'] = {'model_name': 'tglf', 'tglf_exec_path': '~/tglf'}
    torax_config = model_config.ToraxConfig.from_dict(config)
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    transport_model = torax_config.transport.build_transport_model()
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(
        t=torax_config.numerics.t_initial,
    )
    geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
    core_profiles = initialization.initial_core_profiles(
        runtime_params=runtime_params,
        geo=geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )

    def _mock_subprocess_run(cmd, **kwargs):
      """Write a fake TGLF output file and return a mock subprocess result."""
      del kwargs  # Unused.

      # cmd is [tglf_exec_path, '-n', n_cores_per_process, '-e', run_directory]
      # Extract the run directory from the command
      run_dir = cmd[-1]

      # Populate the run directory with a fake output file.
      os.makedirs(run_dir, exist_ok=True)
      with open(os.path.join(run_dir, 'out.tglf.gbflux'), 'w') as f:
        f.write('\n'.join(['1.0'] * 12))

      # Return a mock subprocess result with fake stdout and stderr.
      result = mock.Mock()
      result.stdout = 'stdout'
      result.stderr = 'stderr'
      return result

    with mock.patch.object(subprocess, 'run', side_effect=_mock_subprocess_run):
      model_call = (
          jax.jit(transport_model.__call__) if jit else transport_model.__call__
      )
      model_call(
          runtime_params,
          geo,
          core_profiles,
          pedestal_model_output_lib.PedestalModelOutput(
              rho_norm_ped_top=np.inf,
              rho_norm_ped_top_idx=geo.torax_mesh.nx,
              T_i_ped=0.0,
              T_e_ped=0.0,
              n_e_ped=0.0,
          ),
      )


if __name__ == '__main__':
  absltest.main()

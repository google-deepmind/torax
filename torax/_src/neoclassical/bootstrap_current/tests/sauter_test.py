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

from absl.testing import absltest
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.neoclassical.bootstrap_current import sauter
from torax._src.torax_pydantic import model_config

_N_RHO = 10


class SauterTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    torax_config = model_config.ToraxConfig.from_dict({
        'profile_conditions': {},
        'numerics': {},
        'plasma_composition': {'Z_eff': 2.0},
        'geometry': {
            'geometry_type': 'circular',
            'n_rho': _N_RHO,
        },
        'transport': {},
        'solver': {},
        'pedestal': {},
        'sources': {},
        'neoclassical': {
            'bootstrap_current': {'model_name': 'sauter'},
        },
    })
    params_provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    self.runtime_params, self.geo = (
        build_runtime_params.get_consistent_runtime_params_and_geometry(
            t=torax_config.numerics.t_initial,
            runtime_params_provider=params_provider,
            geometry_provider=torax_config.geometry.build_provider,
            is_initialization=True,
        )
    )
    self.core_profiles = initialization.initial_core_profiles(
        self.runtime_params,
        self.geo,
        source_models=torax_config.sources.build_models(),
        neoclassical_models=torax_config.neoclassical.build_models(),
    )

  def test_sauter_bootstrap_current_is_correct_shape(self):
    model = sauter.SauterModel()
    result = model.calculate_bootstrap_current(
        self.runtime_params, self.geo, self.core_profiles
    )
    self.assertEqual(result.j_parallel_bootstrap.shape, (_N_RHO,))
    self.assertEqual(result.j_parallel_bootstrap_face.shape, (_N_RHO + 1,))


if __name__ == '__main__':
  absltest.main()

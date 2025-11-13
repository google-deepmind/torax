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
from torax._src.config import build_runtime_params
from torax._src.pedestal_model import no_pedestal
from torax._src.torax_pydantic import model_config


class NoPedestalTest(absltest.TestCase):

  def test_build_and_call_model(self):
    torax_config = model_config.ToraxConfig.from_dict({
        'pedestal': {},
        'transport': {},
        'solver': {},
        'profile_conditions': {},
        'numerics': {},
        'sources': {},
        'geometry': {'geometry_type': 'circular'},
        'plasma_composition': {},
    })
    pedestal_policy = torax_config.pedestal.set_pedestal.build_pedestal_policy()
    no_pedestal_model = no_pedestal.NoPedestal(pedestal_policy=pedestal_policy)
    geo = mock.Mock()
    geo.torax_mesh.nx = 10
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(
        t=torax_config.numerics.t_initial,
    )
    pedestal_policy_state = pedestal_policy.initial_state(
        t=torax_config.numerics.t_initial,
        runtime_params=runtime_params.pedestal_policy,
    )
    result = no_pedestal_model(
        runtime_params, geo, mock.Mock(), pedestal_policy_state
    )
    self.assertEqual(result.rho_norm_ped_top, jnp.inf)
    self.assertEqual(result.rho_norm_ped_top_idx, 10)


if __name__ == '__main__':
  absltest.main()

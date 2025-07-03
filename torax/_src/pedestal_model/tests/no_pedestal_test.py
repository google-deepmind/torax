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
from torax._src.pedestal_model import no_pedestal


class NoPedestalTest(absltest.TestCase):

  def test_build_and_call_model(self):
    no_pedestal_model = no_pedestal.NoPedestal()
    geo = mock.Mock()
    geo.torax_mesh.nx = 10
    dynamic_runtime_params_slice = mock.Mock()
    dynamic_runtime_params_slice.pedestal.set_pedestal = True
    result = no_pedestal_model(dynamic_runtime_params_slice, geo, mock.Mock())
    self.assertEqual(result.rho_norm_ped_top, jnp.inf)
    self.assertEqual(result.rho_norm_ped_top_idx, 10)


if __name__ == '__main__':
  absltest.main()

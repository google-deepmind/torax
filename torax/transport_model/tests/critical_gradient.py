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
"""Tests for critical gradient transport model."""
from absl.testing import absltest
from torax import geometry
from torax.transport_model import critical_gradient


class RuntimeParamsTest(absltest.TestCase):

  def test_runtime_params_builds_dynamic_params(self):
    runtime_params = critical_gradient.RuntimeParams()
    geo = geometry.build_circular_geometry()
    provider = runtime_params.make_provider(geo.torax_mesh)
    provider.build_dynamic_params(t=0.0)


if __name__ == '__main__':
  absltest.main()
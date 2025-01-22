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
"""Tests for runtime params for sources."""
from absl.testing import absltest
from torax.geometry import circular_geometry
from torax.sources import runtime_params as runtime_params_lib


class RuntimeParamsTest(absltest.TestCase):

  def test_runtime_params_builds_dynamic_params(self):
    runtime_params = runtime_params_lib.RuntimeParams()
    geo = circular_geometry.build_circular_geometry()
    provider = runtime_params.make_provider(geo.torax_mesh)
    dynamic_params = provider.build_dynamic_params(t=0.0)
    self.assertIsInstance(
        dynamic_params, runtime_params_lib.DynamicRuntimeParams
    )


if __name__ == '__main__':
  absltest.main()

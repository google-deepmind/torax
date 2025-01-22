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
"""Tests for qualikiz transport_model transport model."""
from absl.testing import absltest
from torax.geometry import circular_geometry
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
    geo = circular_geometry.build_circular_geometry()
    provider = runtime_params.make_provider(geo.torax_mesh)
    provider.build_dynamic_params(t=0.0)


if __name__ == '__main__':
  absltest.main()

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

"""Unit tests for torax.transport_model.qlknn_wrapper."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from torax import config as config_lib
from torax import config_slice
from torax import geometry
from torax import initial_states
from torax.transport_model import qlknn_wrapper


class QlknnWrapperTest(parameterized.TestCase):
  """Unit tests for the `torax.transport_model.qlknn_wrapper` module."""

  def test_qlknn_wrapper_cache_works(self):
    """Tests that QLKNN calls are properly cached."""
    # This test can uncover and changes to the data structures which break the
    # caching.
    qlknn = qlknn_wrapper.QLKNNTransportModel()
    # Caching only works when compiled.
    qlknn_jitted = jax.jit(qlknn)
    config = config_lib.Config()
    geo = geometry.build_circular_geometry(config)
    dynamic_config_slice = config_slice.build_dynamic_config_slice(config)
    static_config_slice = config_slice.build_static_config_slice(config)
    core_profiles = initial_states.initial_core_profiles(
        dynamic_config_slice, static_config_slice, geo
    )
    qlknn_jitted(dynamic_config_slice, geo, core_profiles)
    # The call should be cached. If there was an error, the cache size would be
    # 0.
    self.assertGreaterEqual(
        qlknn._cached_combined.cache_info().currsize,  # pylint: disable=protected-access
        1,
    )


if __name__ == '__main__':
  absltest.main()

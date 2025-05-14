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
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax.sources import source_profiles as source_profiles_lib


# pylint: disable=invalid-name
class SourceProfilesTest(parameterized.TestCase):

  def test_summed_T_i_profiles_dont_change_when_jitting(self):
    geo = geometry_pydantic_model.CircularConfig().build_geometry()

    # Make some dummy source profiles that could have come from these sources.
    ones = jnp.ones_like(geo.rho)
    profiles = source_profiles_lib.SourceProfiles(
        bootstrap_current=bootstrap_current_base.BootstrapCurrent.zeros(geo),
        qei=source_profiles_lib.QeiInfo.zeros(geo),
        T_i={
            'generic_heat': ones,
            'fusion': ones * 3,
        },
        T_e={
            'generic_heat': ones * 2,
            'fusion': ones * 4,
            'bremsstrahlung': -ones,
            'ohmic': ones * 5,
        },
        n_e={},
        psi={},
    )
    with self.subTest('without_jit'):
      summed_T_i = profiles.total_sources('T_i', geo)
      np.testing.assert_allclose(summed_T_i, ones * 4 * geo.vpr)
      summed_T_e = profiles.total_sources('T_e', geo)
      np.testing.assert_allclose(summed_T_e, ones * 10 * geo.vpr)

    with self.subTest('with_jit'):
      sum_temp = jax.jit(
          profiles.total_sources, static_argnames=('source_type')
      )
      jitted_T_i = sum_temp('T_i', geo)
      np.testing.assert_allclose(jitted_T_i, ones * 4 * geo.vpr)
      jitted_T_e = sum_temp('T_e', geo)
      np.testing.assert_allclose(jitted_T_e, ones * 10 * geo.vpr)


if __name__ == '__main__':
  absltest.main()

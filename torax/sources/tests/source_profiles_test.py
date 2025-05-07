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
from torax.sources import source_profiles as source_profiles_lib


class SourceProfilesTest(parameterized.TestCase):

  def test_summed_temp_ion_profiles_dont_change_when_jitting(self):
    geo = geometry_pydantic_model.CircularConfig().build_geometry()

    # Make some dummy source profiles that could have come from these sources.
    ones = jnp.ones_like(geo.rho)
    profiles = source_profiles_lib.SourceProfiles(
        j_bootstrap=source_profiles_lib.BootstrapCurrentProfile.zero_profile(
            geo
        ),
        qei=source_profiles_lib.QeiInfo.zeros(geo),
        temp_ion={
            'generic_heat': ones,
            'fusion': ones * 3,
        },
        temp_el={
            'generic_heat': ones * 2,
            'fusion': ones * 4,
            'bremsstrahlung': -ones,
            'ohmic': ones * 5,
        },
        n_e={},
        psi={},
    )
    with self.subTest('without_jit'):
      summed_temp_ion = profiles.total_sources('temp_ion', geo)
      np.testing.assert_allclose(summed_temp_ion, ones * 4 * geo.vpr)
      summed_temp_el = profiles.total_sources('temp_el', geo)
      np.testing.assert_allclose(summed_temp_el, ones * 10 * geo.vpr)

    with self.subTest('with_jit'):
      sum_temp = jax.jit(
          profiles.total_sources, static_argnames=('source_type')
      )
      jitted_temp_ion = sum_temp('temp_ion', geo)
      np.testing.assert_allclose(jitted_temp_ion, ones * 4 * geo.vpr)
      jitted_temp_el = sum_temp('temp_el', geo)
      np.testing.assert_allclose(jitted_temp_el, ones * 10 * geo.vpr)


if __name__ == '__main__':
  absltest.main()

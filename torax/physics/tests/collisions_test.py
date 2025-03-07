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
from absl.testing import parameterized
from jax import numpy as jnp
import numpy as np
from torax import state
from torax.fvm import cell_variable
from torax.physics import collisions


# pylint: disable=invalid-name
class CollisionsTest(parameterized.TestCase):

  def test_fast_ion_fractional_heating_formula_ion_heating_limit(self):
    # Inertial energy small compared to critical energy.
    birth_energy = 1e-3
    temp_el = jnp.array(0.1, dtype=jnp.float32)
    fast_ion_mass = 1
    frac_i = collisions.fast_ion_fractional_heating_formula(
        birth_energy, temp_el, fast_ion_mass
    )
    np.testing.assert_allclose(frac_i, 1.0, atol=1e-3)

  def test_fast_ion_fractional_heating_formula_electron_heating_limit(self):
    # Inertial energy large compared to critical energy.
    birth_energy = 1e10
    temp_el = jnp.array(0.1, dtype=jnp.float32)
    fast_ion_mass = 1
    frac_i = collisions.fast_ion_fractional_heating_formula(
        birth_energy, temp_el, fast_ion_mass
    )
    np.testing.assert_allclose(frac_i, 0.0, atol=1e-9)

  # TODO(b/377225415): generalize to arbitrary number of ions.
  @parameterized.parameters([
      dict(Aimp=20.0, Zimp=10.0, Zi=1.0, Ai=1.0, ni=1.0, expected=1.0),
      dict(Aimp=20.0, Zimp=10.0, Zi=1.0, Ai=2.0, ni=1.0, expected=0.5),
      dict(Aimp=20.0, Zimp=10.0, Zi=2.0, Ai=4.0, ni=0.5, expected=0.5),
      dict(Aimp=20.0, Zimp=10.0, Zi=1.0, Ai=2.0, ni=0.9, expected=0.5),
      dict(Aimp=40.0, Zimp=20.0, Zi=1.0, Ai=2.0, ni=0.92, expected=0.5),
  ])
  def test_calculate_weighted_Zeff(self, Aimp, Zimp, Zi, Ai, ni, expected):
    """Compare `_calculate_weighted_Zeff` to a reference value."""
    ne = 1.0
    nimp = (ne - ni * Zi) / Zimp
    core_profiles = mock.create_autospec(
        state.CoreProfiles,
        instance=True,
        ne=cell_variable.CellVariable(
            value=jnp.array(ne),
            dr=jnp.array(1.0),
        ),
        ni=cell_variable.CellVariable(
            value=jnp.array(ni),
            dr=jnp.array(1.0),
        ),
        nimp=cell_variable.CellVariable(
            value=jnp.array(nimp),
            dr=jnp.array(1.0),
        ),
        Zi=Zi,
        Ai=Ai,
        Zimp=Zimp,
        Aimp=Aimp,
    )
    np.testing.assert_allclose(
        collisions._calculate_weighted_Zeff(core_profiles), expected
    )

if __name__ == '__main__':
  absltest.main()

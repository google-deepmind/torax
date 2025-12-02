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
import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax._src import jax_utils
from torax._src import math_utils
from torax._src.fvm import cell_variable
from torax._src.geometry import circular_geometry
from torax._src.physics import formulas
from torax._src.test_utils import core_profile_helpers


# pylint: disable=invalid-name
class FormulasTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    jax_utils.enable_errors(True)
    self.geo = circular_geometry.CircularConfig(
        n_rho=10, a_minor=1.0
    ).build_geometry()
    self.core_profiles = core_profile_helpers.make_zero_core_profiles(self.geo)
    self.core_profiles = dataclasses.replace(
        self.core_profiles,
        T_i=core_profile_helpers.make_constant_core_profile(self.geo, 1.0),
        T_e=core_profile_helpers.make_constant_core_profile(self.geo, 2.0),
        n_e=core_profile_helpers.make_constant_core_profile(self.geo, 3.0e20),
        n_i=core_profile_helpers.make_constant_core_profile(self.geo, 2.5e20),
        n_impurity=core_profile_helpers.make_constant_core_profile(
            self.geo, 0.25e20
        ),
        Ip_profile_face=[np.pi * 1e6],
    )

  # TODO(b/377225415): generalize to arbitrary number of ions.
  @parameterized.parameters([
      dict(Z_i=1.0, Z_impurity=10.0, Z_eff=1.0, expected=1.0),
      dict(Z_i=1.0, Z_impurity=5.0, Z_eff=1.0, expected=1.0),
      dict(Z_i=2.0, Z_impurity=10.0, Z_eff=2.0, expected=0.5),
      dict(Z_i=2.0, Z_impurity=5.0, Z_eff=2.0, expected=0.5),
      dict(Z_i=1.0, Z_impurity=10.0, Z_eff=1.9, expected=0.9),
      dict(Z_i=2.0, Z_impurity=10.0, Z_eff=3.6, expected=0.4),
  ])
  def test_calculate_main_ion_dilution_factor(
      self, Z_i, Z_impurity, Z_eff, expected
  ):
    """Unit test of `calculate_main_ion_dilution_factor`."""
    np.testing.assert_allclose(
        formulas.calculate_main_ion_dilution_factor(Z_i, Z_impurity, Z_eff),
        expected,
    )

  def test_calculate_stored_thermal_energy(self):
    """Test that stored thermal energy is computed correctly."""
    p_el = core_profile_helpers.make_constant_core_profile(self.geo, 1.0)
    p_ion = core_profile_helpers.make_constant_core_profile(self.geo, 2.0)
    p_tot = core_profile_helpers.make_constant_core_profile(self.geo, 3.0)
    wth_el, wth_ion, wth_tot = formulas.calculate_stored_thermal_energy(
        p_el, p_ion, p_tot, self.geo
    )

    volume = math_utils.volume_integration(np.array([1.0]), self.geo)

    np.testing.assert_allclose(wth_el, 1.5 * p_el.value[0] * volume)
    np.testing.assert_allclose(wth_ion, 1.5 * p_ion.value[0] * volume)
    np.testing.assert_allclose(wth_tot, 1.5 * p_tot.value[0] * volume)

  def test_calculate_greenwald_fraction(self):
    """Test that Greenwald fraction is calculated correctly."""
    n_e_avg = 1.0e20

    fgw_n_e_volume_avg_calculated = formulas.calculate_greenwald_fraction(
        n_e_avg, self.core_profiles, self.geo
    )

    fgw_n_e_volume_avg_expected = 1.0

    np.testing.assert_allclose(
        fgw_n_e_volume_avg_calculated, fgw_n_e_volume_avg_expected
    )

  def test_calculate_betas(self):
    """Test that betas are calculated correctly."""

    beta_tor, beta_pol, beta_N = formulas.calculate_betas(
        self.core_profiles, self.geo
    )
    beta_tor_expected = 0.012530022
    beta_pol_expected = 1.3549808
    beta_N_expected = 2.113868
    np.testing.assert_allclose(beta_tor, beta_tor_expected)
    np.testing.assert_allclose(beta_pol, beta_pol_expected)
    np.testing.assert_allclose(beta_N, beta_N_expected)

  def test_calculate_radial_electric_field(self):
    """Test that radial electric field is calculated correctly for constant profiles."""
    core_profiles = core_profile_helpers.make_zero_core_profiles(self.geo)
    core_profiles = dataclasses.replace(
        core_profiles,
        T_i=core_profile_helpers.make_constant_core_profile(self.geo, 1.0),
        T_e=core_profile_helpers.make_constant_core_profile(self.geo, 2.0),
        n_e=core_profile_helpers.make_constant_core_profile(self.geo, 3.0e20),
        n_i=core_profile_helpers.make_constant_core_profile(self.geo, 2.5e20),
        n_impurity=core_profile_helpers.make_constant_core_profile(
            self.geo, 0.25e20
        ),
        psi=core_profile_helpers.make_constant_core_profile(self.geo, 1.0),
        toroidal_velocity=core_profile_helpers.make_constant_core_profile(
            self.geo, 0.0
        ),
        Z_i_face=1.0,
    )

    poloidal_velocity = cell_variable.CellVariable(
        value=np.zeros_like(self.geo.rho_norm), dr=self.geo.drho_norm
    )
    E_r = formulas.calculate_radial_electric_field(
        pressure_thermal_i=core_profiles.pressure_thermal_i,
        psi=core_profiles.psi,
        n_i=core_profiles.n_i,
        toroidal_velocity=core_profiles.toroidal_velocity,
        poloidal_velocity=poloidal_velocity,
        Z_i_face=core_profiles.Z_i_face,
        geo=self.geo,
    )

    # For constant profiles and zero velocities, E_r should be zero.
    np.testing.assert_allclose(
        E_r.value, np.zeros_like(self.geo.rho_norm), atol=1e-12
    )


if __name__ == '__main__':
  absltest.main()

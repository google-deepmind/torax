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
import jax
import numpy as np
from torax import constants
from torax import jax_utils
from torax import state
from torax.fvm import cell_variable
from torax.geometry import geometry
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.physics import formulas


# pylint: disable=invalid-name
class FormulasTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    jax_utils.enable_errors(True)
    self.geo = geometry_pydantic_model.CircularConfig(n_rho=4).build_geometry()

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

  def test_calculate_pressure(self):
    """Test that pressure is computed correctly."""

    def _make_constant_core_profile(
        value: float,
    ) -> cell_variable.CellVariable:
      return cell_variable.CellVariable(
          value=value * np.ones_like(self.geo.rho_norm),
          left_face_grad_constraint=np.zeros(()),
          left_face_constraint=None,
          right_face_grad_constraint=None,
          right_face_constraint=jax.numpy.array(value),
          dr=self.geo.drho_norm,
      )

    core_profiles = mock.create_autospec(
        state.CoreProfiles,
        instance=True,
        density_reference=1e20,
        T_i=_make_constant_core_profile(1.0),
        T_e=_make_constant_core_profile(2.0),
        n_e=_make_constant_core_profile(3.0),
        n_i=_make_constant_core_profile(2.5),
        n_impurity=_make_constant_core_profile(0.25),
    )

    p_el, p_ion, p_tot = formulas.calculate_pressure(core_profiles)
    # Make sure that we are grabbing the values from the face grid.
    self.assertEqual(p_el.shape, self.geo.rho_face.shape)
    # Ignore boundary condition terms and just check formula sanity.
    np.testing.assert_allclose(
        p_el, 6 * constants.CONSTANTS.keV2J * core_profiles.density_reference
    )
    np.testing.assert_allclose(
        p_ion,
        2.75 * constants.CONSTANTS.keV2J * core_profiles.density_reference,
    )
    np.testing.assert_allclose(
        p_tot,
        8.75 * constants.CONSTANTS.keV2J * core_profiles.density_reference,
    )

  def test_calculate_stored_thermal_energy(self):
    """Test that stored thermal energy is computed correctly."""
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    p_el = np.ones_like(geo.rho_face)
    p_ion = 2 * np.ones_like(geo.rho_face)
    p_tot = p_el + p_ion
    wth_el, wth_ion, wth_tot = formulas.calculate_stored_thermal_energy(
        p_el, p_ion, p_tot, geo
    )

    volume = np.trapezoid(geo.vpr_face, geo.rho_face_norm)

    np.testing.assert_allclose(wth_el, 1.5 * p_el[0] * volume)
    np.testing.assert_allclose(wth_ion, 1.5 * p_ion[0] * volume)
    np.testing.assert_allclose(wth_tot, 1.5 * p_tot[0] * volume)

  def test_calculate_greenwald_fraction(self):
    """Test that Greenwald fraction is calculated correctly."""
    n_e = 1.0

    core_profiles = mock.create_autospec(
        state.CoreProfiles,
        instance=True,
        density_reference=1e20,
        Ip_profile_face=[np.pi * 1e6],
    )
    geo = mock.create_autospec(
        geometry.Geometry,
        instance=True,
        a_minor=1.0,
    )

    fgw_n_e_volume_avg_calculated = formulas.calculate_greenwald_fraction(
        n_e, core_profiles, geo
    )

    fgw_n_e_volume_avg_expected = 1.0

    np.testing.assert_allclose(
        fgw_n_e_volume_avg_calculated, fgw_n_e_volume_avg_expected
    )


if __name__ == '__main__':
  absltest.main()

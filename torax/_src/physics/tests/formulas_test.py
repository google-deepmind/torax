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
from torax._src import constants
from torax._src import jax_utils
from torax._src import math_utils
from torax._src import state
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.geometry import pydantic_model as geometry_pydantic_model
from torax._src.physics import formulas


# pylint: disable=invalid-name
class FormulasTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    jax_utils.enable_errors(True)
    self.geo = geometry_pydantic_model.CircularConfig(n_rho=10).build_geometry()

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

    core_profiles = mock.create_autospec(
        state.CoreProfiles,
        instance=True,
        T_i=_make_constant_core_profile(self.geo, 1.0),
        T_e=_make_constant_core_profile(self.geo, 2.0),
        n_e=_make_constant_core_profile(self.geo, 3.0),
        n_i=_make_constant_core_profile(self.geo, 2.5),
        n_impurity=_make_constant_core_profile(self.geo, 0.25),
    )

    p_el, p_ion, p_tot = formulas.calculate_pressure(core_profiles)

    # Ignore boundary condition terms and just check formula sanity.
    np.testing.assert_allclose(p_el.value, 6 * constants.CONSTANTS.keV2J)
    np.testing.assert_allclose(p_ion.value, 2.75 * constants.CONSTANTS.keV2J)
    np.testing.assert_allclose(p_tot.value, 8.75 * constants.CONSTANTS.keV2J)

  def test_calculate_stored_thermal_energy(self):
    """Test that stored thermal energy is computed correctly."""
    p_el = _make_constant_core_profile(self.geo, 1.0)
    p_ion = _make_constant_core_profile(self.geo, 2.0)
    p_tot = _make_constant_core_profile(self.geo, 3.0)
    wth_el, wth_ion, wth_tot = formulas.calculate_stored_thermal_energy(
        p_el, p_ion, p_tot, self.geo
    )

    volume = math_utils.volume_integration(1.0, self.geo)

    np.testing.assert_allclose(wth_el, 1.5 * p_el.value[0] * volume)
    np.testing.assert_allclose(wth_ion, 1.5 * p_ion.value[0] * volume)
    np.testing.assert_allclose(wth_tot, 1.5 * p_tot.value[0] * volume)

  def test_calculate_greenwald_fraction(self):
    """Test that Greenwald fraction is calculated correctly."""
    n_e = 1.0e20

    core_profiles = mock.create_autospec(
        state.CoreProfiles,
        instance=True,
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


def _make_constant_core_profile(
    geo: geometry.Geometry,
    value: float,
) -> cell_variable.CellVariable:
  return cell_variable.CellVariable(
      value=value * np.ones_like(geo.rho_norm),
      left_face_grad_constraint=np.zeros(()),
      left_face_constraint=None,
      right_face_grad_constraint=None,
      right_face_constraint=jax.numpy.array(value),
      dr=geo.drho_norm,
  )


if __name__ == '__main__':
  absltest.main()

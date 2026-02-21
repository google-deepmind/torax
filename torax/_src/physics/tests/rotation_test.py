# Copyright 2025 DeepMind Technologies Limited
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
"""Tests for the rotation module."""

from absl.testing import absltest
import numpy as np
from torax._src.fvm import cell_variable
from torax._src.geometry import circular_geometry
from torax._src.physics import rotation
from torax._src.test_utils import core_profile_helpers


# pylint: disable=invalid-name
class RotationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.geo = circular_geometry.CircularConfig(
        n_rho=10, a_minor=1.0
    ).build_geometry()

  def test_calculate_rotation_shapes_are_correct(self):
    rotation_output = rotation.calculate_rotation(
        T_i=core_profile_helpers.make_constant_core_profile(
            geo=self.geo, value=1.0
        ),
        psi=core_profile_helpers.make_constant_core_profile(
            geo=self.geo, value=1.0
        ),
        n_i=core_profile_helpers.make_constant_core_profile(self.geo, 1.0),
        q_face=np.ones_like(self.geo.rho_face_norm),
        Z_eff_face=np.ones_like(self.geo.rho_face_norm),
        Z_i_face=np.ones_like(self.geo.rho_face_norm),
        toroidal_angular_velocity=core_profile_helpers.make_constant_core_profile(
            self.geo, 0.0
        ),
        pressure_total_i=core_profile_helpers.make_constant_core_profile(
            geo=self.geo, value=1.0
        ),
        geo=self.geo,
    )
    self.assertEqual(rotation_output.v_ExB.shape, self.geo.rho_face_norm.shape)
    self.assertEqual(
        rotation_output.Er.face_value().shape, self.geo.rho_face_norm.shape
    )
    self.assertEqual(
        rotation_output.poloidal_velocity.face_value().shape,
        self.geo.rho_face_norm.shape,
    )

  def test_electric_field_is_zero_for_zero_velocities_and_constant_pressure(
      self,
  ):
    E_r, _, _ = rotation._calculate_radial_electric_field(
        pressure_total_i=core_profile_helpers.make_constant_core_profile(
            geo=self.geo, value=1.0
        ),
        toroidal_angular_velocity=core_profile_helpers.make_constant_core_profile(
            self.geo, 0.0
        ),
        poloidal_velocity=core_profile_helpers.make_constant_core_profile(
            self.geo, 0.0
        ),
        n_i=core_profile_helpers.make_constant_core_profile(self.geo, 1e19),
        Z_i_face=1.0,
        B_pol_face=np.ones_like(self.geo.rho_face_norm),
        B_tor_face=np.ones_like(self.geo.rho_face_norm),
        geo=self.geo,
    )
    np.testing.assert_allclose(
        E_r.value, np.zeros_like(self.geo.rho_norm), atol=1e-12
    )

  def test_electric_field_is_not_zero_for_toroidal_velocity(self):
    E_r, _, _ = rotation._calculate_radial_electric_field(
        pressure_total_i=core_profile_helpers.make_constant_core_profile(
            geo=self.geo, value=1.0
        ),
        toroidal_angular_velocity=core_profile_helpers.make_constant_core_profile(
            self.geo, 1.0
        ),
        poloidal_velocity=core_profile_helpers.make_constant_core_profile(
            self.geo, 0.0
        ),
        n_i=core_profile_helpers.make_constant_core_profile(self.geo, 1.0),
        Z_i_face=1.0,
        B_pol_face=np.ones_like(self.geo.rho_face_norm),
        B_tor_face=np.ones_like(self.geo.rho_face_norm),
        geo=self.geo,
    )
    self.assertTrue(np.any(np.abs(E_r.value) > 1e-12))

  def test_electric_field_is_not_zero_for_poloidal_velocity(self):
    """Test that radial electric field is not zero for non-zero poloidal velocity."""
    E_r, _, _ = rotation._calculate_radial_electric_field(
        pressure_total_i=core_profile_helpers.make_constant_core_profile(
            geo=self.geo, value=1.0
        ),
        toroidal_angular_velocity=core_profile_helpers.make_constant_core_profile(
            self.geo, 0.0
        ),
        poloidal_velocity=core_profile_helpers.make_constant_core_profile(
            self.geo, 1.0
        ),
        n_i=core_profile_helpers.make_constant_core_profile(self.geo, 1.0),
        Z_i_face=1.0,
        B_pol_face=np.ones_like(self.geo.rho_face_norm),
        B_tor_face=np.ones_like(self.geo.rho_face_norm),
        geo=self.geo,
    )
    self.assertTrue(np.any(np.abs(E_r.value) > 1e-12))

  def test_electric_field_is_not_zero_for_non_constant_pressure(self):
    """Test that radial electric field is not zero for non-zero poloidal velocity."""
    E_r, _, _ = rotation._calculate_radial_electric_field(
        pressure_total_i=cell_variable.CellVariable(
            value=np.linspace(1.0, 2.0, self.geo.rho_norm.size),
            face_centers=self.geo.rho_face_norm,
        ),
        toroidal_angular_velocity=core_profile_helpers.make_constant_core_profile(
            self.geo, 0.0
        ),
        poloidal_velocity=core_profile_helpers.make_constant_core_profile(
            self.geo, 0.0
        ),
        n_i=core_profile_helpers.make_constant_core_profile(self.geo, 1.0),
        Z_i_face=1.0,
        B_pol_face=np.ones_like(self.geo.rho_face_norm),
        B_tor_face=np.ones_like(self.geo.rho_face_norm),
        geo=self.geo,
    )
    self.assertTrue(np.any(np.abs(E_r.value) > 1e-12))

  def test_v_ExB_is_zero_for_zero_electric_field(self):
    """Test that v_ExB is zero for zero electric field."""
    v_ExB = rotation._calculate_v_ExB(
        Er_face=np.zeros_like(self.geo.rho_face_norm),
        B_total_face=np.ones_like(self.geo.rho_face_norm),
    )
    # v_ExB should be zero when Er is zero.
    np.testing.assert_allclose(
        v_ExB, np.zeros_like(self.geo.rho_face_norm), atol=1e-12
    )

  def test_v_ExB_is_not_zero_for_non_zero_electric_field(self):
    """Test that v_ExB is not zero for non-zero electric field."""
    v_ExB = rotation._calculate_v_ExB(
        Er_face=np.ones_like(self.geo.rho_face_norm),
        B_total_face=np.ones_like(self.geo.rho_face_norm),
    )
    # v_ExB should not be zero when Er is non-zero.
    self.assertTrue(np.any(np.abs(v_ExB) > 1e-12))


if __name__ == "__main__":
  absltest.main()

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
import os

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from torax._src.geometry import fbt
from torax._src.geometry import geometry
from torax._src.geometry import get_example_L_LY_data
from torax._src.geometry import pydantic_model as geometry_pydantic_model
from torax._src.geometry import standard_geometry
from torax._src.torax_pydantic import interpolated_param_2d

# pylint: disable=invalid-name


class GeometryTest(parameterized.TestCase):
  """Unit tests for the `geometry` module."""

  def test_standard_geometry_can_be_input_to_jitted_function(self):

    @jax.jit
    def foo(geo: geometry.Geometry):
      return geo.R_major

    intermediate = standard_geometry.StandardGeometryIntermediates(
        geometry_type=geometry.GeometryType.FBT,
        Ip_from_parameters=True,
        R_major=6.2,
        a_minor=2.0,
        B_0=5.3,
        # Use the same dummy value for the rest.
        psi=np.arange(0, 1.0, 0.01),
        Ip_profile=np.arange(0, 1.0, 0.01),
        Phi=np.arange(0, 1.0, 0.01),
        R_in=np.arange(1, 2, 0.01),
        R_out=np.arange(1, 2, 0.01),
        F=np.arange(0, 1.0, 0.01),
        int_dl_over_Bp=np.arange(0, 1.0, 0.01),
        flux_surf_avg_1_over_R=np.arange(0, 1.0, 0.01),
        flux_surf_avg_1_over_R2=np.arange(0, 1.0, 0.01),
        flux_surf_avg_grad_psi2=np.arange(0, 1.0, 0.01),
        flux_surf_avg_grad_psi=np.arange(0, 1.0, 0.01),
        flux_surf_avg_grad_psi2_over_R2=np.arange(0, 1.0, 0.01),
        flux_surf_avg_B2=np.arange(0, 1.0, 0.01),
        flux_surf_avg_1_over_B2=np.arange(0, 1.0, 0.01),
        delta_upper_face=np.arange(0, 1.0, 0.01),
        delta_lower_face=np.arange(0, 1.0, 0.01),
        elongation=np.arange(0, 1.0, 0.01),
        vpr=np.arange(0, 1.0, 0.01),
        face_centers=interpolated_param_2d.get_face_centers(25),
        hires_factor=4,
        z_magnetic_axis=np.array(0.0),
        diverted=None,
        connection_length_target=None,
        connection_length_divertor=None,
        angle_of_incidence_target=None,
        R_OMP=None,
        R_target=None,
        B_pol_OMP=None,
    )
    geo = standard_geometry.build_standard_geometry(intermediate)
    foo(geo)

  def test_stack_geometries_standard_geometries(self):
    """Test stack_geometries for standard geometries."""
    # Create a few different geometries
    L, LY0 = get_example_L_LY_data.get_example_L_LY_data(10, 1, fill_value=1.0)
    _, LY1 = get_example_L_LY_data.get_example_L_LY_data(10, 1, fill_value=2.0)
    _, LY2 = get_example_L_LY_data.get_example_L_LY_data(10, 1, fill_value=3.0)
    geo0_intermediate = fbt._from_fbt_single_slice(
        geometry_directory=None,
        LY_object=LY0,
        L_object=L,
        face_centers=interpolated_param_2d.get_face_centers(25),
    )

    geo1_intermediate = fbt._from_fbt_single_slice(
        geometry_directory=None,
        LY_object=LY1,
        L_object=L,
        face_centers=interpolated_param_2d.get_face_centers(25),
    )
    geo2_intermediate = fbt._from_fbt_single_slice(
        geometry_directory=None,
        LY_object=LY2,
        L_object=L,
        face_centers=interpolated_param_2d.get_face_centers(25),
    )
    geo0 = standard_geometry.build_standard_geometry(geo0_intermediate)
    geo1 = standard_geometry.build_standard_geometry(geo1_intermediate)
    geo2 = standard_geometry.build_standard_geometry(geo2_intermediate)

    # Stack them
    stacked_geo = geometry.stack_geometries([geo0, geo1, geo2])

    # Check that the stacked geometry has the correct type and mesh
    self.assertEqual(stacked_geo.geometry_type, geo0.geometry_type)
    self.assertEqual(stacked_geo.torax_mesh, geo0.torax_mesh)

    # Check some specific stacked values
    np.testing.assert_allclose(
        stacked_geo.g2[:, -1],
        np.array([geo0.g2[-1], geo1.g2[-1], geo2.g2[-1]]),
    )
    np.testing.assert_allclose(
        stacked_geo.Ip_profile_face[:, -1],
        np.array([
            geo0.Ip_profile_face[-1],
            geo1.Ip_profile_face[-1],
            geo2.Ip_profile_face[-1],
        ]),
    )

    # Check stacking of derived properties
    np.testing.assert_allclose(
        stacked_geo.rho_b, np.array([geo0.rho_b, geo1.rho_b, geo2.rho_b])
    )

    # Check a property that depends on a stacked property (rho depends on rho_b)
    # Note that rho_norm is the same for all geometries.
    np.testing.assert_allclose(
        stacked_geo.rho,
        np.array([
            geo0.rho_norm * geo0.rho_b,
            geo0.rho_norm * geo1.rho_b,
            geo0.rho_norm * geo2.rho_b,
        ]),
    )

    # Check properties with special handling for on-axis values.
    np.testing.assert_allclose(
        stacked_geo.g0_over_vpr_face[:, 0], 1 / stacked_geo.rho_b
    )
    np.testing.assert_allclose(
        stacked_geo.g1_over_vpr2_face[:, 0], 1 / stacked_geo.rho_b**2
    )

  def _make_intermediates(self, **overrides):
    defaults = dict(
        geometry_type=geometry.GeometryType.FBT,
        Ip_from_parameters=True,
        R_major=6.2,
        a_minor=2.0,
        B_0=5.3,
        psi=np.linspace(0, 1.0, 100),
        Ip_profile=np.linspace(0, 1e6, 100),
        Phi=np.linspace(0, 1.0, 100),
        R_in=np.linspace(4.0, 4.2, 100),
        R_out=np.linspace(8.0, 8.4, 100),
        F=np.linspace(30.0, 33.0, 100),
        int_dl_over_Bp=np.linspace(0.01, 1.0, 100),
        flux_surf_avg_1_over_R=np.linspace(0.1, 0.2, 100),
        flux_surf_avg_1_over_R2=np.linspace(0.01, 0.04, 100),
        flux_surf_avg_grad_psi2=np.linspace(0.01, 1.0, 100),
        flux_surf_avg_grad_psi=np.linspace(0.01, 1.0, 100),
        flux_surf_avg_grad_psi2_over_R2=np.linspace(0.01, 1.0, 100),
        flux_surf_avg_B2=np.linspace(25.0, 30.0, 100),
        flux_surf_avg_1_over_B2=np.linspace(0.03, 0.04, 100),
        delta_upper_face=np.linspace(0.0, 0.3, 100),
        delta_lower_face=np.linspace(0.0, 0.3, 100),
        elongation=np.linspace(1.0, 1.7, 100),
        vpr=np.linspace(0.01, 1.0, 100),
        face_centers=interpolated_param_2d.get_face_centers(25),
        hires_factor=4,
        z_magnetic_axis=np.array(0.0),
        diverted=None,
        connection_length_target=None,
        connection_length_divertor=None,
        angle_of_incidence_target=None,
        R_OMP=None,
        R_target=None,
        B_pol_OMP=None,
    )
    defaults.update(overrides)
    return standard_geometry.StandardGeometryIntermediates(**defaults)

  def test_post_init_flips_psi_when_decreasing(self):
    psi_decreasing = np.linspace(1.0, 0.0, 100)
    intermediates = self._make_intermediates(psi=psi_decreasing)
    self.assertGreater(intermediates.psi[-1], intermediates.psi[0])

  def test_post_init_preserves_psi_when_increasing(self):
    psi_increasing = np.linspace(0.0, 1.0, 100)
    intermediates = self._make_intermediates(psi=psi_increasing)
    np.testing.assert_array_equal(intermediates.psi, psi_increasing)

  def test_post_init_flips_negative_Ip(self):
    Ip_negative = np.linspace(0, -1e6, 100)
    intermediates = self._make_intermediates(Ip_profile=Ip_negative)
    self.assertGreater(intermediates.Ip_profile[-1], 0)

  def test_post_init_preserves_positive_Ip(self):
    Ip_positive = np.linspace(0, 1e6, 100)
    intermediates = self._make_intermediates(Ip_profile=Ip_positive)
    np.testing.assert_array_equal(intermediates.Ip_profile, Ip_positive)

  def test_post_init_enforces_positive_definite_quantities(self):
    intermediates = self._make_intermediates(
        Phi=-np.linspace(0, 1.0, 100),
        F=-np.linspace(30.0, 33.0, 100),
        int_dl_over_Bp=-np.linspace(0.01, 1.0, 100),
        vpr=-np.linspace(0.01, 1.0, 100),
        flux_surf_avg_grad_psi=-np.linspace(0.01, 1.0, 100),
        flux_surf_avg_grad_psi2=-np.linspace(0.01, 1.0, 100),
        flux_surf_avg_grad_psi2_over_R2=-np.linspace(0.01, 1.0, 100),
    )
    np.testing.assert_array_less(-1e-15, intermediates.Phi)
    np.testing.assert_array_less(0, intermediates.F)
    np.testing.assert_array_less(0, intermediates.int_dl_over_Bp)
    np.testing.assert_array_less(0, intermediates.vpr)
    np.testing.assert_array_less(0, intermediates.flux_surf_avg_grad_psi)
    np.testing.assert_array_less(0, intermediates.flux_surf_avg_grad_psi2)
    np.testing.assert_array_less(
        0, intermediates.flux_surf_avg_grad_psi2_over_R2
    )

  @parameterized.named_parameters(
      ('circular', {'geometry_type': 'circular'}),
      (
          'chease',
          {
              'geometry_type': 'chease',
              'geometry_file': 'iterhybrid.mat2cols',
          },
      ),
      (
          'eqdsk',
          {
              'geometry_type': 'eqdsk',
              'geometry_file': 'iterhybrid_cocos02.eqdsk',
              'cocos': 2,
          },
      ),
      (
          'imas',
          {
              'geometry_type': 'imas',
              'imas_filepath': 'ITERhybrid_COCOS17_IDS_ddv4.nc',
          },
      ),
  )
  def test_g0_g1_cauchy_schwarz_consistency(self, geo_config):
    """Tests that g0^2 / g1 is in a physically reasonable range."""
    geo_provider = geometry_pydantic_model.Geometry.from_dict(
        geo_config
    ).build_provider
    geo = geo_provider(0.0)

    # By Cauchy-Schwarz, <|nabla V|>^2 <= <(nabla V)^2>, so g0^2 / g1
    # must be <= 1. For a shaped tokamak this ratio is typically 0.6-1.0.
    ratio = geo.g0_face[1:] ** 2 / geo.g1_face[1:]
    trimmed = ratio[3:]
    np.testing.assert_array_less(trimmed, 1.0 + 1e-10)
    np.testing.assert_array_less(0.5, trimmed)


if __name__ == '__main__':
  absltest.main()

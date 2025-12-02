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
import numpy as np
from torax._src.geometry import fbt
from torax._src.geometry import geometry
from torax._src.geometry import get_example_L_LY_data
from torax._src.geometry import standard_geometry

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
        n_rho=25,
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
        geometry_directory=None, LY_object=LY0, L_object=L
    )

    geo1_intermediate = fbt._from_fbt_single_slice(
        geometry_directory=None, LY_object=LY1, L_object=L
    )
    geo2_intermediate = fbt._from_fbt_single_slice(
        geometry_directory=None, LY_object=LY2, L_object=L
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


if __name__ == '__main__':
  absltest.main()

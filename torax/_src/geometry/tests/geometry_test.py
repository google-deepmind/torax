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
import jax
from jax import numpy as jnp
import numpy as np
from torax._src.geometry import circular_geometry
from torax._src.geometry import geometry
from torax._src.output_tools import output_grid_context
from torax._src.output_tools import output_keys


class GeometryTest(parameterized.TestCase):
  """Unit tests for the `geometry` module."""

  @parameterized.parameters([
      dict(n_rho=25, seed=20220930),
  ])
  def test_face_to_cell(self, n_rho, seed):
    """Compare `face_to_cell` to a PINT reference."""

    rng_state = jax.random.PRNGKey(seed)
    del seed  # Make sure seed isn't accidentally re-used

    # Generate face variables
    rng_use, _ = jax.random.split(rng_state)
    face = jax.random.normal(rng_use, (n_rho + 1,))
    del rng_use  # Make sure rng_use isn't accidentally re-used

    # Convert face values to cell values using jax code being tested
    cell_jax = geometry.face_to_cell(jnp.array(face))

    # Make ground truth
    cell_np = _pint_face_to_cell(n_rho, face)

    np.testing.assert_allclose(cell_jax, cell_np)

  def test_none_z_magnetic_axis_raises_an_error(self):
    geo = circular_geometry.CircularConfig().build_geometry()
    geo = dataclasses.replace(geo, _z_magnetic_axis=None)

    with self.subTest('non_jitted_function'):
      with self.assertRaisesRegex(
          ValueError, 'Geometry does not have a z magnetic axis.'
      ):
        geo.z_magnetic_axis()

    with self.subTest('jitted_function'):
      with self.assertRaisesRegex(
          ValueError, 'Geometry does not have a z magnetic axis.'
      ):
        foo = jax.jit(geo.z_magnetic_axis)
        _ = foo()

  def test_stack_geometries_circular_geometries(self):
    """Test stack_geometries for circular geometries."""
    # Create a few different geometries
    geo0 = circular_geometry.CircularConfig(
        a_minor=0.5, R_major=1.0, B_0=2.0, n_rho=10
    ).build_geometry()
    geo1 = circular_geometry.CircularConfig(
        a_minor=0.5, R_major=1.5, B_0=2.5, n_rho=10
    ).build_geometry()
    geo2 = circular_geometry.CircularConfig(
        a_minor=0.5, R_major=2.0, B_0=3.0, n_rho=10
    ).build_geometry()

    # Stack them
    stacked_geo = geometry.stack_geometries([geo0, geo1, geo2])

    # Check that the stacked geometry has the correct type and mesh
    self.assertEqual(stacked_geo.geometry_type, geo0.geometry_type)
    self.assertEqual(stacked_geo.torax_mesh, geo0.torax_mesh)

    # Check some specific stacked values
    np.testing.assert_allclose(stacked_geo.R_major, np.array([1.0, 1.5, 2.0]))
    np.testing.assert_allclose(stacked_geo.B_0, np.array([2.0, 2.5, 3.0]))
    np.testing.assert_allclose(
        stacked_geo.Phi_face[:, -1],
        np.array([geo0.Phi_face[-1], geo1.Phi_face[-1], geo2.Phi_face[-1]]),
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

  def test_stack_geometries_error_handling_empty_list(self):
    """Test stacking with an empty list (should raise ValueError)."""
    with self.assertRaisesRegex(ValueError, 'No geometries provided.'):
      geometry.stack_geometries([])

  def test_stack_geometries_error_handling_different_mesh_sizes(self):
    """Test error handling for stack_geometries with different mesh sizes."""
    geo0 = circular_geometry.CircularConfig(
        a_minor=0.5, R_major=1.0, B_0=2.0, n_rho=10
    ).build_geometry()
    geo_diff_mesh = circular_geometry.CircularConfig(
        a_minor=0.5, R_major=1.0, B_0=2.0, n_rho=20
    ).build_geometry()  # Different n_rho
    with self.assertRaisesRegex(
        ValueError, 'All geometries must have the same mesh.'
    ):
      geometry.stack_geometries([geo0, geo_diff_mesh])

  def test_stack_geometries_error_handling_different_geometry_types(self):
    """Test different geometry type error handling for stack_geometries."""
    geo0 = circular_geometry.CircularConfig(
        a_minor=0.5, R_major=1.0, B_0=2.0, n_rho=10
    ).build_geometry()
    geo_diff_geometry_type = dataclasses.replace(
        geo0, geometry_type=geometry.GeometryType(3)
    )
    with self.assertRaisesRegex(
        ValueError, 'All geometries must have the same geometry type'
    ):
      geometry.stack_geometries([geo0, geo_diff_geometry_type])

  def test_update_phibdot(self):
    """Test update_phibdot for circular geometries."""
    geo = circular_geometry.CircularConfig().build_geometry()
    geo0 = dataclasses.replace(geo, Phi_face=np.array([1.0]))
    geo1 = dataclasses.replace(geo, Phi_face=np.array([2.0]))
    geo0_updated, geo1_updated = geometry.update_geometries_with_Phibdot(
        dt=0.1, geo_t=geo0, geo_t_plus_dt=geo1
    )
    np.testing.assert_allclose(geo0_updated.Phi_b_dot, 10.0)
    np.testing.assert_allclose(geo1_updated.Phi_b_dot, 10.0)

  def test_geometry_eq(self):
    geo1 = circular_geometry.CircularConfig().build_geometry()
    geo2 = circular_geometry.CircularConfig().build_geometry()
    with self.subTest('same_geometries_are_equal'):
      self.assertEqual(geo1, geo2)

    with self.subTest('different_geometries_are_not_equal'):
      geo3 = dataclasses.replace(geo1, Phi_face=np.array([2.0]))
      self.assertNotEqual(geo1, geo3)

  @parameterized.parameters([
      dict(
          arr=np.array([1.0, 2.0, 3.0]),
          factor=2,
          expected_arr=np.array([1.0, 1.5, 2.0, 2.5, 3.0]),
      ),
      dict(
          arr=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
          factor=2,
          expected_arr=np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
      ),
      dict(
          arr=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
          factor=3,
          expected_arr=np.array(
              [1.0, 1.33333333, 1.66666667, 2.0, 2.33333333, 2.66666667, 3.0,
               3.33333333, 3.66666667, 4.0, 4.33333333, 4.66666667, 5.0]
          ),
      ),
      dict(
          arr=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
          factor=4,
          expected_arr=np.array(
              [0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5,
               0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0]
          ),
      ),
  ])
  def test_increase_resolution(self, arr, factor, expected_arr):
    np.testing.assert_allclose(
        geometry.increase_grid_resolution(arr, factor), expected_arr
    )

  def test_to_output_dict(self):
    geo0 = circular_geometry.CircularConfig(
        a_minor=0.5, R_major=1.0, B_0=2.0, n_rho=10
    ).build_geometry()
    geo1 = circular_geometry.CircularConfig(
        a_minor=0.5, R_major=1.5, B_0=2.5, n_rho=10
    ).build_geometry()
    stacked_geo = geometry.stack_geometries([geo0, geo1])

    times = np.array([0.0, 1.0])
    context = output_grid_context.OutputGridContext(
        times=times,
        rho_face_norm=stacked_geo.rho_face_norm,
        rho_cell_norm=stacked_geo.rho_norm,
        rho_cell_plus_boundaries_norm=np.concatenate(
            [[0.0], stacked_geo.rho_norm, [1.0]]
        ),
    )

    out = stacked_geo.to_output_dict(context)

    # Extended cell+boundaries profiles (time, rho_norm)
    extended_profile_keys = [
        output_keys.PHI,
        output_keys.VOLUME,
        output_keys.AREA,
        output_keys.VPR,
        output_keys.SPR,
        output_keys.ELONGATION,
        output_keys.G0,
        output_keys.G1,
        output_keys.G2,
        output_keys.G3,
        output_keys.GM4,
        output_keys.GM5,
        output_keys.G2G3_OVER_RHON,
        output_keys.TFF,
        output_keys.R_IN,
        output_keys.R_OUT,
        output_keys.R_MID,
        output_keys.EPSILON,
        output_keys.G1_OVER_VPR,
        output_keys.G1_OVER_VPR2,
        output_keys.GM9,
        output_keys.R_MAJOR_PROFILE,
    ]
    for key in extended_profile_keys:
      self.assertIn(key, out)
      dims, arr, _ = out[key]
      self.assertEqual(dims, (output_keys.TIME, output_keys.RHO_NORM))
      self.assertEqual(arr.shape, (2, 12))  # 10 cells + 2 boundaries = 12

    # Face profiles (time, rho_face_norm)
    face_profile_keys = [
        output_keys.DELTA,
        output_keys.TRAPPED_FRACTION,
        output_keys.G0_OVER_VPR,
    ]
    for key in face_profile_keys:
      self.assertIn(key, out)
      dims, arr, _ = out[key]
      self.assertEqual(dims, (output_keys.TIME, output_keys.RHO_FACE_NORM))
      self.assertEqual(arr.shape, (2, 11))  # 10 cells + 1 = 11 faces

    # Scalars (time,)
    scalar_keys = [
        output_keys.R_MAJOR,
        output_keys.A_MINOR,
        output_keys.B_0,
        output_keys.PHI_B_DOT,
        output_keys.RHO_B,
        output_keys.PHI_B,
    ]
    for key in scalar_keys:
      self.assertIn(key, out)
      dims, arr, _ = out[key]
      self.assertEqual(dims, (output_keys.TIME,))
      self.assertEqual(arr.shape, (2,))

    # z_magnetic_axis present vs None
    self.assertIn(output_keys.Z_MAGNETIC_AXIS, out)
    dims, _, _ = out[output_keys.Z_MAGNETIC_AXIS]
    self.assertEqual(dims, (output_keys.TIME,))

    # Test with z_magnetic_axis = None
    geo_none_z = dataclasses.replace(stacked_geo, _z_magnetic_axis=None)
    out_none_z = geo_none_z.to_output_dict(context)
    self.assertNotIn(output_keys.Z_MAGNETIC_AXIS, out_none_z)


def _pint_face_to_cell(n_rho, face):
  cell = np.zeros(n_rho)
  cell[:] = 0.5 * (face[1:] + face[:-1])
  return cell


if __name__ == '__main__':
  absltest.main()

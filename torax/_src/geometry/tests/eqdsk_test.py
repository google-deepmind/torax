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
from collections.abc import Callable
import dataclasses
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import eqdsk as eqdsk_lib
import numpy as np
from torax._src import array_typing
from torax._src.geometry import base
from torax._src.geometry import eqdsk
from torax._src.geometry import geometry_loader

# pylint: disable=invalid-name


class EqdskGeometryTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(geometry_file='iterhybrid_cocos02.eqdsk', cocos=2),
      dict(geometry_file='iterhybrid_cocos11.eqdsk', cocos=11),
  ])
  def test_build_geometry_from_eqdsk(self, geometry_file, cocos):
    """Test that EQDSK geometries can be built."""
    config = eqdsk.EQDSKConfig(geometry_file=geometry_file, cocos=cocos)
    config.build_geometry()

  def test_eqdsk_cocos_conversion_is_consistent(self):
    """Tests that EQDSK geometries from different COCOS are identical after conversion."""
    geo_cocos2 = eqdsk.EQDSKConfig(
        geometry_file='iterhybrid_cocos02.eqdsk', cocos=2
    ).build_geometry()
    geo_cocos11 = eqdsk.EQDSKConfig(
        geometry_file='iterhybrid_cocos11.eqdsk', cocos=11
    ).build_geometry()
    for field in dataclasses.fields(geo_cocos2):
      name = field.name
      val1 = getattr(geo_cocos2, name)
      val2 = getattr(geo_cocos11, name)
      if isinstance(val1, array_typing.Array):
        np.testing.assert_allclose(
            val1, val2, err_msg=f'Field "{name}" mismatch.'
        )
      elif val1 is None:
        self.assertIsNone(val2, msg=f'Field "{name}" mismatch.')
      else:
        self.assertEqual(val1, val2, msg=f'Field "{name}" mismatch.')

  def test_build_geometry_from_eqdsk_object(self):
    """Test that EQDSK geometries can be built from an EQDSKInterface object."""
    geo_dir = geometry_loader.get_geometry_dir()
    file_name = 'iterhybrid_cocos02.eqdsk'
    file_path = f'{geo_dir}/{file_name}'

    eqdsk_obj = eqdsk_lib.EQDSKInterface.from_file(file_path, from_cocos=2)

    geo_file = eqdsk.EQDSKConfig(
        geometry_file=file_name, cocos=2
    ).build_geometry()
    geo_obj = eqdsk.EQDSKConfig(
        eqdsk_object=eqdsk_obj, cocos=2
    ).build_geometry()

    for field in dataclasses.fields(geo_file):
      name = field.name
      val1 = getattr(geo_file, name)
      val2 = getattr(geo_obj, name)
      if isinstance(val1, array_typing.Array):
        np.testing.assert_allclose(
            val1, val2, err_msg=f'Field "{name}" mismatch.'
        )
      elif val1 is None:
        self.assertIsNone(val2, msg=f'Field "{name}" mismatch.')
      else:
        self.assertEqual(val1, val2, msg=f'Field "{name}" mismatch.')

  def test_trapped_fraction_is_physically_sensible(self):
    """Tests that the exact trapped particle fraction is well-behaved."""
    geo = eqdsk.EQDSKConfig(
        geometry_file='iterhybrid_cocos11.eqdsk',
        cocos=11,
        trapped_fraction_source=base.TrappedFractionSource.EXACT,
    ).build_geometry()
    trapped_fraction = geo.trapped_fraction_face
    self.assertIsNotNone(trapped_fraction)
    with self.subTest('axis_zero'):
      # No trapped particles on the magnetic axis, where B is uniform.
      self.assertAlmostEqual(float(trapped_fraction[0]), 0.0)
    with self.subTest('range_zero_to_one'):
      # The trapped particle fraction is a fraction, so must lie in [0, 1].
      self.assertTrue(np.all(trapped_fraction >= 0.0))
      self.assertTrue(np.all(trapped_fraction <= 1.0))
    with self.subTest('monotonicity'):
      # Trapped fraction increases with normalized radius over most of the
      # profile (small deviations from strict monotonicity are possible near
      # the edge for diverted geometries, due to the X-point).
      self.assertGreater(
          np.mean(np.diff(trapped_fraction) >= -1e-6),
          0.8,
      )

  def test_exact_trapped_fraction_warns_for_bad_values(self):
    """Tests that EXACT trapped fraction warns when bad values are overwritten."""
    with mock.patch.object(
        eqdsk.trapped_fraction_lib,
        'calculate_bounce_averaged_trapped_fraction',
        return_value=np.nan,
    ):
      with self.assertLogs(level='WARNING') as cm:
        geo = eqdsk.EQDSKConfig(
            geometry_file='iterhybrid_cocos11.eqdsk',
            cocos=11,
            trapped_fraction_source=base.TrappedFractionSource.EXACT,
        ).build_geometry()
    self.assertTrue(
        any('Overwriting' in msg and 'EXACT' in msg for msg in cm.output)
    )
    self.assertIsNotNone(geo.trapped_fraction_face)

  def test_trapped_fraction_source_file_not_supported(self):
    """Tests that FILE is rejected for EQDSK (no precomputed value)."""
    with self.assertRaisesRegex(ValueError, 'not supported for EQDSKConfig'):
      eqdsk.EQDSKConfig(
          geometry_file='iterhybrid_cocos11.eqdsk',
          cocos=11,
          trapped_fraction_source=base.TrappedFractionSource.FILE,
      )

  def test_trapped_fraction_geometry_consistent_with_sauter(self):
    """Tests that the exact and Sauter trapped fractions roughly agree."""
    geo_sauter = eqdsk.EQDSKConfig(
        geometry_file='iterhybrid_cocos11.eqdsk',
        cocos=11,
        trapped_fraction_source=base.TrappedFractionSource.SAUTER,
    ).build_geometry()
    geo_geometry = eqdsk.EQDSKConfig(
        geometry_file='iterhybrid_cocos11.eqdsk',
        cocos=11,
        trapped_fraction_source=base.TrappedFractionSource.EXACT,
    ).build_geometry()
    # Moderately coarse tolerance: Sauter is only an analytic approximation,
    # so it need not match the exact integral closely, but a large deviation
    # would indicate a bug rather than the expected model discrepancy.
    np.testing.assert_allclose(
        geo_geometry.trapped_fraction_face,
        geo_sauter.trapped_fraction_face,
        atol=0.05,
        rtol=0.15,
    )

  def test_eqdsk_serialization_round_trip(self):
    """Test that EQDSKConfig with eqdsk_object can be serialized and deserialized."""
    geo_dir = geometry_loader.get_geometry_dir()
    file_name = 'iterhybrid_cocos11.eqdsk'
    file_path = f'{geo_dir}/{file_name}'
    eqdsk_obj = eqdsk_lib.EQDSKInterface.from_file(file_path, from_cocos=11)

    config = eqdsk.EQDSKConfig(eqdsk_object=eqdsk_obj, cocos=11)
    dumped = config.model_dump()
    config_restored = eqdsk.EQDSKConfig.from_dict(dumped)
    self.assertIsNotNone(config_restored.eqdsk_object)
    self.assertIsInstance(
        config_restored.eqdsk_object, eqdsk_lib.EQDSKInterface
    )

    # Verify that the built geometries match
    geo_original = config.build_geometry()
    geo_restored = config_restored.build_geometry()

    for field in dataclasses.fields(geo_original):
      name = field.name
      val1 = getattr(geo_original, name)
      val2 = getattr(geo_restored, name)
      if isinstance(val1, array_typing.Array):
        np.testing.assert_allclose(
            val1, val2, err_msg=f'Field "{name}" mismatch (dict).'
        )
      elif val1 is None:
        self.assertIsNone(val2, msg=f'Field "{name}" mismatch (dict).')
      else:
        self.assertEqual(val1, val2, msg=f'Field "{name}" mismatch (dict).')

  def _coarsened_iterhybrid_eqdsk(
      self, stride: int
  ) -> eqdsk_lib.EQDSKInterface:
    """Returns the ITER hybrid EQDSK with every `stride`-th grid point kept."""
    geo_dir = geometry_loader.get_geometry_dir()
    eqdsk_obj = eqdsk_lib.EQDSKInterface.from_file(
        f'{geo_dir}/iterhybrid_cocos11.eqdsk', from_cocos=11
    )
    eqdsk_obj.psi = eqdsk_obj.psi[::stride, ::stride]
    eqdsk_obj.nx, eqdsk_obj.nz = eqdsk_obj.psi.shape
    # 1D profiles are on nx evenly spaced psi_norm points.
    for name in ['fpol', 'pressure', 'pprime', 'ffprime', 'qpsi', 'psinorm']:
      setattr(eqdsk_obj, name, getattr(eqdsk_obj, name)[::stride])
    return eqdsk_obj

  def test_target_psi_grid_resolution_improves_coarse_grid_accuracy(self):
    """Tests that grid refinement recovers the fine-grid plasma current."""
    # Use last_surface_factor=0.97 so that all contours remain closed on the
    # unrefined stride=2 grid (at the default 0.99, the outermost two contours
    # are clipped open by the LCFS bounding box).
    reference_Ip = (
        eqdsk.EQDSKConfig(
            geometry_file='iterhybrid_cocos11.eqdsk',
            cocos=11,
            last_surface_factor=0.97,
        )
        .build_geometry()
        .Ip_profile_face[-1]
    )
    # coarsened_obj has nx = 65, nz = 65.
    # resolution=None: no refinement. resolution=130: ceil(130/65) = 2.
    coarse_Ip_unrefined = (
        eqdsk.EQDSKConfig(
            eqdsk_object=self._coarsened_iterhybrid_eqdsk(stride=2),
            cocos=11,
            last_surface_factor=0.97,
            target_psi_grid_resolution=None,
        )
        .build_geometry()
        .Ip_profile_face[-1]
    )
    coarse_Ip_refined = (
        eqdsk.EQDSKConfig(
            eqdsk_object=self._coarsened_iterhybrid_eqdsk(stride=2),
            cocos=11,
            last_surface_factor=0.97,
            target_psi_grid_resolution=130,
        )
        .build_geometry()
        .Ip_profile_face[-1]
    )
    error_unrefined = abs(coarse_Ip_unrefined / reference_Ip - 1)
    error_refined = abs(coarse_Ip_refined / reference_Ip - 1)
    self.assertLess(error_refined, 5e-3)
    self.assertLess(error_refined, error_unrefined)

  def test_target_psi_grid_resolution_enables_contouring_on_coarse_grid(self):
    """Tests that grid refinement fixes contouring failures on coarse grids."""
    with self.subTest('unrefined_fails'):
      with self.assertRaisesRegex(
          ValueError, 'No closed contour enclosing the magnetic axis'
      ):
        eqdsk.EQDSKConfig(
            eqdsk_object=self._coarsened_iterhybrid_eqdsk(stride=4),
            cocos=11,
            target_psi_grid_resolution=None,
        ).build_geometry()
    with self.subTest('refined_succeeds'):
      # stride 4 coarsened_obj has nx = 33, nz = 33.
      # target_psi_grid_resolution = 130 leads to ceil(130 / 33) = 4.
      eqdsk.EQDSKConfig(
          eqdsk_object=self._coarsened_iterhybrid_eqdsk(stride=4),
          cocos=11,
          target_psi_grid_resolution=130,
      ).build_geometry()

  def test_target_psi_grid_resolution_preserves_well_resolved_geometry(self):
    """Tests that refining an already well-resolved grid changes little."""
    # iterhybrid_cocos11 has nx = 129, nz = 129.
    geo_unrefined = eqdsk.EQDSKConfig(
        geometry_file='iterhybrid_cocos11.eqdsk',
        cocos=11,
        target_psi_grid_resolution=None,
    ).build_geometry()
    # target_psi_grid_resolution = 260 leads to ceil(260 / 129) = 2.
    geo_refined = eqdsk.EQDSKConfig(
        geometry_file='iterhybrid_cocos11.eqdsk',
        cocos=11,
        target_psi_grid_resolution=260,
    ).build_geometry()
    for name in ['Phi_face', 'volume_face', 'area_face', 'F_face']:
      np.testing.assert_allclose(
          getattr(geo_unrefined, name),
          getattr(geo_refined, name),
          rtol=5e-3,
          atol=1e-8,
          err_msg=f'Field "{name}" mismatch.',
      )

  def test_default_target_psi_grid_resolution(self):
    """Tests that default target_psi_grid_resolution is 128."""
    coarsened_obj = self._coarsened_iterhybrid_eqdsk(stride=2)
    # coarsened_obj has nx = 65, nz = 65.
    # Default target_psi_grid_resolution = 128 gives ceil(128 / 65) = 2.
    geo_default = eqdsk.EQDSKConfig(
        eqdsk_object=coarsened_obj,
        cocos=11,
    ).build_geometry()
    geo_explicit_128 = eqdsk.EQDSKConfig(
        eqdsk_object=coarsened_obj,
        cocos=11,
        target_psi_grid_resolution=128,
    ).build_geometry()
    for name in ['Phi_face', 'volume_face', 'area_face', 'F_face']:
      np.testing.assert_allclose(
          getattr(geo_default, name),
          getattr(geo_explicit_128, name),
          err_msg=f'Field "{name}" mismatch for default target resolution.',
      )

  def test_target_psi_grid_resolution_invalid_value_raises(self):
    """Tests that target_psi_grid_resolution < 1 raises a validation error."""
    with self.assertRaises(ValueError):
      eqdsk.EQDSKConfig(
          geometry_file='iterhybrid_cocos11.eqdsk',
          cocos=11,
          target_psi_grid_resolution=0,
      )

  def test_flux_surface_quantities_are_invariant_to_contour_seam(self):
    # Moving the start/end vertex of each closed contour must not change any
    # geometry quantity, since a closed flux surface has no preferred start.
    reference = _build_iterhybrid_geometry()
    shifted_seam = _build_iterhybrid_geometry_with_contours(
        lambda contours: [_shift_closed_contour_seam(c, 10) for c in contours]
    )
    _assert_geometries_close(shifted_seam, reference)

  @parameterized.parameters(8, 64)
  def test_closed_line_elements_sum_to_polygon_perimeter(self, n_points):
    radius = 0.5
    theta = np.linspace(0.0, 2 * np.pi, n_points, endpoint=False)
    r = 6.2 + radius * np.cos(theta)
    z = 0.3 + radius * np.sin(theta)

    dl = eqdsk._line_elements(r, z)

    # Perimeter of a regular n-gon inscribed in a circle of this radius.
    polygon_perimeter = 2 * n_points * radius * np.sin(np.pi / n_points)
    np.testing.assert_allclose(np.sum(dl), polygon_perimeter, rtol=1e-12)

  def test_coarse_grid_raises_for_open_flux_surfaces(self):
    # On this coarsened grid at the default last_surface_factor=0.99, the
    # outermost two contours are cut open by the LCFS bounding box.
    with self.assertRaisesRegex(
        ValueError,
        r'No closed contour enclosing the magnetic axis found for 2 EQDSK flux'
        r' surface\(s\)',
    ):
      eqdsk.EQDSKConfig(
          eqdsk_object=self._coarsened_iterhybrid_eqdsk(stride=2),
          cocos=11,
          target_psi_grid_resolution=None,
      ).build_geometry()

  def test_select_contour_enclosing_magnetic_axis_skips_other_contours(self):
    open_contour = np.array([[6.0, 1.0], [6.5, 1.2], [7.0, 1.0], [7.5, 0.5]])
    flux_surface = np.array([
        [6.0, 0.0],
        [7.0, 0.0],
        [7.0, 1.0],
        [6.0, 1.0],
        [6.0, 0.0],
    ])

    selected = eqdsk._select_contour_enclosing_magnetic_axis(
        [open_contour, _SPURIOUS_LOOP, flux_surface], r_axis=6.5, z_axis=0.5
    )

    np.testing.assert_array_equal(selected, flux_surface[:-1])

  def test_select_contour_enclosing_magnetic_axis_returns_none_when_none_match(
      self,
  ):
    open_contour = np.array([[6.0, 1.0], [6.5, 1.2], [7.0, 1.0], [7.5, 0.5]])

    self.assertIsNone(
        eqdsk._select_contour_enclosing_magnetic_axis(
            [open_contour, _SPURIOUS_LOOP], r_axis=6.4, z_axis=0.5
        )
    )

  def test_select_contour_enclosing_magnetic_axis_raises_on_multiple_matches(
      self,
  ):
    inner_loop = np.array([
        [6.0, 0.0],
        [7.0, 0.0],
        [7.0, 1.0],
        [6.0, 1.0],
        [6.0, 0.0],
    ])
    outer_loop = np.array([
        [5.5, -0.5],
        [7.5, -0.5],
        [7.5, 1.5],
        [5.5, 1.5],
        [5.5, -0.5],
    ])

    with self.assertRaisesRegex(
        ValueError, 'Expected at most one closed contour'
    ):
      eqdsk._select_contour_enclosing_magnetic_axis(
          [inner_loop, outer_loop], r_axis=6.5, z_axis=0.5
      )

  @parameterized.named_parameters(
      ('interior', 6.3, 0.3, True),
      ('exterior_left', 5.0, 0.5, False),
      ('exterior_right', 8.0, 0.5, False),
      ('ray_through_vertex', 6.3, 0.5, True),
      ('ray_along_horizontal_edge', 5.0, 0.0, False),
      ('non_convex_concavity_outside', 6.8, 0.5, False),
      ('non_convex_wing_inside', 6.8, 0.8, True),
  )
  def test_encloses_point(self, r, z, expected):
    # Non-convex polygon (C-shaped indentation on the right side from z=0.2 to
    # z=0.7, plus a vertex at (6.5, 0.5) to exercise ray-vertex intersection
    # and a horizontal bottom edge at z=0.0).
    polygon = np.array([
        [6.0, 0.0],
        [7.0, 0.0],
        [7.0, 0.2],
        [6.5, 0.5],
        [7.0, 0.7],
        [7.0, 1.0],
        [6.0, 1.0],
    ])

    self.assertEqual(eqdsk._encloses_point(polygon, r=r, z=z), expected)


# A closed contour loop far from the plasma, standing in for a contour piece in
# the private-flux region. contourpy repeats the first vertex at the end.
_SPURIOUS_LOOP = np.array([
    [1.0, -5.0],
    [1.2, -5.0],
    [1.2, -4.8],
    [1.0, -4.8],
    [1.0, -5.0],
])


def _shift_closed_contour_seam(contour: np.ndarray, shift: int) -> np.ndarray:
  """Returns a closed contour that starts `shift` vertices later.

  The contour traces the same loop, but its first (and repeated last) vertex is
  a different point on the loop.

  Args:
    contour: Closed contour from contourpy, with its first vertex repeated at
      the end.
    shift: Number of vertices by which to move the start of the contour.

  Returns:
    Closed contour with the start/end vertex shifted by `shift` positions.
  """
  if not eqdsk._is_closed(contour):
    return contour
  rolled = np.roll(contour[:-1], shift, axis=0)
  return np.vstack([rolled, rolled[:1]])


def _build_iterhybrid_geometry():
  return eqdsk.EQDSKConfig(
      geometry_file='iterhybrid_cocos11.eqdsk', cocos=11
  ).build_geometry()


def _build_iterhybrid_geometry_with_contours(
    transform_contours: Callable[[list[np.ndarray]], list[np.ndarray]],
):
  """Builds the ITER hybrid EQDSK geometry with modified contourpy output.

  Args:
    transform_contours: Applied to the list of contour lines that contourpy
      returns for each psi level, before the geometry code sees them.

  Returns:
    The constructed StandardGeometry for the ITER hybrid EQDSK equilibrium.
  """
  original_contour_generator = eqdsk.contourpy.contour_generator

  def _contour_generator(*args, **kwargs):
    generator = original_contour_generator(*args, **kwargs)
    return mock.Mock(
        create_contour=lambda level: transform_contours(
            generator.create_contour(level)
        )
    )

  with mock.patch.object(
      eqdsk.contourpy, 'contour_generator', _contour_generator
  ):
    return _build_iterhybrid_geometry()


def _assert_geometries_close(actual, expected):
  """Asserts that all array fields of two geometries agree to rounding error."""
  for field in dataclasses.fields(expected):
    expected_value = getattr(expected, field.name)
    if isinstance(expected_value, array_typing.Array):
      np.testing.assert_allclose(
          getattr(actual, field.name),
          expected_value,
          rtol=1e-12,
          atol=1e-12,
          err_msg=field.name,
      )


if __name__ == '__main__':
  absltest.main()

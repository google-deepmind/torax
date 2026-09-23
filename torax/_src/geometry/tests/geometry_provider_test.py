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
import chex
import jax
import jax.numpy as jnp
import numpy as np
from torax._src import jax_utils
from torax._src.geometry import circular_geometry
from torax._src.geometry import geometry
from torax._src.geometry import geometry_provider
from torax._src.geometry import pydantic_model as geometry_pydantic_model
from torax._src.geometry import standard_geometry


def _build_time_dependent_circular_provider(
    calcphibdot: bool = True,
) -> geometry_provider.TimeDependentGeometryProvider:
  geo_0 = circular_geometry.CircularConfig(
      R_major=6.2, a_minor=2.0, B_0=5.3
  ).build_geometry()
  geo_1 = circular_geometry.CircularConfig(
      R_major=7.4, a_minor=1.0, B_0=6.5
  ).build_geometry()
  return geometry_provider.TimeDependentGeometryProvider.create_provider(
      {0.0: geo_0, 10.0: geo_1}, calcphibdot=calcphibdot
  )


def _build_time_dependent_chease_provider() -> (
    standard_geometry.StandardGeometryProvider
):
  provider = geometry_pydantic_model.Geometry.from_dict({
      'geometry_type': 'chease',
      'Ip_from_parameters': True,
      'n_rho': 10,
      'geometry_configs': {
          0.0: {'geometry_file': 'iterhybrid.mat2cols', 'R_major': 6.2},
          10.0: {'geometry_file': 'iterhybrid.mat2cols', 'R_major': 7.0},
      },
  }).build_provider
  assert isinstance(provider, standard_geometry.StandardGeometryProvider)
  return provider


class GeometryProviderTest(parameterized.TestCase):

  def test_constant_geometry_return_same_value(self):
    geo = circular_geometry.CircularConfig().build_geometry()
    provider = geometry_provider.ConstantGeometryProvider(geo)
    self.assertEqual(provider(0.0), geo)
    self.assertEqual(provider(1.0), geo)
    self.assertEqual(provider(2.0), geo)

  def test_time_dependent_geometry_return_different_values(self):
    geo_0 = circular_geometry.CircularConfig(
        R_major=6.2, a_minor=2.0, B_0=5.3
    ).build_geometry()
    geo_1 = circular_geometry.CircularConfig(
        R_major=7.4, a_minor=1.0, B_0=6.5
    ).build_geometry()
    provider = geometry_provider.TimeDependentGeometryProvider.create_provider(
        {0.0: geo_0, 10.0: geo_1},
        calcphibdot=True,
    )
    geo = provider(5.0)
    np.testing.assert_allclose(geo.R_major, 6.8)
    np.testing.assert_allclose(geo.a_minor, 1.5)
    np.testing.assert_allclose(geo.B_0, 5.9)

  def test_time_dependent_different_types(self):
    geo_0 = circular_geometry.CircularConfig().build_geometry()
    geo_1 = dataclasses.replace(geo_0, geometry_type=geometry.GeometryType.FBT)
    with self.assertRaisesRegex(
        ValueError, 'All geometries must have the same geometry type.'
    ):
      geometry_provider.TimeDependentGeometryProvider.create_provider(
          {0.0: geo_0, 10.0: geo_1},
          calcphibdot=True,
      )

  def test_time_dependent_different_meshes(self):
    geo_0 = circular_geometry.CircularConfig(n_rho=25).build_geometry()
    geo_1 = circular_geometry.CircularConfig(n_rho=50).build_geometry()
    with self.assertRaisesRegex(
        ValueError, 'All geometries must have the same mesh.'
    ):
      geometry_provider.TimeDependentGeometryProvider.create_provider(
          {0.0: geo_0, 10.0: geo_1},
          calcphibdot=True,
      )

  def test_none_z_magnetic_axis_stays_none_time_dependent(self):
    geo = circular_geometry.CircularConfig().build_geometry()
    geo = dataclasses.replace(geo, _z_magnetic_axis=None)
    provider = geometry_provider.TimeDependentGeometryProvider.create_provider(
        {0.0: geo, 10.0: geo},
        calcphibdot=True,
    )
    self.assertIsNone(provider(0.0)._z_magnetic_axis)
    self.assertIsNone(provider(10.0)._z_magnetic_axis)

  @parameterized.named_parameters(
      ('circular', _build_time_dependent_circular_provider),
      ('chease', _build_time_dependent_chease_provider),
  )
  def test_precomputed_matches_interpolating_provider(self, build_provider):
    provider = build_provider()
    times = np.array([0.0, 2.5, 5.0, 7.5, 10.0], dtype=jax_utils.get_np_dtype())
    precomputed = geometry_provider.PrecomputedGeometryProvider.from_provider(
        provider, times
    )
    self.assertEqual(precomputed.torax_mesh, provider.torax_mesh)
    for t in times:
      with self.subTest(t=t):
        expected = provider(t)
        actual = precomputed(t)
        self.assertIs(type(actual), type(expected))
        self.assertEqual(actual.geometry_type, expected.geometry_type)
        chex.assert_trees_all_close(actual, expected)

  def test_precomputed_matches_interpolating_provider_under_jit(self):
    provider = _build_time_dependent_circular_provider()
    times = np.array([0.0, 5.0, 10.0], dtype=jax_utils.get_np_dtype())
    precomputed = geometry_provider.PrecomputedGeometryProvider.from_provider(
        provider, times
    )
    jitted = jax.jit(lambda p, t: p(t))
    chex.assert_trees_all_close(
        jitted(precomputed, jnp.asarray(5.0)), provider(5.0)
    )

  def test_precomputed_preserves_none_attributes_and_zero_phibdot(self):
    provider = _build_time_dependent_circular_provider(calcphibdot=False)
    geo = dataclasses.replace(provider(0.0), _z_magnetic_axis=None)
    provider = geometry_provider.TimeDependentGeometryProvider.create_provider(
        {0.0: geo, 10.0: geo}, calcphibdot=False
    )
    precomputed = geometry_provider.PrecomputedGeometryProvider.from_provider(
        provider, np.array([0.0, 10.0])
    )
    for t in (0.0, 4.0):
      geo_precomputed = precomputed(t)
      self.assertIsNone(geo_precomputed._z_magnetic_axis)
      np.testing.assert_array_equal(geo_precomputed.Phi_b_dot, 0.0)

  @parameterized.product(
      build_provider=(
          _build_time_dependent_circular_provider,
          _build_time_dependent_chease_provider,
      ),
      errors_enabled=(False, True),
      execution=('eager', 'jit', 'vmap'),
  )
  def test_precomputed_falls_back_to_original_provider(
      self, build_provider, errors_enabled, execution
  ):
    provider = build_provider()
    # The cache deliberately covers only part of the provider's time interval.
    precomputed = geometry_provider.PrecomputedGeometryProvider.from_provider(
        provider, np.array([2.0, 5.0, 8.0])
    )
    times = jnp.asarray([-1.0, 0.0, 2.0, 3.5, 5.0, 6.0, 8.0, 9.0, 11.0])

    def evaluate(p, t):
      return p(t)

    with jax_utils.enable_errors(errors_enabled):
      if execution == 'vmap':
        evaluate = jax.jit(jax.vmap(evaluate, in_axes=(None, 0)))
        chex.assert_trees_all_close(
            evaluate(precomputed, times), evaluate(provider, times)
        )
      else:
        if execution == 'jit':
          evaluate = jax.jit(evaluate)
        for t in times:
          expected = provider(t)
          actual = evaluate(precomputed, t)
          self.assertIs(type(actual), type(expected))
          chex.assert_trees_all_close(actual, expected)

  @parameterized.named_parameters(
      ('singleton', [0.0]), ('multiple', [0.0, 5.0, 10.0])
  )
  def test_precomputed_uses_cache_only_for_exact_hits(self, times):
    provider = _build_time_dependent_circular_provider()
    precomputed = geometry_provider.PrecomputedGeometryProvider.from_provider(
        provider, np.asarray(times)
    )
    # Distinguish the cache from the fallback to verify which branch runs.
    precomputed = dataclasses.replace(
        precomputed,
        geometries=dataclasses.replace(
            precomputed.geometries,
            R_major=precomputed.geometries.R_major + 1.0,
        ),
    )
    evaluate = jax.jit(lambda p, t: p(t))
    for t in times:
      np.testing.assert_allclose(
          evaluate(precomputed, t).R_major, provider(t).R_major + 1.0
      )
    chex.assert_trees_all_close(evaluate(precomputed, 4.0), provider(4.0))

  def test_precomputed_does_not_snap_nearby_times(self):
    original = _build_time_dependent_circular_provider()
    provider = geometry_provider.TimeDependentGeometryProvider.create_provider(
        {0.0: original(0.0), 1e-9: original(10.0), 2e-9: original(0.0)},
        calcphibdot=True,
    )
    precomputed = geometry_provider.PrecomputedGeometryProvider.from_provider(
        provider, np.array([0.0, 1e-9, 2e-9])
    )
    dtype = jax_utils.get_np_dtype()
    # Include both sides of a derivative discontinuity and sub-tolerance steps.
    times = jnp.asarray([
        4e-10,
        np.nextafter(dtype(1e-9), dtype(0.0)),
        np.nextafter(dtype(1e-9), dtype(2e-9)),
        1.6e-9,
    ])
    evaluate = jax.jit(jax.vmap(lambda p, t: p(t), in_axes=(None, 0)))
    chex.assert_trees_all_close(
        evaluate(precomputed, times), evaluate(provider, times)
    )

  @parameterized.named_parameters(
      ('empty', []),
      ('scalar', 1.0),
      ('rank_two', [[0.0, 1.0]]),
      ('duplicate', [0.0, 0.0]),
      ('unsorted', [1.0, 0.0]),
      ('nan', [0.0, np.nan]),
      ('infinite', [0.0, np.inf]),
  )
  def test_precomputed_rejects_invalid_times(self, times):
    with self.assertRaisesRegex(ValueError, 'nonempty 1D array'):
      geometry_provider.PrecomputedGeometryProvider.from_provider(
          _build_time_dependent_circular_provider(), np.asarray(times)
      )

  def test_precomputed_rejects_times_that_collapse_in_simulation_dtype(self):
    with self.assertRaisesRegex(ValueError, 'simulation dtype'):
      geometry_provider.PrecomputedGeometryProvider.from_provider(
          _build_time_dependent_circular_provider(),
          np.array([1.0, 1.0 + np.finfo(jax_utils.get_np_dtype()).eps / 4]),
      )


if __name__ == '__main__':
  absltest.main()

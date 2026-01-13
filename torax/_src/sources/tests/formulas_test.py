# Copyright 2026 DeepMind Technologies Limited
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
import numpy as np
from torax._src import math_utils
from torax._src.geometry import circular_geometry
from torax._src.sources import formulas


class GaussianProfileTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.geo = circular_geometry.CircularConfig(
        n_rho=25,
    ).build_geometry()

  def test_output_shape_matches_grid(self):
    """Output profile shape matches the geometry grid."""
    profile = formulas.gaussian_profile(
        self.geo,
        center=0.5,
        width=0.1,
        total=1e22,
    )
    self.assertEqual(profile.shape, self.geo.rho_norm.shape)

  @parameterized.parameters(
      dict(total=1e20),
      dict(total=1e22),
      dict(total=5e23),
  )
  def test_volume_integral_equals_total(self, total):
    """Volume integral of gaussian profile equals the specified total."""
    profile = formulas.gaussian_profile(
        self.geo,
        center=0.5,
        width=0.1,
        total=total,
    )
    integrated = math_utils.volume_integration(profile, self.geo)
    np.testing.assert_allclose(integrated, total, rtol=1e-6)

  def test_profile_is_peaked_at_center(self):
    """Gaussian profile is maximum near the specified center."""
    center = 0.3
    profile = formulas.gaussian_profile(
        self.geo,
        center=center,
        width=0.1,
        total=1e22,
    )
    # Find index closest to center.
    center_idx = np.argmin(np.abs(self.geo.rho_norm - center))
    # Profile should be maximum at or near center.
    self.assertEqual(np.argmax(profile), center_idx)

  @parameterized.parameters(
      dict(center=0.2),
      dict(center=0.5),
      dict(center=0.8),
  )
  def test_profile_peak_location_varies_with_center(self, center):
    """Profile peak location follows the specified center parameter."""
    profile = formulas.gaussian_profile(
        self.geo,
        center=center,
        width=0.1,
        total=1e22,
    )
    peak_rho = self.geo.rho_norm[np.argmax(profile)]
    # Peak should be within one grid spacing of specified center.
    np.testing.assert_allclose(peak_rho, center, atol=self.geo.drho_norm)

  def test_wider_profile_is_less_peaked(self):
    """A wider gaussian profile has a lower peak value."""
    narrow_profile = formulas.gaussian_profile(
        self.geo,
        center=0.5,
        width=0.05,
        total=1e22,
    )
    wide_profile = formulas.gaussian_profile(
        self.geo,
        center=0.5,
        width=0.2,
        total=1e22,
    )
    self.assertGreater(np.max(narrow_profile), np.max(wide_profile))


class ExponentialProfileTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.geo = circular_geometry.CircularConfig(
        n_rho=25,
    ).build_geometry()

  def test_output_shape_matches_grid(self):
    """Output profile shape matches the geometry grid."""
    profile = formulas.exponential_profile(
        self.geo,
        decay_start=0.8,
        width=0.1,
        total=1e22,
    )
    self.assertEqual(profile.shape, self.geo.rho_norm.shape)

  @parameterized.parameters(
      dict(total=1e20),
      dict(total=1e22),
      dict(total=5e23),
  )
  def test_volume_integral_equals_total(self, total):
    """Volume integral of exponential profile equals the specified total."""
    profile = formulas.exponential_profile(
        self.geo,
        decay_start=0.8,
        width=0.1,
        total=total,
    )
    integrated = math_utils.volume_integration(profile, self.geo)
    np.testing.assert_allclose(integrated, total, rtol=1e-6)

  def test_profile_increases_toward_decay_start(self):
    """Exponential profile increases as rho approaches decay_start."""
    decay_start = 0.8
    profile = formulas.exponential_profile(
        self.geo,
        decay_start=decay_start,
        width=0.1,
        total=1e22,
    )
    # Profile should be monotonically increasing up to decay_start.
    decay_start_idx = np.argmin(np.abs(self.geo.rho_norm - decay_start))
    for i in range(decay_start_idx):
      self.assertLessEqual(profile[i], profile[i + 1])

  def test_narrower_width_gives_steeper_profile(self):
    """A smaller width gives a steeper exponential decay."""
    decay_start = 0.8
    narrow_profile = formulas.exponential_profile(
        self.geo,
        decay_start=decay_start,
        width=0.05,
        total=1e22,
    )
    wide_profile = formulas.exponential_profile(
        self.geo,
        decay_start=decay_start,
        width=0.2,
        total=1e22,
    )
    # The narrow profile should have higher values near decay_start
    # and lower values far from it.
    decay_idx = np.argmin(np.abs(self.geo.rho_norm - decay_start))
    core_idx = np.argmin(np.abs(self.geo.rho_norm - 0.2))
    ratio_narrow = narrow_profile[decay_idx] / narrow_profile[core_idx]
    ratio_wide = wide_profile[decay_idx] / wide_profile[core_idx]
    self.assertGreater(ratio_narrow, ratio_wide)


if __name__ == '__main__':
  absltest.main()

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
"""Tests for domain-agnostic adaptive physics module."""

import shutil
import tempfile
from absl.testing import absltest
import numpy as np
from torax._src.data_harvesting import StagingSink
from torax._src.physics import adaptive_physics_module


class AdaptivePhysicsModuleTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.test_dir)
    super().tearDown()

  def test_spatial_smoothing(self):
    x = np.linspace(0.0, 1.0, 20)
    # Step function
    y = np.where(x > 0.5, 10.0, 0.0)
    y_smooth = adaptive_physics_module.apply_spatial_smoothing(y, x, sigma=0.1)

    self.assertEqual(y.shape, y_smooth.shape)
    # Smoothing must soften the step discontinuity
    self.assertLess(y_smooth[10], 10.0)
    self.assertGreater(y_smooth[9], 0.0)

  def test_decide_fallback_full_profile(self):
    cfg = adaptive_physics_module.AdaptivePhysicsConfig(
        uncertainty_threshold=0.25,
        fallback_mode="full_profile",
    )
    engine = adaptive_physics_module.AdaptivePhysicsEngine(config=cfg)

    # Case 1: All uncertainties low
    rel_unc_low = np.array([0.1, 0.05, 0.20, 0.15])
    needs_fallback, mask = engine.decide_fallback(rel_unc_low)
    self.assertFalse(needs_fallback)
    self.assertFalse(np.any(mask))

    # Case 2: One point exceeds threshold
    rel_unc_high = np.array([0.1, 0.05, 0.30, 0.15])
    needs_fallback, mask = engine.decide_fallback(rel_unc_high)
    self.assertTrue(needs_fallback)
    # In full_profile mode, entire profile is flagged
    self.assertTrue(np.all(mask))

  def test_decide_fallback_per_face(self):
    cfg = adaptive_physics_module.AdaptivePhysicsConfig(
        uncertainty_threshold=0.25,
        fallback_mode="per_face",
    )
    engine = adaptive_physics_module.AdaptivePhysicsEngine(config=cfg)

    rel_unc = np.array([0.1, 0.35, 0.15, 0.40])
    needs_fallback, mask = engine.decide_fallback(rel_unc)
    self.assertTrue(needs_fallback)
    np.testing.assert_array_equal(mask, [False, True, False, True])

  def test_fuse_and_smooth(self):
    cfg = adaptive_physics_module.AdaptivePhysicsConfig(
        uncertainty_threshold=0.25,
        fallback_mode="per_face",
        smoothing_sigma=0.05,
    )
    engine = adaptive_physics_module.AdaptivePhysicsEngine(config=cfg)

    x = np.linspace(0.0, 1.0, 5)
    surr = np.ones(5) * 1.0
    hi_fi = np.ones(5) * 5.0
    mask = np.array([False, False, True, False, False])

    fused = engine.fuse_and_smooth(surr, hi_fi, mask, x)
    self.assertEqual(fused.shape, surr.shape)
    # High-fidelity point was at index 2, so fused[2] should be highest
    self.assertGreater(fused[2], fused[0])

  def test_harvest_if_enabled(self):
    sink = StagingSink(output_dir=self.test_dir, run_id="harvest_test")
    cfg = adaptive_physics_module.AdaptivePhysicsConfig(
        enable_data_harvesting=True
    )
    engine = adaptive_physics_module.AdaptivePhysicsEngine(config=cfg, sink=sink)

    engine.harvest_if_enabled(
        fingerprint="tglf_fingerprint123",
        inputs={"RLTS_1": np.ones(5)},
        high_fidelity_outputs={"efi_gb": np.ones(5) * 2.0},
    )
    flushed_path = sink.flush()
    self.assertIsNotNone(flushed_path)


if __name__ == "__main__":
  absltest.main()

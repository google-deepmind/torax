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
"""Tests for active learning data harvesting."""

import os
import shutil
import tempfile

from absl.testing import absltest
import numpy as np
import pandas as pd
from torax._src.data_harvesting import compute_solver_fingerprint
from torax._src.data_harvesting import HarvestSample
from torax._src.data_harvesting import StagingSink


class HarvestingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.test_dir)
    super().tearDown()

  def test_solver_fingerprint_deterministic(self):
    s1 = {"sat_rule": 1, "n_ky": 12, "collisionality": 0.05}
    s2 = {"collisionality": 0.0500000001, "sat_rule": 1, "n_ky": 12}
    s3 = {"sat_rule": 2, "n_ky": 12, "collisionality": 0.05}

    fp1 = compute_solver_fingerprint("tglf", s1)
    fp2 = compute_solver_fingerprint("tglf", s2)
    fp3 = compute_solver_fingerprint("tglf", s3)

    self.assertEqual(fp1, fp2)
    self.assertNotEqual(fp1, fp3)
    self.assertTrue(fp1.startswith("tglf_"))

  def test_staging_sink_record_and_flush(self):
    sink = StagingSink(output_dir=self.test_dir, run_id="test_run_1")
    n_faces = 20
    sample = HarvestSample(
        fingerprint="tglf_test1234",
        inputs={
            "RLTS_1": np.linspace(1.0, 5.0, n_faces),
            "Q_LOC": np.linspace(1.2, 3.5, n_faces),
        },
        outputs={
            "efi_gb": np.ones(n_faces) * 2.5,
            "efe_gb": np.ones(n_faces) * 1.8,
        },
        uncertainties={
            "unc_efi": np.ones(n_faces) * 0.1,
        },
    )

    sink.record(sample)
    written_file = sink.flush()

    self.assertIsNotNone(written_file)
    self.assertTrue(os.path.exists(written_file))
    self.assertTrue(written_file.endswith(".parquet"))

    # Verify parquet contents
    df = pd.read_parquet(written_file)
    self.assertEqual(len(df), n_faces)
    self.assertIn("in_RLTS_1", df.columns)
    self.assertIn("in_Q_LOC", df.columns)
    self.assertIn("out_efi_gb", df.columns)
    self.assertIn("unc_unc_efi", df.columns)
    self.assertIn("run_id", df.columns)
    self.assertEqual(df["run_id"].iloc[0], "test_run_1")


if __name__ == "__main__":
  absltest.main()

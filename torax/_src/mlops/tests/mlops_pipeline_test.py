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
"""Integration and unit tests for the surrogate retraining and watcher pipeline."""

import json
import os
import shutil
import tempfile
from absl.testing import absltest
import numpy as np
import pandas as pd
from torax._src.data_harvesting import HarvestSample
from torax._src.data_harvesting import StagingSink
from torax._src.mlops import train_surrogate
from torax._src.mlops import watch_and_retrain


class MLOpsPipelineTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.harvest_dir = tempfile.mkdtemp()
    self.models_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.harvest_dir)
    shutil.rmtree(self.models_dir)
    super().tearDown()

  def _stage_dummy_harvest_data(self, n_rows=50, fingerprint="tglf_test_fp"):
    sink = StagingSink(output_dir=self.harvest_dir, run_id="unit_test_run")
    sample = HarvestSample(
        fingerprint=fingerprint,
        inputs={
            "RLNS_1": np.linspace(1.0, 4.0, n_rows),
            "RLTS_1": np.linspace(3.0, 10.0, n_rows),
            "RLTS_2": np.linspace(3.0, 10.0, n_rows),
            "Q_LOC": np.linspace(1.5, 4.0, n_rows),
            "SHAT": np.linspace(0.5, 2.0, n_rows),
            "KAPPA_LOC": np.linspace(1.6, 2.0, n_rows),
            "DELTA_LOC": np.linspace(0.3, 0.5, n_rows),
        },
        outputs={
            "efe_gb": np.linspace(0.5, 5.0, n_rows),
            "efi_gb": np.linspace(1.0, 8.0, n_rows),
            "pfi_gb": np.linspace(0.1, 1.2, n_rows),
        },
    )
    sink.record(sample)
    sink.flush()

  def test_ingest_harvested_dataset(self):
    self._stage_dummy_harvest_data(n_rows=30, fingerprint="tglf_target")
    self._stage_dummy_harvest_data(n_rows=20, fingerprint="tglf_other")

    df_filtered = train_surrogate.ingest_harvested_dataset(
        self.harvest_dir, solver_fingerprint="tglf_target"
    )
    self.assertEqual(len(df_filtered), 30)
    self.assertIn("in_RLTS_1", df_filtered.columns)
    self.assertIn("out_efi_gb", df_filtered.columns)

    df_all = train_surrogate.ingest_harvested_dataset(
        self.harvest_dir, solver_fingerprint=None
    )
    self.assertEqual(len(df_all), 50)

  def test_retrain_and_gate_workflow(self):
    self._stage_dummy_harvest_data(n_rows=40, fingerprint="tglf_sat1")

    res = train_surrogate.retrain_and_gate(
        harvest_dir=self.harvest_dir,
        output_dir=self.models_dir,
        solver_fingerprint="tglf_sat1",
        epochs=5,
        archive_consumed=True,
    )

    self.assertIsNotNone(res.model_version)
    self.assertTrue(os.path.exists(res.model_path))
    self.assertTrue(os.path.exists(res.manifest_path))
    self.assertEqual(res.num_samples_trained, 40)

    # Verify manifest JSON structure
    with open(res.manifest_path, "r") as f:
      manifest = json.load(f)
    self.assertEqual(manifest["solver_fingerprint"], "tglf_sat1")
    self.assertEqual(manifest["num_training_samples"], 40)
    self.assertIn("candidate_rmse", manifest)

    # Verify archiving
    archived_files = os.listdir(os.path.join(self.harvest_dir, "archived"))
    self.assertGreater(len(archived_files), 0)

  def test_watcher_scan_and_single_pass_trigger(self):
    self._stage_dummy_harvest_data(n_rows=25, fingerprint="tglf_sat1")

    count, files = watch_and_retrain.scan_harvest_dir(
        self.harvest_dir, solver_fingerprint="tglf_sat1"
    )
    self.assertEqual(count, 25)
    self.assertEqual(len(files), 1)

    # Single pass with threshold <= 25 triggers retraining
    triggered = watch_and_retrain.run_watch_loop(
        harvest_dir=self.harvest_dir,
        output_dir=self.models_dir,
        solver_fingerprint="tglf_sat1",
        min_samples=20,
        single_pass=True,
    )
    self.assertTrue(triggered)

    # Now files are archived, next scan should find 0 unarchived samples
    count_after, _ = watch_and_retrain.scan_harvest_dir(
        self.harvest_dir, solver_fingerprint="tglf_sat1"
    )
    self.assertEqual(count_after, 0)


if __name__ == "__main__":
  absltest.main()

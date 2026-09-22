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
"""Tests for multi-domain surrogate benchmarking and gating."""

from absl.testing import absltest
import numpy as np
from torax._src.mlops import benchmarks


class BenchmarksTest(absltest.TestCase):

  def test_compute_metrics(self):
    preds = {"efe_gb": np.array([2.0, 4.0]), "efi_gb": np.array([1.0, 3.0])}
    truth = {"efe_gb": np.array([2.0, 5.0]), "efi_gb": np.array([1.0, 3.0])}
    uncs = {"efe_gb": np.array([0.1, 0.2]), "efi_gb": np.array([0.05, 0.05])}

    res = benchmarks.compute_metrics(preds, truth, uncs)
    self.assertEqual(res.num_samples, 2)
    self.assertAlmostEqual(res.mae["efe_gb"], 0.5)
    self.assertAlmostEqual(res.mae["efi_gb"], 0.0)
    self.assertAlmostEqual(res.rmse["efi_gb"], 0.0)
    self.assertAlmostEqual(res.mean_uncertainty["efe_gb"], 0.15)

  def test_generate_synthetic_benchmark_suite(self):
    suite = benchmarks.generate_synthetic_benchmark_suite(n_samples_per_domain=50)
    self.assertIn("iter_like", suite)
    self.assertIn("step_like", suite)
    self.assertIn("harvested_recent", suite)

    for domain_name, data in suite.items():
      self.assertEqual(len(data["RLTS_1"]), 50)
      self.assertIn("efe_gb", data)
      self.assertIn("efi_gb", data)
      self.assertIn("pfi_gb", data)

  def test_evaluate_benchmark_suite_pass(self):
    suite = benchmarks.generate_synthetic_benchmark_suite(n_samples_per_domain=20)

    # Candidate matches ground truth closely
    def candidate_eval(inputs):
      preds = {
          "efe_gb": inputs["efe_gb"] + 0.01,
          "efi_gb": inputs["efi_gb"] + 0.01,
          "pfi_gb": inputs["pfi_gb"] + 0.01,
      }
      return preds, None

    # Champion has higher error
    def champion_eval(inputs):
      preds = {
          "efe_gb": inputs["efe_gb"] + 0.5,
          "efi_gb": inputs["efi_gb"] + 0.5,
          "pfi_gb": inputs["pfi_gb"] + 0.5,
      }
      return preds, None

    res = benchmarks.evaluate_benchmark_suite(
        candidate_fn=candidate_eval,
        champion_fn=champion_eval,
        benchmark_suite=suite,
    )
    self.assertTrue(res.passed_gate)
    self.assertTrue(any("improved" in r.lower() for r in res.decision_reasons))

  def test_evaluate_benchmark_suite_regression_fails(self):
    suite = benchmarks.generate_synthetic_benchmark_suite(n_samples_per_domain=20)

    # Candidate has severe regression on historical 'iter_like' domain
    def candidate_eval(inputs):
      preds = {
          "efe_gb": inputs["efe_gb"] * 2.0,
          "efi_gb": inputs["efi_gb"] * 2.0,
          "pfi_gb": inputs["pfi_gb"] * 2.0,
      }
      return preds, None

    # Champion is accurate
    def champion_eval(inputs):
      preds = {
          "efe_gb": inputs["efe_gb"],
          "efi_gb": inputs["efi_gb"],
          "pfi_gb": inputs["pfi_gb"],
      }
      return preds, None

    res = benchmarks.evaluate_benchmark_suite(
        candidate_fn=candidate_eval,
        champion_fn=champion_eval,
        benchmark_suite=suite,
        max_allowed_regression_pct=2.0,
    )
    self.assertFalse(res.passed_gate)
    self.assertTrue(any("regression" in r.lower() for r in res.decision_reasons))


if __name__ == "__main__":
  absltest.main()

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
"""Multi-domain physics benchmarks and gating evaluation for surrogate models."""

from collections.abc import Mapping
import dataclasses
from typing import Any, Protocol

import numpy as np


class SurrogateEvaluator(Protocol):
  """Protocol for surrogate model evaluation in benchmark suite."""

  def __call__(
      self, inputs: Mapping[str, np.ndarray]
  ) -> tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray] | None]:
    """Returns (predictions, uncertainties)."""
    ...


@dataclasses.dataclass(frozen=True)
class DomainMetricResult:
  """Metric evaluation results for a single domain benchmark slice."""

  mae: dict[str, float]
  rmse: dict[str, float]
  mean_uncertainty: dict[str, float]
  num_samples: int


@dataclasses.dataclass(frozen=True)
class BenchmarkComparisonResult:
  """Comparison results across domain slices between candidate and champion models."""

  candidate_metrics: dict[str, DomainMetricResult]
  champion_metrics: dict[str, DomainMetricResult]
  passed_gate: bool
  decision_reasons: list[str]


def compute_metrics(
    predictions: Mapping[str, np.ndarray],
    ground_truth: Mapping[str, np.ndarray],
    uncertainties: Mapping[str, np.ndarray] | None = None,
) -> DomainMetricResult:
  """Calculates MAE, RMSE, and mean uncertainty across channels."""
  mae_dict = {}
  rmse_dict = {}
  unc_dict = {}

  channels = list(ground_truth.keys())
  num_samples = len(ground_truth[channels[0]]) if channels else 0

  for ch in channels:
    if ch in predictions:
      pred = np.asarray(predictions[ch])
      true = np.asarray(ground_truth[ch])
      err = pred - true
      mae_dict[ch] = float(np.mean(np.abs(err)))
      rmse_dict[ch] = float(np.sqrt(np.mean(err**2)))
    if uncertainties and ch in uncertainties:
      unc_dict[ch] = float(np.mean(np.asarray(uncertainties[ch])))

  return DomainMetricResult(
      mae=mae_dict,
      rmse=rmse_dict,
      mean_uncertainty=unc_dict,
      num_samples=num_samples,
  )


def generate_synthetic_benchmark_suite(
    n_samples_per_domain: int = 100,
    seed: int = 42,
) -> dict[str, dict[str, np.ndarray]]:
  """Generates synthetic physics data slices representing distinct tokamak regimes."""
  rng = np.random.default_rng(seed)
  suite = {}

  # 1. ITER-like conventional aspect ratio
  iter_rlns = rng.uniform(1.0, 4.0, n_samples_per_domain)
  iter_rlts = rng.uniform(4.0, 9.0, n_samples_per_domain)
  iter_q = rng.uniform(1.1, 3.5, n_samples_per_domain)
  iter_shat = rng.uniform(0.2, 1.5, n_samples_per_domain)
  iter_kappa = rng.uniform(1.6, 1.8, n_samples_per_domain)
  iter_delta = rng.uniform(0.3, 0.4, n_samples_per_domain)
  suite["iter_like"] = {
      "RLNS_1": iter_rlns,
      "RLTS_1": iter_rlts,
      "RLTS_2": iter_rlts,
      "Q_LOC": iter_q,
      "SHAT": iter_shat,
      "KAPPA_LOC": iter_kappa,
      "DELTA_LOC": iter_delta,
      "efi_gb": 0.5 * iter_rlts + 0.2 * iter_q,
      "efe_gb": 0.4 * iter_rlts + 0.1 * iter_q,
      "pfi_gb": 0.1 * iter_rlns,
  }

  # 2. STEP-like spherical tokamak (low aspect ratio, strong shaping)
  step_rlns = rng.uniform(2.0, 6.0, n_samples_per_domain)
  step_rlts = rng.uniform(6.0, 14.0, n_samples_per_domain)
  step_q = rng.uniform(2.0, 6.0, n_samples_per_domain)
  step_shat = rng.uniform(0.5, 3.0, n_samples_per_domain)
  step_kappa = rng.uniform(2.4, 3.0, n_samples_per_domain)
  step_delta = rng.uniform(0.5, 0.7, n_samples_per_domain)
  suite["step_like"] = {
      "RLNS_1": step_rlns,
      "RLTS_1": step_rlts,
      "RLTS_2": step_rlts,
      "Q_LOC": step_q,
      "SHAT": step_shat,
      "KAPPA_LOC": step_kappa,
      "DELTA_LOC": step_delta,
      "efi_gb": 0.6 * step_rlts + 0.3 * step_q,
      "efe_gb": 0.5 * step_rlts + 0.15 * step_q,
      "pfi_gb": 0.15 * step_rlns,
  }

  # 3. Newly harvested active-learning data slice
  harv_rlns = rng.uniform(1.5, 5.0, n_samples_per_domain)
  harv_rlts = rng.uniform(5.0, 12.0, n_samples_per_domain)
  harv_q = rng.uniform(1.5, 4.5, n_samples_per_domain)
  suite["harvested_recent"] = {
      "RLNS_1": harv_rlns,
      "RLTS_1": harv_rlts,
      "RLTS_2": harv_rlts,
      "Q_LOC": harv_q,
      "SHAT": rng.uniform(0.3, 2.0, n_samples_per_domain),
      "KAPPA_LOC": rng.uniform(1.6, 2.2, n_samples_per_domain),
      "DELTA_LOC": rng.uniform(0.3, 0.5, n_samples_per_domain),
      "efi_gb": 0.55 * harv_rlts + 0.25 * harv_q,
      "efe_gb": 0.45 * harv_rlts + 0.12 * harv_q,
      "pfi_gb": 0.12 * harv_rlns,
  }

  return suite


def evaluate_benchmark_suite(
    candidate_fn: SurrogateEvaluator,
    champion_fn: SurrogateEvaluator,
    benchmark_suite: Mapping[str, Mapping[str, np.ndarray]],
    max_allowed_regression_pct: float = 3.0,
    harvested_key: str = "harvested_recent",
) -> BenchmarkComparisonResult:
  """Compares candidate against champion across all domain slices and applies gate criteria.

  Args:
    candidate_fn: Function or model providing (preds, uncs) for candidate.
    champion_fn: Function or model providing (preds, uncs) for current production champion.
    benchmark_suite: Mapping of domain_name -> dataset dictionary.
    max_allowed_regression_pct: Maximum allowed % error increase on historical domains.
    harvested_key: Identifier of the newly harvested domain slice.

  Returns:
    BenchmarkComparisonResult with gate status and detailed explanation.
  """
  candidate_metrics = {}
  champion_metrics = {}
  reasons = []
  passed_gate = True

  output_keys = ["efe_gb", "efi_gb", "pfi_gb"]

  for domain_name, data in benchmark_suite.items():
    # Ground truth targets
    gt = {k: data[k] for k in output_keys if k in data}

    cand_preds, cand_uncs = candidate_fn(data)
    champ_preds, champ_uncs = champion_fn(data)

    cand_res = compute_metrics(cand_preds, gt, cand_uncs)
    champ_res = compute_metrics(champ_preds, gt, champ_uncs)

    candidate_metrics[domain_name] = cand_res
    champion_metrics[domain_name] = champ_res

    # Gating checks:
    # 1. Historical slices must not regress beyond tolerance
    for ch in gt.keys():
      cand_rmse = cand_res.rmse.get(ch, float("inf"))
      champ_rmse = champ_res.rmse.get(ch, float("inf"))

      if champ_rmse > 1e-8:
        rel_change_pct = ((cand_rmse - champ_rmse) / champ_rmse) * 100.0
      elif cand_rmse > 1e-8:
        rel_change_pct = float("inf")
      else:
        rel_change_pct = 0.0

      if domain_name != harvested_key:
        if rel_change_pct > max_allowed_regression_pct:
          passed_gate = False
          reasons.append(
              f"Regression on historical domain '{domain_name}', channel '{ch}': "
              f"+{rel_change_pct:.2f}% RMSE (limit: {max_allowed_regression_pct}%)"
          )

    # 2. On harvested data, candidate should improve or match champion
    if domain_name == harvested_key:
      cand_total_rmse = sum(cand_res.rmse.values())
      champ_total_rmse = sum(champ_res.rmse.values())
      if cand_total_rmse > champ_total_rmse:
        passed_gate = False
        reasons.append(
            f"Candidate total RMSE on '{harvested_key}' ({cand_total_rmse:.4f}) "
            f"did not beat champion ({champ_total_rmse:.4f})"
        )
      else:
        reasons.append(
            f"Candidate improved total RMSE on '{harvested_key}' from "
            f"{champ_total_rmse:.4f} to {cand_total_rmse:.4f}"
        )

  if passed_gate and not reasons:
    reasons.append("All multi-domain benchmark slices passed gating criteria.")

  return BenchmarkComparisonResult(
      candidate_metrics=candidate_metrics,
      champion_metrics=champion_metrics,
      passed_gate=passed_gate,
      decision_reasons=reasons,
  )

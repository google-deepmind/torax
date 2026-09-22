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
"""MLOps retraining and benchmarking pipeline for physics surrogates."""

from torax._src.mlops.benchmarks import BenchmarkComparisonResult
from torax._src.mlops.benchmarks import compute_metrics
from torax._src.mlops.benchmarks import evaluate_benchmark_suite
from torax._src.mlops.benchmarks import generate_synthetic_benchmark_suite
from torax._src.mlops.train_surrogate import ingest_harvested_dataset
from torax._src.mlops.train_surrogate import retrain_and_gate
from torax._src.mlops.train_surrogate import RetrainResult
from torax._src.mlops.train_surrogate import train_surrogate_model
from torax._src.mlops.watch_and_retrain import run_watch_loop
from torax._src.mlops.watch_and_retrain import scan_harvest_dir

__all__ = [
    "BenchmarkComparisonResult",
    "compute_metrics",
    "evaluate_benchmark_suite",
    "generate_synthetic_benchmark_suite",
    "ingest_harvested_dataset",
    "retrain_and_gate",
    "RetrainResult",
    "train_surrogate_model",
    "run_watch_loop",
    "scan_harvest_dir",
]

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
"""Public API for TORAX MLOps active learning pipeline."""

from torax._src.mlops import BenchmarkComparisonResult
from torax._src.mlops import compute_metrics
from torax._src.mlops import evaluate_benchmark_suite
from torax._src.mlops import generate_synthetic_benchmark_suite
from torax._src.mlops import ingest_harvested_dataset
from torax._src.mlops import retrain_and_gate
from torax._src.mlops import RetrainResult
from torax._src.mlops import run_watch_loop
from torax._src.mlops import scan_harvest_dir
from torax._src.mlops import train_surrogate_model

__all__ = [
    "BenchmarkComparisonResult",
    "compute_metrics",
    "evaluate_benchmark_suite",
    "generate_synthetic_benchmark_suite",
    "ingest_harvested_dataset",
    "retrain_and_gate",
    "RetrainResult",
    "run_watch_loop",
    "scan_harvest_dir",
    "train_surrogate_model",
]

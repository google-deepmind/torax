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
"""Decoupled file watcher and scheduler daemon for active learning retraining."""

import argparse
import glob
import os
import subprocess
import time

from absl import logging
import pandas as pd
from torax._src.mlops import train_surrogate


def scan_harvest_dir(
    harvest_dir: str,
    solver_fingerprint: str | None = None,
) -> tuple[int, list[str]]:
  """Scans harvest_dir for unarchived parquet/npz files and counts samples."""
  if not os.path.exists(harvest_dir):
    return 0, []

  parquet_files = glob.glob(os.path.join(harvest_dir, "*.parquet"))
  npz_files = glob.glob(os.path.join(harvest_dir, "*.npz"))
  all_files = parquet_files + npz_files

  total_samples = 0
  matching_files = []

  for f in parquet_files:
    try:
      df = pd.read_parquet(f)
      if solver_fingerprint and "fingerprint" in df.columns:
        df = df[df["fingerprint"] == solver_fingerprint]
      count = len(df)
      if count > 0:
        total_samples += count
        matching_files.append(f)
    except Exception as e:
      logging.warning("Error reading parquet file %s: %s", f, e)

  for f in npz_files:
    try:
      data = pd.DataFrame(dict(np.load(f)))
      if solver_fingerprint and "fingerprint" in data.columns:
        data = data[data["fingerprint"] == solver_fingerprint]
      count = len(data)
      if count > 0:
        total_samples += count
        matching_files.append(f)
    except Exception as e:
      logging.warning("Error reading npz file %s: %s", f, e)

  return total_samples, matching_files


def run_watch_loop(
    harvest_dir: str = "/tmp/torax_harvest",
    output_dir: str = "/tmp/torax_models",
    solver_fingerprint: str = "tglf_sat1",
    min_samples: int = 500,
    max_interval_seconds: float = 86400,  # 24 hours
    poll_interval_seconds: float = 10,
    slurm_script: str | None = None,
    single_pass: bool = False,
) -> bool:
  """Polls harvest directory and triggers retraining when criteria are satisfied.

  Returns:
    True if a retraining run was executed, False otherwise.
  """
  logging.info(
      "Starting active learning watcher on %s (min_samples=%d, max_interval=%.1fs)",
      harvest_dir,
      min_samples,
      max_interval_seconds,
  )

  last_retrain_time = time.time()
  retrained_any = False

  while True:
    total_samples, staged_files = scan_harvest_dir(
        harvest_dir, solver_fingerprint
    )
    elapsed = time.time() - last_retrain_time

    trigger_by_count = total_samples >= min_samples
    trigger_by_time = (elapsed >= max_interval_seconds) and (total_samples > 0)

    if trigger_by_count or trigger_by_time:
      logging.info(
          "Retraining triggered (samples=%d, elapsed=%.1fs). Trigger mode: %s",
          total_samples,
          elapsed,
          "count" if trigger_by_count else "timeout",
      )

      if slurm_script and os.path.exists(slurm_script):
        logging.info("Submitting SLURM job via %s", slurm_script)
        cmd = [
            "sbatch",
            slurm_script,
            "--harvest_dir",
            harvest_dir,
            "--output_dir",
            output_dir,
            "--fingerprint",
            solver_fingerprint,
        ]
        subprocess.run(cmd, check=True)
      else:
        # In-process retraining
        res = train_surrogate.retrain_and_gate(
            harvest_dir=harvest_dir,
            output_dir=output_dir,
            solver_fingerprint=solver_fingerprint,
        )
        logging.info(
            "Retraining finished. Model: %s, Passed Gate: %s",
            res.model_version,
            res.passed_gate,
        )

      last_retrain_time = time.time()
      retrained_any = True

    if single_pass:
      break

    time.sleep(poll_interval_seconds)

  return retrained_any


def main():
  parser = argparse.ArgumentParser(
      description="Decoupled active learning watcher daemon"
  )
  parser.add_argument("--harvest_dir", type=str, default="/tmp/torax_harvest")
  parser.add_argument("--output_dir", type=str, default="/tmp/torax_models")
  parser.add_argument("--fingerprint", type=str, default="tglf_sat1")
  parser.add_argument("--min_samples", type=int, default=500)
  parser.add_argument(
      "--max_interval_seconds", type=float, default=86400
  )  # 24h
  parser.add_argument("--poll_interval_seconds", type=float, default=10)
  parser.add_argument("--slurm_script", type=str, default=None)
  parser.add_argument("--single_pass", action="store_true")

  args = parser.parse_args()
  run_watch_loop(
      harvest_dir=args.harvest_dir,
      output_dir=args.output_dir,
      solver_fingerprint=args.fingerprint,
      min_samples=args.min_samples,
      max_interval_seconds=args.max_interval_seconds,
      poll_interval_seconds=args.poll_interval_seconds,
      slurm_script=args.slurm_script,
      single_pass=args.single_pass,
  )


if __name__ == "__main__":
  main()

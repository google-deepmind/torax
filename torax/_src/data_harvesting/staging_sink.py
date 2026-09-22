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
"""In-situ local staging sink for active learning data harvesting."""

from collections.abc import Mapping
import dataclasses
import os
import time
from typing import Any
import uuid

import numpy as np


@dataclasses.dataclass(frozen=True)
class HarvestSample:
  """A single time-step batch of harvested input-output data pairs."""

  fingerprint: str
  inputs: Mapping[str, np.ndarray]
  outputs: Mapping[str, np.ndarray]
  uncertainties: Mapping[str, np.ndarray] | None = None
  metadata: Mapping[str, Any] | None = None


class StagingSink:
  """Buffers harvested physics evaluations and writes them to local staging files."""

  def __init__(
      self,
      output_dir: str = "/tmp/torax_harvest",
      run_id: str | None = None,
      max_buffer_size: int = 1000,
  ):
    self.output_dir = output_dir
    self.run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
    self.max_buffer_size = max_buffer_size
    self._buffer: list[HarvestSample] = []
    os.makedirs(self.output_dir, exist_ok=True)

  def record(self, sample: HarvestSample) -> None:
    """Records a harvested sample into the in-memory buffer."""
    self._buffer.append(sample)
    if len(self._buffer) >= self.max_buffer_size:
      self.flush()

  def flush(self) -> str | None:
    """Flushes buffered samples to a staging Parquet file.

    Returns:
      The path to the written file, or None if the buffer was empty.
    """
    if not self._buffer:
      return None

    samples_to_write = list(self._buffer)
    self._buffer.clear()

    # Aggregate tabular rows across samples
    rows: list[dict[str, Any]] = []
    fingerprint = samples_to_write[0].fingerprint

    for sample in samples_to_write:
      # Determine number of radial face evaluation points
      sample_keys = list(sample.inputs.keys())
      if not sample_keys:
        continue
      n_faces = len(sample.inputs[sample_keys[0]])

      for i in range(n_faces):
        row: dict[str, Any] = {
            "run_id": self.run_id,
            "fingerprint": sample.fingerprint,
            "face_index": i,
            "timestamp": time.time(),
        }
        for k, v in sample.inputs.items():
          val = v[i] if hasattr(v, "__getitem__") else v
          row[f"in_{k}"] = float(val)
        for k, v in sample.outputs.items():
          val = v[i] if hasattr(v, "__getitem__") else v
          row[f"out_{k}"] = float(val)
        if sample.uncertainties:
          for k, v in sample.uncertainties.items():
            val = v[i] if hasattr(v, "__getitem__") else v
            row[f"unc_{k}"] = float(val)
        rows.append(row)

    if not rows:
      return None

    timestamp_str = int(time.time() * 1000)
    file_path = os.path.join(
        self.output_dir,
        f"harvest_{fingerprint}_{self.run_id}_{timestamp_str}.parquet",
    )

    try:
      import pandas as pd
      import pyarrow as pa
      import pyarrow.parquet as pq

      df = pd.DataFrame(rows)
      table = pa.Table.from_pandas(df)
      pq.write_table(table, file_path)
      return file_path
    except Exception:  # Fallback to npz if pyarrow/pandas has an issue
      npz_path = file_path.replace(".parquet", ".npz")
      keys = list(rows[0].keys())
      data_dict = {
          k: np.array([r[k] for r in rows])
          for k in keys
          if isinstance(rows[0][k], (int, float))
      }
      np.savez_compressed(npz_path, **data_dict)
      return npz_path

  def finalize(self) -> str | None:
    """Flushes any remaining samples upon simulation completion."""
    return self.flush()

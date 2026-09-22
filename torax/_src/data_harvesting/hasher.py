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
"""Deterministic solver settings hasher for dataset grouping and model provenance."""

from collections.abc import Mapping
import hashlib
import json
from typing import Any


def _normalize_value(val: Any) -> Any:
  """Normalizes values for deterministic JSON serialization."""
  if isinstance(val, float):
    # Round float to 8 decimal places to avoid IEEE-754 precision noise across platforms
    return round(val, 8)
  if isinstance(val, (int, str, bool)) or val is None:
    return val
  if isinstance(val, Mapping):
    return {k: _normalize_value(v) for k, v in sorted(val.items())}
  if isinstance(val, (list, tuple)):
    return [_normalize_value(v) for v in val]
  return str(val)


def compute_solver_fingerprint(
    model_name: str,
    settings: Mapping[str, Any],
    species: Mapping[str, Any] | None = None,
) -> str:
  """Computes a deterministic SHA-256 fingerprint for a solver configuration.

  Args:
    model_name: Base physics code or model name (e.g., 'tglf', 'qlknn').
    settings: Dictionary of numeric settings, flags, and physical options.
    species: Optional dictionary of ion species definitions (Z, A, masses).

  Returns:
    A string fingerprint of format '{model_name}_{hash12}'.
  """
  canonical_payload = {
      'model_name': model_name.strip().lower(),
      'settings': _normalize_value(settings),
      'species': _normalize_value(species or {}),
  }
  payload_json = json.dumps(canonical_payload, sort_keys=True)
  sha256_hash = hashlib.sha256(payload_json.encode('utf-8')).hexdigest()
  return f'{model_name.strip().lower()}_{sha256_hash[:12]}'

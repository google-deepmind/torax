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
"""Domain-agnostic Active Learning and Adaptive Physics Module architecture."""

from collections.abc import Callable, Mapping
import dataclasses
from typing import Any, Literal

import numpy as np
from torax._src.data_harvesting import HarvestSample
from torax._src.data_harvesting import StagingSink

FallbackMode = Literal["full_profile", "per_face"]


def apply_spatial_smoothing(
    values: np.ndarray,
    coords: np.ndarray,
    sigma: float,
) -> np.ndarray:
  """Applies a normalized 1D Gaussian smoothing filter to spatial profile values.

  Args:
    values: 1D array of values across radial faces or cells.
    coords: 1D coordinate array (e.g. rho_face_norm).
    sigma: Gaussian smoothing width in coordinate units. If <= 0, returns values unchanged.

  Returns:
    Smoothed 1D array of identical shape.
  """
  if sigma <= 0.0 or len(values) <= 1:
    return values

  # Compute pairwise squared distances: (N, N)
  diffs = coords[:, None] - coords[None, :]
  weights = np.exp(-0.5 * (diffs / sigma) ** 2)
  # Normalize rows
  weights /= np.sum(weights, axis=1, keepdims=True)
  return weights @ values


@dataclasses.dataclass(frozen=True)
class AdaptivePhysicsConfig:
  """Configuration parameters for adaptive physics surrogate execution."""

  uncertainty_threshold: float = 0.20
  fallback_mode: FallbackMode = "full_profile"
  smoothing_sigma: float = 0.05
  enable_data_harvesting: bool = True


class AdaptivePhysicsEngine:
  """Coordinates surrogate inference, uncertainty gating, fallback execution, and harvesting."""

  def __init__(
      self,
      config: AdaptivePhysicsConfig,
      sink: StagingSink | None = None,
  ):
    self.config = config
    self.sink = sink

  def decide_fallback(
      self,
      relative_uncertainty: np.ndarray,
  ) -> tuple[bool, np.ndarray]:
    """Determines whether fallback is triggered and produces the execution mask.

    Args:
      relative_uncertainty: 1D array of uncertainty estimates (e.g. sigma / |mu|).

    Returns:
      needs_fallback: True if any face requires high-fidelity solver execution.
      run_mask: Boolean 1D array indicating which radial points require high-fidelity solver.
    """
    exceeds = relative_uncertainty > self.config.uncertainty_threshold
    any_exceeds = bool(np.any(exceeds))

    if not any_exceeds:
      return False, np.zeros_like(relative_uncertainty, dtype=bool)

    if self.config.fallback_mode == "full_profile":
      return True, np.ones_like(relative_uncertainty, dtype=bool)
    elif self.config.fallback_mode == "per_face":
      return True, exceeds
    else:
      raise ValueError(f"Unknown fallback mode: {self.config.fallback_mode}")

  def fuse_and_smooth(
      self,
      surrogate_vals: np.ndarray,
      high_fidelity_vals: np.ndarray,
      run_mask: np.ndarray,
      coords: np.ndarray,
  ) -> np.ndarray:
    """Combines surrogate and high-fidelity evaluations and smooths across boundaries."""
    if self.config.fallback_mode == "full_profile":
      return high_fidelity_vals

    # Splicing in per_face mode
    fused = np.where(run_mask, high_fidelity_vals, surrogate_vals)
    return apply_spatial_smoothing(fused, coords, self.config.smoothing_sigma)

  def harvest_if_enabled(
      self,
      fingerprint: str,
      inputs: Mapping[str, np.ndarray],
      high_fidelity_outputs: Mapping[str, np.ndarray],
      uncertainties: Mapping[str, np.ndarray] | None = None,
      metadata: Mapping[str, Any] | None = None,
  ) -> None:
    """Dispatches evaluated input-output pairs to the data harvesting sink."""
    if not self.config.enable_data_harvesting or self.sink is None:
      return

    sample = HarvestSample(
        fingerprint=fingerprint,
        inputs=inputs,
        outputs=high_fidelity_outputs,
        uncertainties=uncertainties,
        metadata=metadata,
    )
    self.sink.record(sample)

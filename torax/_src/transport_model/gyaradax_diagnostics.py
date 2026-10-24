# Copyright 2026 DeepMind Technologies Limited
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

"""Optional host-side diagnostics for the gyaradax transport models.

The gyaradax models run a gyrokinetic solve (or a diffusion sampler) inside
TORAX's jitted PDE step, so nothing about a transport call is visible from
Python once a simulation is running. This module records per-call, per-radius
diagnostics without changing the traced graph: every quantity is handed to a
host-side writer through `jax.experimental.io_callback`.

Two outputs, both off unless the transport config names a location:

* `diagnostics_path` -> one JSON object per (transport call, radius) appended
  to a `.jsonl` file: the local parameter / conditioning vector, the fluxes
  *before* the gyroBohm unit conversion, the iterations or steps actually
  used, the per-sample flux spread for the diffusion model, and wallclock.
* `latent_dump_dir` -> one `.npz` per (transport call, radius) holding the
  gyroflow latents `euler_sample` returned (pre-decoder) together with the
  conditioning vector, the radius and the fluxes those latents decoded to, so
  a latent can always be traced back to the flux it produced.

Both are emitted by the *same* callback per radius, so the npz filename in a
jsonl row always refers to the latents that row's flux came from.

Ordering: TORAX vmaps the per-radius solve over `rho_match`, and JAX refuses
to batch an *ordered* io_callback, so the callbacks here are unordered. The
host recovers the (call, radius) labelling from the arrival order and the
statically known number of radii per call, which is exact because a vmapped
unordered callback fires once per batch element per call.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import jax
from jax.experimental import io_callback
import jax.numpy as jnp
import numpy as np

_SINKS: Dict[Tuple[str, str, int], "DiagnosticsSink"] = {}


def _to_native(value: Any) -> Any:
  """numpy/jax leaf -> json-serializable scalar or nested list."""
  arr = np.asarray(value)
  return arr.item() if arr.ndim == 0 else arr.tolist()


@dataclasses.dataclass(eq=False)
class DiagnosticsSink:
  """Host-side writer for one transport model's diagnostics and latents."""

  diagnostics_path: str
  latent_dump_dir: str
  n_radii: int
  latent_dump_max_calls: int = 0
  latent_dump_decoded_max_calls: int = 0
  n_rows: int = 0
  # perf_counter stamp left by the entry marker of the call being recorded
  t_begin: Optional[float] = None

  def __post_init__(self):
    if self.diagnostics_path:
      parent = os.path.dirname(os.path.abspath(self.diagnostics_path))
      os.makedirs(parent, exist_ok=True)
      # truncate: a stale file from a previous run would corrupt the labelling
      open(self.diagnostics_path, "w").close()
    if self.latent_dump_dir:
      os.makedirs(self.latent_dump_dir, exist_ok=True)

  @property
  def call_index(self) -> int:
    return self.n_rows // max(self.n_radii, 1)

  @property
  def radius_index(self) -> int:
    return self.n_rows % max(self.n_radii, 1)

  def mark_begin(self, _rho_idx) -> np.ndarray:
    """Entry marker: stamp the clock that `wall_s` is measured from."""
    if self.radius_index == 0:
      self.t_begin = time.perf_counter()
    return np.int32(0)

  def write(
      self,
      payload: Dict[str, Any],
      latents: Optional[Dict[str, Any]] = None,
      static: Optional[Dict[str, Any]] = None,
  ) -> None:
    """Write one (call, radius) latent npz (optional) and jsonl row."""
    now = time.perf_counter()
    call, radius = self.call_index, self.radius_index
    row: Dict[str, Any] = {
        "call": call,
        "radius": radius,
        "wall_s": None if self.t_begin is None else now - self.t_begin,
    }
    row.update(static or {})
    if latents and self.latent_dump_dir:
      path = self._dump_latents(call, radius, latents, payload)
      if path is not None:
        row["latents_file"] = os.path.basename(path)
    row.update({k: _to_native(v) for k, v in payload.items()})
    if self.diagnostics_path:
      with open(self.diagnostics_path, "a") as f:
        f.write(json.dumps(row) + "\n")
    self.n_rows += 1

  def _dump_latents(self, call, radius, latents, payload) -> Optional[str]:
    if self.latent_dump_max_calls and call >= self.latent_dump_max_calls:
      return None
    if self.latent_dump_decoded_max_calls and call >= self.latent_dump_decoded_max_calls:
      latents = {k: v for k, v in latents.items() if not k.startswith("df_norm")}
    path = os.path.join(
        self.latent_dump_dir, f"latents_call{call:05d}_rho{radius:02d}.npz"
    )
    arrays = {k: np.asarray(v) for k, v in latents.items()}
    arrays.update({k: np.asarray(v) for k, v in payload.items()})
    nbytes = sum(a.nbytes for a in arrays.values())
    # turbulence fields do not compress, and zipping GBs of them is slow
    save = np.savez if nbytes > 64 * 1024**2 else np.savez_compressed
    save(path, call=call, radius=radius, **arrays)
    return path


def get_sink(
    diagnostics_path: str,
    latent_dump_dir: str,
    n_radii: int,
    latent_dump_max_calls: int = 0,
    latent_dump_decoded_max_calls: int = 0,
) -> Optional[DiagnosticsSink]:
  """Process-wide sink for this (paths, n_radii) triple; None when both empty."""
  if not diagnostics_path and not latent_dump_dir:
    return None
  key = (diagnostics_path, latent_dump_dir, n_radii)
  if key not in _SINKS:
    _SINKS[key] = DiagnosticsSink(
        diagnostics_path=diagnostics_path,
        latent_dump_dir=latent_dump_dir,
        n_radii=n_radii,
        latent_dump_max_calls=latent_dump_max_calls,
        latent_dump_decoded_max_calls=latent_dump_decoded_max_calls,
    )
  return _SINKS[key]


def reset_sinks() -> None:
  """Drop every registered sink (tests, and repeated in-process runs)."""
  _SINKS.clear()


def mark_begin(sink: Optional[DiagnosticsSink], rho_idx):
  """Stamp the per-radius block's start clock, returning `rho_idx` unchanged.

  The callback returns an integer zero which is *added* to the index, so every
  downstream value carries a data dependency on it. An `optimization_barrier`
  is not enough here: its unused token output gets eliminated and the marker
  then drifts past the work it is meant to time.
  """
  if sink is None:
    return rho_idx
  idx = jnp.asarray(rho_idx, dtype=jnp.int32)
  token = io_callback(
      sink.mark_begin,
      jax.ShapeDtypeStruct((), jnp.int32),
      idx,
      ordered=False,
  )
  return idx + token


def record(
    sink: Optional[DiagnosticsSink],
    static: Dict[str, Any],
    payload: Dict[str, Any],
    latents: Optional[Dict[str, Any]] = None,
) -> None:
  """Emit one row (and its latent npz). `static` is host-side, rest is traced."""
  if sink is None:
    return
  if not sink.latent_dump_dir:
    latents = None
  io_callback(
      functools.partial(sink.write, static=static),
      (),
      payload,
      latents,
      ordered=False,
  )


def load_jsonl(path: str) -> List[dict]:
  """Read a diagnostics jsonl back into a list of dicts."""
  with open(path) as f:
    return [json.loads(line) for line in f if line.strip()]

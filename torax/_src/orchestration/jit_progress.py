# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Progress bar support for the jitted TORAX run loop.

The jitted run loop executes entirely inside a single compiled
``jax.lax.while_loop``, so the Python-level tqdm progress bar used by the
eager run loop cannot be driven directly from the loop.  Instead, the loop
body emits the current simulation time to the host through
``jax.debug.callback`` and a host-side tqdm bar converts it into a
percentage of the requested simulation interval, mirroring the eager bar
(current simulation time in the description, percent complete as the bar).

Design notes:

* A single module-level ``_active_bar`` slot is updated by a stable,
  module-level callback function (``_host_update``). This keeps the compiled
  program identical across runs; closing over a tqdm object (or a bound method)
  in traced code would embed a fresh Python callable in every trace and defeat
  compilation caching.

* Callbacks are emitted unordered. Unordered callbacks may arrive on the host
  out of program order, so the host bar clamps updates to be monotonically
  non-decreasing, which makes ordering irrelevant for a progress display and
  avoids the cost of ordered callbacks.

* Emission is throttled inside the trace by checking if the step crossed an
  interval boundary (integer bucket check on previous vs. current time). This
  bounds the number of host callbacks (and device-to-host transfers) to roughly
  100 per simulation, independent of the number of solver steps, without needing
  to carry any progress state through the while loop.
"""

import math
import threading
from typing import Self

import jax
import jax.numpy as jnp
from torax._src import jax_utils
from tqdm import auto as tqdm


DEFAULT_REPORT_FRACTION = 0.01
_PERCENT_TOTAL = 100.0

_active_bar: "_HostBar | None" = None
_bar_lock = threading.Lock()


class _HostBar:
  """Host-side tqdm wrapper that maps simulation time to percent complete."""

  def __init__(self, t_initial: float, t_final: float):
    self._t_initial = float(t_initial)
    duration = float(t_final) - float(t_initial)
    # Guard against a degenerate interval so we never divide by zero.
    self._duration = duration if duration > 0.0 else 1.0
    self._started = False
    self._closed = False
    self._bar = tqdm.tqdm(
        total=_PERCENT_TOTAL,
        desc="Compiling",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]",
    )

  def report_time(self, t: float) -> None:
    """Advances the bar to simulation time ``t`` (monotonic, clamped)."""
    if self._closed or not math.isfinite(t):
      return
    raw_percent = _PERCENT_TOTAL * (t - self._t_initial) / self._duration
    # Ignore out-of-order callbacks with an earlier time so neither the bar
    # fill nor the simulation time description moves backwards.
    if self._started and raw_percent < self._bar.n:
      return
    percent = min(max(raw_percent, self._bar.n), _PERCENT_TOTAL)
    self._started = True
    self._bar.set_description(f"Simulating (t={t:.5f})", refresh=False)
    # tqdm handles terminal redraw throttling via its own mininterval.
    self._bar.update(percent - self._bar.n)

  def close(self, final_time: float | None = None) -> None:
    """Performs a final update (if a time is known) and closes the bar.

    Args:
      final_time: The actual final time of the returned state, so that a run
        terminated early (error state or max_steps) honestly leaves the bar
        partial instead of jumping to completion.
    """
    if self._closed:
      return
    if final_time is not None:
      self.report_time(float(final_time))
    self._bar.close()
    self._closed = True


def _host_update(t: jax.Array | float) -> None:
  """Module-level callback target; identity is stable across runs."""
  with _bar_lock:
    if _active_bar is not None:
      _active_bar.report_time(float(t))


def emit_progress(
    t: jax.Array,
    previous_t: jax.Array,
    t_initial: jax.Array | float,
    report_interval: jax.Array | float,
) -> None:
  """Reports simulation time ``t`` to the host bar if an interval boundary was crossed.

  Call this from inside the while loop body after the state has been
  advanced.

  Args:
    t: Current simulation time (traced scalar).
    previous_t: Simulation time before this step (traced scalar).
    t_initial: Initial simulation time (traced or concrete scalar).
    report_interval: Minimum advance in simulation time between reports (traced
      or concrete scalar; typically one percent of the interval).
  """
  dtype = jax_utils.get_dtype()
  t_initial = jnp.asarray(t_initial, dtype=dtype)
  report_interval = jnp.asarray(report_interval, dtype=dtype)
  prev_bucket = jnp.floor((previous_t - t_initial) / report_interval)
  curr_bucket = jnp.floor((t - t_initial) / report_interval)
  should_emit = curr_bucket > prev_bucket

  def _emit():
    jax.debug.callback(_host_update, t)

  def _skip():
    pass

  # Debug callbacks under lax.cond execute only when their branch is taken.
  jax.lax.cond(should_emit, _emit, _skip)


class JitProgressBar:
  """Context manager owning the host bar for one jitted simulation run.

  Usage in the non-traced wrapper around the compiled loop::

      with JitProgressBar(t_initial, t_final, enabled=progress_bar) as pbar:
        outputs = compiled_loop(..., progress_bar=progress_bar)
        pbar.finalize(final_time=float(<final t from outputs>))

  The context manager unregisters the bar and closes it on exit, including
  on exceptions.  ``finalize`` should be called once the outputs are ready
  so the bar reflects the true final simulation time.
  """

  def __init__(
      self,
      t_initial: float,
      t_final: float,
      report_fraction: float = DEFAULT_REPORT_FRACTION,
      enabled: bool = True,
  ):
    self._t_initial = float(t_initial)
    self._t_final = float(t_final)
    self._report_fraction = (
        float(report_fraction)
        if report_fraction > 0.0
        else DEFAULT_REPORT_FRACTION
    )
    self._enabled = enabled
    self._host_bar: _HostBar | None = None
    self._final_time: float | None = None

  @property
  def report_interval(self) -> float:
    """The minimum simulation time advance between host callbacks."""
    duration = self._t_final - self._t_initial
    return duration * self._report_fraction if duration > 0.0 else 1.0

  def finalize(self, final_time: float) -> None:
    """Records the actual final simulation time for the closing update."""
    self._final_time = final_time

  def __enter__(self) -> Self:
    global _active_bar
    if self._enabled:
      self._host_bar = _HostBar(self._t_initial, self._t_final)
      with _bar_lock:
        _active_bar = self._host_bar
    return self

  def __exit__(self, exc_type, exc_value, traceback) -> None:
    global _active_bar
    if not self._enabled:
      return
    try:
      # Flush any callbacks still in flight before touching the bar from
      # the host thread.
      jax.effects_barrier()
    finally:
      with _bar_lock:
        if _active_bar is self._host_bar:
          _active_bar = None
        if self._host_bar is not None:
          # On an exception we close without a final update (passing None) so
          # the bar shows where the simulation actually got to.
          self._host_bar.close(None if exc_type else self._final_time)
          self._host_bar = None

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
"""

import dataclasses
import itertools
import threading
from typing import Self

import jax
import jax.numpy as jnp
from tqdm import auto as tqdm


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class JitProgressParams:
  """Dynamic parameters for progress reporting inside a jitted function.

  This is a lightweight PyTree holding the dynamic array data needed by the
  compiled loop to emit progress callbacks, without embedding host-side
  objects (such as tqdm bars or locks) into the trace.
  """

  bar_id: jax.Array
  report_interval: jax.Array


_PERCENT_TOTAL = 100.0

_registry: dict[int, "_HostBar"] = {}
_registry_lock = threading.Lock()
_id_counter = itertools.count()


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
    if self._closed:
      return
    percent = _PERCENT_TOTAL * (t - self._t_initial) / self._duration
    # Clamp: never move backwards (unordered callbacks) and never exceed
    # 100 percent (final step may overshoot t_final).
    percent = min(max(percent, self._bar.n), _PERCENT_TOTAL)
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


def _host_update(bar_id, t) -> None:
  """Module-level callback target; identity is stable across runs."""
  bar = _registry.get(int(bar_id))
  if bar is not None:
    bar.report_time(float(t))


def emit_progress(
    bar_id: jax.Array,
    t: jax.Array,
    previous_t: jax.Array,
    t_initial: jax.Array | float,
    report_interval: jax.Array | float,
) -> None:
  """Reports simulation time ``t`` to the host bar if an interval boundary was crossed.

  Call this from inside the while loop body after the state has been
  advanced.

  Args:
    bar_id: Traced int32 scalar identifying the host bar (from
      ``JitProgressBar.id_array``).
    t: Current simulation time (traced scalar).
    previous_t: Simulation time before this step (traced scalar).
    t_initial: Initial simulation time (traced or concrete scalar).
    report_interval: Minimum advance in simulation time between reports (traced
      or concrete scalar; typically one percent of the interval).
  """
  prev_bucket = jnp.floor((previous_t - t_initial) / report_interval)
  curr_bucket = jnp.floor((t - t_initial) / report_interval)
  should_emit = curr_bucket > prev_bucket

  def _emit():
    jax.debug.callback(_host_update, bar_id, t)

  def _skip():
    pass

  # Debug callbacks under lax.cond execute only when their branch is taken.
  jax.lax.cond(should_emit, _emit, _skip)


class JitProgressBar:
  """Context manager owning the host bar for one jitted simulation run.

  Usage in the non-traced wrapper around the compiled loop::

      with JitProgressBar(t_initial, t_final) as pbar:
        outputs = compiled_loop(..., pbar.id_array)
        jax.block_until_ready(outputs)
        pbar.finalize(final_time=float(<final t from outputs>))

  The context manager unregisters the bar and closes it on exit, including
  on exceptions.  ``finalize`` should be called once the outputs are ready
  so the bar reflects the true final simulation time.
  """

  def __init__(
      self,
      t_initial: float,
      t_final: float,
      report_fraction: float = 0.01,
  ):
    self._t_initial = float(t_initial)
    self._t_final = float(t_final)
    self._report_fraction = (
        float(report_fraction) if report_fraction > 0.0 else 0.01
    )
    self._id = next(_id_counter)
    self._host_bar: _HostBar | None = None
    self._final_time: float | None = None

  @property
  def id_array(self) -> jax.Array:
    """The bar identifier as a traced-friendly int32 scalar.

    Pass this as an ordinary argument into the compiled function.  Because
    it is data rather than a Python constant, different runs reuse the same
    compiled program.
    """
    return jnp.asarray(self._id, dtype=jnp.int32)

  @property
  def report_interval(self) -> float:
    """The minimum simulation time advance between host callbacks."""
    duration = self._t_final - self._t_initial
    return duration * self._report_fraction if duration > 0.0 else 1.0

  @property
  def params(self) -> JitProgressParams:
    """The progress parameters as a JAX-friendly PyTree."""
    return JitProgressParams(
        bar_id=self.id_array,
        report_interval=jnp.asarray(self.report_interval),
    )

  def finalize(self, final_time: float) -> None:
    """Records the actual final simulation time for the closing update."""
    self._final_time = final_time

  def __enter__(self) -> Self:
    self._host_bar = _HostBar(self._t_initial, self._t_final)
    with _registry_lock:
      _registry[self._id] = self._host_bar
    return self

  def __exit__(self, exc_type, exc_value, traceback) -> None:
    try:
      # Flush any callbacks still in flight before touching the bar from
      # the host thread.
      jax.effects_barrier()
    finally:
      with _registry_lock:
        _registry.pop(self._id, None)
      if self._host_bar is not None:
        # On an exception we close without a final update (passing None) so the
        # bar shows where the simulation actually got to.
        self._host_bar.close(None if exc_type else self._final_time)
        self._host_bar = None

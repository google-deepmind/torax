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

"""Time step calculator that aligns steps with pellet trigger windows."""

from typing import Any

import jax
from jax import numpy as jnp
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.orchestration import sim_state as sim_state_lib
from torax._src.time_step_calculator import time_step_calculator


class PelletAwareTimeStepCalculator(time_step_calculator.TimeStepCalculator):
  """TimeStepCalculator that resolves pellet trigger and ablation windows.

  The pellet_aware time step calculator ensures that time steps are aligned with
  pellet trigger times and ablation windows.

  It checks the current simulation time against the pellet trigger times
  and ablation duration, and adjusts the time step to ensure that steps
  do not skip over these events.

  The calculator is generic over pellet sources: it reads the 'pellet' source's
  runtime parameters, expecting 'trigger_times' or 'frequency' and
  'ablation_time', and optionally a model-predicted ablation window exposed via a
  'use_model_ablation_time' flag and an 'ablation_step(geo, core_profiles)'
  method.

  Arguments:
    base_calculator: The base time step calculator used away from pellet events
      (for example a chi or fixed calculator).
    trigger_tolerance: Fallback time tolerance for deciding whether the current
      time is at a pellet trigger. The pellet source's own 'trigger_tolerance' is
      used instead when it exposes one, so that the step alignment and the
      source's deposition agree on when a pellet fires.
    window_after_pellet: The duration of the window after a pellet trigger during
      which the time step is adjusted. The duration of the first time step will always
      be equal to the ablation time, the other will be equal to dt_after_pellet.
      Not used by default.
    dt_after_pellet: The time step to use during the window after a pellet trigger.
      If None, the base calculator's time step is used. Not used by default.

    Returns:
      dt: Scalar time step duration.
  """

  def __init__(
      self,
      base_calculator: time_step_calculator.TimeStepCalculator,
      trigger_tolerance: float = 1e-8,
      window_after_pellet: float = 0.0,
      dt_after_pellet: float | None = None,
  ):
    self._base_calculator = base_calculator
    self._trigger_tolerance = float(trigger_tolerance)
    self._window_after_pellet = float(window_after_pellet)
    self._dt_after_pellet = (
        float(dt_after_pellet) if dt_after_pellet is not None else None
    )

  def _next_dt(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      sim_state: sim_state_lib.SimState,
  ) -> jax.Array:
    """Returns a dt aligned with pellet trigger and ablation windows."""
    dt_standard = jnp.asarray(
        self._base_calculator._next_dt(runtime_params, sim_state)
    )
    dtype = dt_standard.dtype

    # A compatible pellet source is guaranteed by the ToraxConfig validator.
    pellet_params = runtime_params.sources.get('pellet')
    trigger_times = getattr(pellet_params, 'trigger_times', None)
    frequency = getattr(pellet_params, 'frequency', None)
    if (trigger_times is None) == (frequency is None):
      raise ValueError(
          "The 'pellet_aware' time step calculator requires the pellet source"
          " to configure exactly one of 'trigger_times' or 'frequency'."
      )

    t = jnp.asarray(sim_state.t, dtype=dtype)
    # Use the pellet source's own trigger tolerance so that the step alignment
    # and the source's deposition agree on when a pellet fires, fall back to the
    # configured tolerance for sources that do not expose one.
    tol = jnp.asarray(
        getattr(pellet_params, 'trigger_tolerance', self._trigger_tolerance),
        dtype=dtype,
    )

    # The whole ablation is resolved as a single step landing on the trigger.
    # A pellet source can predict it via a 'use_model_ablation_time' flag and an
    # 'ablation_step(geo, core_profiles)' method. The model output is only valid
    # at the firing instant, but the calculator detects the trigger itself,
    # so only the returned duration is used. Otherwise the configured constant
    # 'ablation_time' is used.
    ablation_window = jnp.asarray(
        getattr(pellet_params, 'ablation_time'), dtype=dtype
    )
    ablation_step_fn: Any = getattr(pellet_params, 'ablation_step', None)
    use_model_ablation = bool(
        getattr(pellet_params, 'use_model_ablation_time', False)
    ) and callable(ablation_step_fn)
    if use_model_ablation:
      _, model_ablation_time = ablation_step_fn(
          sim_state.geometry, sim_state.core_profiles
      )
      ablation_window = jnp.asarray(model_ablation_time, dtype=dtype)

    if trigger_times is not None:
      dt_trigger, dt_after_trigger, at_trigger = self._dt_for_trigger_times(
          t, dt_standard, trigger_times, ablation_window, tol
      )
    else:
      # frequency is not None here (guaranteed by the check above).
      assert frequency is not None
      frequency_t_start = getattr(pellet_params, 'frequency_t_start', 0.0)
      # Whether the periodic injector is on at t. Defaults to on, so a source
      # that does not expose it keeps firing at every period.
      injection_enabled = getattr(pellet_params, 'injection_enabled', True)
      dt_trigger, dt_after_trigger, at_trigger = self._dt_for_frequency(
          t, dt_standard, frequency, frequency_t_start, injection_enabled,
          ablation_window, tol,
      )

    dt = jnp.minimum(dt_standard, jnp.minimum(dt_trigger, dt_after_trigger))
    # During ablation, never split the window, even if dt_standard is smaller.
    dt = jnp.where(at_trigger, ablation_window, dt)
    return dt

  def _dt_for_trigger_times(
      self,
      t: jax.Array,
      dt_standard: jax.Array,
      trigger_times: tuple[float, ...],
      ablation_window: jax.Array,
      tol: jax.Array,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Aligns dt with an explicit list of pellet trigger times.

    Returns:
      (dt_trigger, dt_after_trigger, at_trigger).
    """
    dtype = dt_standard.dtype
    inf = jnp.asarray(jnp.inf, dtype=dtype)
    triggers = jnp.asarray(trigger_times, dtype=dtype)
    # Earliest trigger strictly after t (inf if none), and most recent trigger
    # at or before t (-inf if none).
    next_trigger = jnp.min(jnp.where(triggers > t - tol, triggers, inf))
    last_trigger = jnp.max(jnp.where(triggers <= t + tol, triggers, -inf))
    # A single step covers the whole ablation window, so we only need to detect
    # the firing instant (should be the same test the pellet source uses to deposit).
    at_trigger = jnp.any(jnp.abs(t - triggers) <= tol)
    delta_to_next_trigger = next_trigger - t
    dt_trigger = jnp.where(at_trigger, ablation_window, delta_to_next_trigger)

    dt_after_trigger = dt_standard
    if self._dt_after_pellet is not None:
      dt_after_pellet = jnp.asarray(self._dt_after_pellet, dtype=dtype)
      window_after_pellet = jnp.asarray(self._window_after_pellet, dtype=dtype)
      post_window_end = last_trigger + window_after_pellet
      in_post_pellet = jnp.logical_and(
          jnp.isfinite(last_trigger),
          jnp.logical_and(
              jnp.logical_and(
                  t > last_trigger + tol, jnp.logical_not(at_trigger)
              ),
              t < post_window_end - tol,
          ),
      )
      dt_after_trigger = jnp.where(
          in_post_pellet,
          jnp.minimum(dt_after_pellet, post_window_end - t),
          dt_standard,
      )
    return dt_trigger, dt_after_trigger, at_trigger

  def _dt_for_frequency(
      self,
      t: jax.Array,
      dt_standard: jax.Array,
      frequency: jax.Array,
      frequency_t_start: float | jax.Array,
      injection_enabled: bool | jax.Array,
      ablation_window: jax.Array,
      tol: jax.Array,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Aligns dt with a periodic pellet injection frequency.

    The frequency is assumed strictly positive (validated by the pellet source
    config). injection_enabled toggles the injector on or off. When off, no
    pellet fires and the base time step is used.

    Returns:
      (dt_trigger, dt_after_trigger, at_trigger).
    """
    dtype = dt_standard.dtype
    inf = jnp.asarray(jnp.inf, dtype=dtype)
    frequency = jnp.asarray(frequency, dtype=dtype)
    frequency_t_start = jnp.asarray(frequency_t_start, dtype=dtype)
    injection_enabled = jnp.asarray(injection_enabled, dtype=bool)
    period = 1.0 / frequency
    # Phase measured from frequency_t_start (consistent with the source).
    phase = jnp.mod(t - frequency_t_start, period)
    # Float rounding can leave phase just below period instead of wrapping
    # to 0 at a pellet time.
    phase = jnp.where(period - phase < tol, jnp.asarray(0.0, dtype=dtype), phase)
    delta_to_next_period = period - phase
    # A single step covers the whole ablation window, so we only need to detect
    # the firing instant (phase wrapped to 0, the same test the source should use).
    # A pellet fires when the step lands within tol of a period boundary.
    # Floating point means the step never lands exactly on the boundary, so a
    # time varying injection_enabled that switches on exactly at a pellet's time
    # can still read "off" at the firing instant. Turn it on a small margin
    # before the intended pellet time.
    at_trigger = jnp.logical_and(injection_enabled, phase <= tol)
    dt_trigger_value = jnp.where(at_trigger, ablation_window, delta_to_next_period)
    dt_trigger = jnp.where(injection_enabled, dt_trigger_value, inf)

    dt_after_trigger = dt_standard
    if self._dt_after_pellet is not None:
      dt_after_pellet = jnp.asarray(self._dt_after_pellet, dtype=dtype)
      window_after_pellet = jnp.asarray(self._window_after_pellet, dtype=dtype)
      in_post_pellet = jnp.logical_and(
          injection_enabled,
          jnp.logical_and(
              jnp.logical_and(phase > tol, jnp.logical_not(at_trigger)),
              phase < window_after_pellet - tol,
          ),
      )
      dt_after_trigger = jnp.where(
          in_post_pellet,
          jnp.minimum(dt_after_pellet, window_after_pellet - phase),
          dt_standard,
      )
    return dt_trigger, dt_after_trigger, at_trigger

  def __eq__(self, other) -> bool:
    return (
        isinstance(other, type(self))
        and self._base_calculator == other._base_calculator
        and self._trigger_tolerance == other._trigger_tolerance
        and self._window_after_pellet == other._window_after_pellet
        and self._dt_after_pellet == other._dt_after_pellet
    )

  def __hash__(self) -> int:
    return hash(
        (
            type(self),
            self._base_calculator,
            self._trigger_tolerance,
            self._window_after_pellet,
            self._dt_after_pellet,
        )
    )
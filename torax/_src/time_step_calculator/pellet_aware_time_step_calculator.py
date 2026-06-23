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

import jax
from jax import numpy as jnp
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.orchestration import sim_state as sim_state_lib
from torax._src.sources import hpi2nn_pellet_source as hpi2nn_pellet_source_lib
from torax._src.time_step_calculator import chi_time_step_calculator
from torax._src.time_step_calculator import fixed_time_step_calculator
from torax._src.time_step_calculator import time_step_calculator


class PelletAwareTimeStepCalculator(time_step_calculator.TimeStepCalculator):
  """TimeStepCalculator that resolves pellet trigger and ablation windows.

  The pellet_aware time step calculator ensures that time steps are aligned with
  pellet trigger times and ablation windows. 

  It checks the current simulation time against the pellet trigger times 
  and ablation duration, and adjusts the time step to ensure that steps 
  do not skip over these events. 
  
  Arguments:
    base_calculator_type: The type of the base time step calculator to use for
      non-pellet-alignment purposes. Must be 'chi' or 'fixed'.
    trigger_tolerance: The time tolerance for determining if the current time is
      at a pellet trigger or ablation boundary.
    pellet_source_name: The name of the pellet source in the runtime parameters.
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
      base_calculator_type: str = 'chi',
      trigger_tolerance: float = 1e-8,
      pellet_source_name: str = 'hpi2nn_pellet_source',
      window_after_pellet: float = 0.0,
      dt_after_pellet: float | None = None,
  ):
    self._base_calculator_type = base_calculator_type
    self._trigger_tolerance = float(trigger_tolerance)
    self._pellet_source_name = pellet_source_name
    self._window_after_pellet = float(window_after_pellet)
    self._dt_after_pellet = (
        float(dt_after_pellet) if dt_after_pellet is not None else None
    )
    if base_calculator_type == 'chi':
      self._base_calculator = chi_time_step_calculator.ChiTimeStepCalculator()
    elif base_calculator_type == 'fixed':
      self._base_calculator = (
          fixed_time_step_calculator.FixedTimeStepCalculator()
      )
    else:
      raise ValueError(
          'base_calculator must be "chi" or "fixed" '
      )

  def _next_dt(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      sim_state: sim_state_lib.SimState,
  ) -> jax.Array:
    """Returns a dt aligned with pellet trigger and ablation windows."""
    dt_standard = self._base_calculator._next_dt(runtime_params, sim_state)
    dt_standard = jnp.asarray(dt_standard)
    dtype = dt_standard.dtype

    t = sim_state.t
    pellet_params = runtime_params.sources.get(self._pellet_source_name)
    if pellet_params is None:
      return dt_standard

    trigger_times = getattr(pellet_params, 'trigger_times', None)
    frequency = getattr(pellet_params, 'frequency', None)
    ablation_time = getattr(pellet_params, 'ablation_time', None)
    if ablation_time is None:
        return dt_standard

    t = jnp.asarray(t, dtype=dtype)
    tol = jnp.asarray(self._trigger_tolerance, dtype=dtype)
    ablation_time = jnp.asarray(ablation_time, dtype=dtype)
    inf = jnp.asarray(jnp.inf, dtype=dtype)
    window_after_pellet = jnp.asarray(self._window_after_pellet, dtype=dtype)

    in_ablation = jnp.asarray(False)
    ablation_remaining = inf
    dt_trigger = inf
    next_trigger = inf
    dt_after_trigger = inf
    at_trigger = jnp.asarray(False)
    model_ablation_time = inf

    use_model_ablation = bool(
        getattr(pellet_params, 'use_hpi2nn_ablation_time', False)
    )
    if use_model_ablation:
      at_trigger, model_ablation_time = hpi2nn_pellet_source_lib.ablation_step(
          pellet_params, sim_state.geometry, sim_state.core_profiles
      )
      at_trigger = jnp.asarray(at_trigger)
      model_ablation_time = jnp.asarray(model_ablation_time, dtype=dtype)

    if trigger_times is not None:
      neg_inf = jnp.asarray(-jnp.inf, dtype=dtype)
      last_trigger = neg_inf

      for trigger in trigger_times:
        trigger = jnp.asarray(trigger, dtype=dtype)
        is_next_trigger = trigger > t - tol
        next_trigger = jnp.where(
          is_next_trigger,
          jnp.minimum(next_trigger, trigger),
          next_trigger,
        )
        is_past = trigger <= t + tol
        last_trigger = jnp.where(
          is_past,
          jnp.maximum(last_trigger, trigger),
          last_trigger,
        )

      delta_to_next_trigger = next_trigger - t
      if use_model_ablation:
        # One step at the trigger covering the model-predicted ablation time.
        # ablation_remaining is only read when in_ablation.
        in_ablation = at_trigger
        ablation_remaining = model_ablation_time
        dt_trigger = jnp.where(
            at_trigger, model_ablation_time, delta_to_next_trigger
        )
      else:
        end = next_trigger + ablation_time
        in_ablation = jnp.logical_and(
          t >= next_trigger - tol,
          t < end - tol,
        )
        ablation_remaining = jnp.where(
            in_ablation,
            end - t,
            ablation_remaining,
        )
        dt_trigger = jnp.where(
            in_ablation,
            ablation_remaining,
            delta_to_next_trigger,
        )
      if self._dt_after_pellet is not None:
        dt_after_pellet = jnp.asarray(self._dt_after_pellet, dtype=dtype)
        has_past_trigger = jnp.isfinite(last_trigger)
        post_window_end = last_trigger + window_after_pellet
        in_post_pellet = jnp.logical_and(
            has_past_trigger,
            jnp.logical_and(
                jnp.logical_and(
                    t > last_trigger + tol, jnp.logical_not(in_ablation)
                ),
                t < post_window_end - tol,
            ),
        )
        post_remaining = post_window_end - t
        dt_after_trigger = jnp.where(
            in_post_pellet,
            jnp.minimum(dt_after_pellet, post_remaining),
            inf,
        )


    elif frequency is not None:
      frequency = jnp.asarray(frequency, dtype=dtype)
      frequency_t_start = jnp.asarray(
          getattr(pellet_params, 'frequency_t_start', 0.0), dtype=dtype
      )
      positive_frequency = frequency > 0.0
      safe_frequency = jnp.where(
          positive_frequency, frequency, jnp.asarray(1.0, dtype=dtype)
      )
      period = 1.0 / safe_frequency
      # Phase measured from frequency_t_start (consistent with the source).
      phase = jnp.mod(t - frequency_t_start + tol, period)
      # Float rounding can leave phase just below period instead of wrapping
      # to 0 at a pellet time.
      phase = jnp.where(
          period - phase < tol, jnp.asarray(0.0, dtype=dtype), phase
      )
      delta_to_next_period = period - phase

      after_start = t > frequency_t_start + tol
      if use_model_ablation:
        in_ablation = at_trigger
        ablation_remaining = model_ablation_time
        dt_trigger_value = jnp.where(
            at_trigger, model_ablation_time, delta_to_next_period
        )
      else:
        in_ablation = jnp.logical_and(
            jnp.logical_and(positive_frequency, after_start),
            phase < ablation_time - tol,
        )
        ablation_remaining = jnp.where(
            in_ablation,
            ablation_time - phase,
            ablation_remaining,
        )
        dt_trigger_value = jnp.where(
            in_ablation,
            ablation_remaining,
            delta_to_next_period,
        )
      dt_trigger = jnp.where(
          jnp.logical_and(positive_frequency, after_start),
          dt_trigger_value,
          dt_trigger,
      )
      if self._dt_after_pellet is not None:
        dt_after_pellet = jnp.asarray(self._dt_after_pellet, dtype=dtype)
        in_post_pellet_freq = jnp.logical_and(
            jnp.logical_and(positive_frequency, after_start),
            jnp.logical_and(
                jnp.logical_and(phase > tol, jnp.logical_not(in_ablation)),
                phase < window_after_pellet - tol,
            ),
        )
        post_remaining_freq = window_after_pellet - phase
        dt_after_trigger = jnp.where(
            in_post_pellet_freq,
            jnp.minimum(dt_after_pellet, post_remaining_freq),
            inf,
        )

    dt = jnp.minimum(dt_standard, jnp.minimum(dt_trigger, dt_after_trigger))
    # During ablation, never split the window: even if dt_standard is smaller.
    dt = jnp.where(in_ablation, ablation_remaining, dt)

    crosses_t_final = (t < runtime_params.numerics.t_final) * (
        t + dt > runtime_params.numerics.t_final
    )
    dt = jax.lax.select(
        jnp.logical_and(
            runtime_params.numerics.exact_t_final,
            crosses_t_final,
        ),
        runtime_params.numerics.t_final - t,
        dt,
    )
    return dt

  def __eq__(self, other) -> bool:
    return (
        isinstance(other, type(self))
        and self._base_calculator_type == other._base_calculator_type
        and self._trigger_tolerance == other._trigger_tolerance
        and self._pellet_source_name == other._pellet_source_name
        and self._window_after_pellet == other._window_after_pellet
        and self._dt_after_pellet == other._dt_after_pellet
    )

  def __hash__(self) -> int:
    return hash(
        (
            type(self),
            self._base_calculator_type,
            self._trigger_tolerance,
            self._pellet_source_name,
            self._window_after_pellet,
            self._dt_after_pellet,
        )
    )

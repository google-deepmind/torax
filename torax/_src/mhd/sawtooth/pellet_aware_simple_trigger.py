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

"""Pellet-aware simple trigger model for sawteeth.

Same critical-shear trigger as the simple model, but suppresses sawtooth
crashes around pellet injection. A pellet makes the pressure profile non monotone. 
A sawtooth crash on such a profile can cause a low temperature collapse. 
Crashes are blocked both during the pellet ablation and for a configurable lockout 
window after each pellet.
"""

import dataclasses
from typing import Annotated, Literal

import chex
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import state
from torax._src import static_dataclass
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.mhd.sawtooth import runtime_params as sawtooth_runtime_params
from torax._src.mhd.sawtooth import trigger_base
from torax._src.sources import hpi2nn_pellet_source
from torax._src.torax_pydantic import torax_pydantic


def _in_pellet_lockout(
    pellet_params: 'hpi2nn_pellet_source.RuntimeParams',
    lockout: array_typing.FloatScalar | array_typing.FloatVector,
) -> jax.Array:
  """True if the current time is within a lockout window of a pellet trigger.

  If trigger times are used, 'lockout' may be a per-trigger array (same length
  as 'trigger_times') so each pellet gets its own lockout window otherwise a scalar
  is used for every trigger. If frequency is used, a single value is used, the
  first element is taken if an array is provided.

  Args:
    pellet_params: Runtime params of the HPI2NN pellet source.
    lockout: Duration(s) after each pellet trigger during which sawtooth
      crashes are suppressed in seconds. Scalar or per trigger array [s].

  Returns:
    Boolean scalar, True inside a lockout window.
  """
  t = jnp.asarray(pellet_params.current_time)
  dtype = t.dtype
  lockout = jnp.asarray(lockout, dtype=dtype)
  tol = jnp.asarray(1e-8, dtype=dtype)
  if pellet_params.trigger_times is not None:
    triggers = jnp.asarray(pellet_params.trigger_times, dtype=dtype)
    # Per-trigger window: each trigger uses its own lockout (or the broadcast
    # scalar). The active trigger is implicitly the one whose window contains t.
    in_window = jnp.logical_and(t >= triggers - tol, t < triggers + lockout)
    return jnp.any(in_window)
  if pellet_params.frequency is not None:
    # Frequency mode uses a single lockout; take the first element if an array
    # was provided.
    lockout = jnp.reshape(lockout, (-1,))[0]
    frequency = jnp.asarray(pellet_params.frequency, dtype=dtype)
    positive = frequency > 0.0
    safe_frequency = jnp.where(
        positive, frequency, jnp.asarray(1.0, dtype=dtype)
    )
    period = 1.0 / safe_frequency
    t_start = jnp.asarray(pellet_params.frequency_t_start, dtype=dtype)
    after_start = t > t_start + tol
    phase = jnp.mod(t - t_start + tol, period)
    return jnp.logical_and(
        jnp.logical_and(positive, after_start), phase < lockout
    )
  return jnp.asarray(False)


@dataclasses.dataclass(frozen=True, eq=False)
class PelletAwareSimpleTrigger(
    trigger_base.TriggerModel, static_dataclass.StaticDataclass
):
  """Simple critical-shear trigger with pellet-injection lockout."""

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> tuple[array_typing.BoolScalar, array_typing.FloatScalar]:
    """Checks the critical shear condition, then applies the pellet lockout.

    Args:
      runtime_params: Runtime parameters.
      geo: Geometry object.
      core_profiles: Core plasma profiles.

    Returns:
      tuple of (True if sawtooth crash is triggered, False otherwise,
        radius of q=1 surface (set to 0.0 if no surface exists))
    """
    sawtooth_params = runtime_params.mhd.sawtooth
    assert isinstance(sawtooth_params, sawtooth_runtime_params.RuntimeParams)
    assert isinstance(sawtooth_params.trigger_params, RuntimeParams)
    minimum_radius = sawtooth_params.trigger_params.minimum_radius
    s_critical = sawtooth_params.trigger_params.s_critical

    q_face = core_profiles.q_face
    s_face = core_profiles.s_face
    rho_face_norm = geo.rho_face_norm
    eps = constants.CONSTANTS.eps

    # Find the rightmost location where q crosses 1.0.
    # Allows for non-monotonic q profiles.
    q1 = q_face[:-1]
    q2 = q_face[1:]
    rho_norm1 = rho_face_norm[:-1]
    rho_norm2 = rho_face_norm[1:]
    # Handle division by zero if q1 == q2
    dq = jnp.where(jnp.abs(q2 - q1) > eps, q2 - q1, eps)
    # Calculate value of rho for q=1 by linearization of the q-profile
    # in the interval. If the calculated rho is within the interval, then the
    # q=1 surface exists for that interval.
    potential_rho_norm_q1 = (
        rho_norm1 + (rho_norm2 - rho_norm1) * (1.0 - q1) / dq
    )
    valid_interval_mask = jnp.logical_and(
        potential_rho_norm_q1 >= rho_norm1, potential_rho_norm_q1 <= rho_norm2
    )

    # Mask out intervals where there isn't actually a crossing.
    valid_rho_norm_q1 = jnp.where(
        valid_interval_mask, potential_rho_norm_q1, 0.0
    )

    rho_norm_q1 = jnp.max(valid_rho_norm_q1)
    s_at_q1 = jnp.interp(rho_norm_q1, rho_face_norm, s_face)

    rho_norm_above_minimum = rho_norm_q1 > minimum_radius
    s_above_critical = s_at_q1 > s_critical
    trigger = jnp.logical_and(rho_norm_above_minimum, s_above_critical)

    # Suppress sawtooth crashes around pellet injection. Crashes are blocked
    # both during the ablation and for a configurable lockout window after each
    # pellet.
    pellet_params = runtime_params.sources.get(
        hpi2nn_pellet_source.HPI2NNPelletSource.SOURCE_NAME
    )
    if pellet_params is not None and hasattr(pellet_params, 'current_time'):
      pellet_active, _ = hpi2nn_pellet_source.is_pellet_active(pellet_params)
      lockout = sawtooth_params.trigger_params.pellet_lockout
      in_lockout = _in_pellet_lockout(pellet_params, lockout)
      suppress = jnp.logical_or(pellet_active, in_lockout)
      trigger = jnp.logical_and(trigger, jnp.logical_not(suppress))

    return (
        trigger,
        rho_norm_q1,
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(sawtooth_runtime_params.TriggerRuntimeParams):
  """Runtime params for pellet-aware simple trigger model.

  Attributes:
    s_critical: Critical shear value at q=1 for sawtooth triggering.
    pellet_lockout: Duration after each pellet trigger during which sawtooth
      crashes are suppressed [s]. Scalar (applied to all triggers) or a
      per-trigger array aligned with trigger_times.
  """

  s_critical: array_typing.FloatScalar
  pellet_lockout: array_typing.FloatScalar | tuple[float, ...]


class PelletAwareSimpleTriggerConfig(trigger_base.TriggerConfig):
  """Pydantic model for pellet-aware simple trigger configuration.

  Attributes:
    s_critical: Critical shear value.
    pellet_lockout: Duration after each pellet trigger during which sawtooth
      crashes are suppressed [s]. A scalar applies to all pellets. A list
      gives a per-trigger value aligned with trigger_times (frequency mode uses
      the first element). Default 0.0 disables the lockout (only the ablation
      window itself suppresses crashes).
  """

  s_critical: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.1)
  )
  pellet_lockout: float | tuple[float, ...] = 0.0
  model_name: Annotated[
      Literal['pellet_aware_simple'], torax_pydantic.JAX_STATIC
  ] = 'pellet_aware_simple'

  def build_runtime_params(
      self,
      t: chex.Numeric,
  ) -> RuntimeParams:
    base_kwargs = dataclasses.asdict(super().build_runtime_params(t))
    return RuntimeParams(
        **base_kwargs,
        s_critical=self.s_critical.get_value(t),
        pellet_lockout=self.pellet_lockout,
    )

  def build_trigger_model(self) -> PelletAwareSimpleTrigger:
    return PelletAwareSimpleTrigger()

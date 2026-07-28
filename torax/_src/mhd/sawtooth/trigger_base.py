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

"""Base pydantic config and model for sawtooth trigger."""

import abc
import dataclasses

import chex
from jax import numpy as jnp
import pydantic
from torax._src import array_typing
from torax._src import state
from torax._src import static_dataclass
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.mhd.sawtooth import runtime_params as sawtooth_runtime_params
from torax._src.torax_pydantic import torax_pydantic


@dataclasses.dataclass(frozen=True, eq=False)
class TriggerModel(static_dataclass.StaticDataclass, abc.ABC):
  """Abstract base class for sawtooth trigger models."""

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> tuple[array_typing.BoolScalar, array_typing.FloatScalar]:
    """Indicates if a crash is triggered and the radius of the q=1 surface.

    Delegates the trigger condition itself to the concrete model, then applies
    the crash suppression windows. This is used to block crashes around a prescribed 
    event. No window is configured by default.
    Args:
      runtime_params: Runtime parameters.
      geo: Geometry object.
      core_profiles: Core plasma profiles.

    Returns:
      tuple of (True if sawtooth crash is triggered, False otherwise,
        radius of q=1 surface (set to 0.0 if no surface exists))
    """
    trigger, rho_norm_q1 = self.compute_trigger(
        runtime_params, geo, core_profiles
    )

    sawtooth_params = runtime_params.mhd.sawtooth
    assert isinstance(sawtooth_params, sawtooth_runtime_params.RuntimeParams)
    trigger_params = sawtooth_params.trigger_params
    if trigger_params.suppression_times:
      t = jnp.asarray(runtime_params.t)
      times = jnp.asarray(trigger_params.suppression_times, dtype=t.dtype)
      durations = jnp.broadcast_to(
          jnp.asarray(trigger_params.suppression_duration, dtype=t.dtype),
          times.shape,
      )
      tol = jnp.asarray(1e-8, dtype=t.dtype)
      # Suppression window is half-open: [t_event, t_event + duration).
      in_suppression_window = jnp.any(
          jnp.logical_and(t >= times - tol, t < times + durations)
      )
      trigger = jnp.logical_and(
          trigger, jnp.logical_not(in_suppression_window)
      )

    return trigger, rho_norm_q1

  @abc.abstractmethod
  def compute_trigger(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> tuple[array_typing.BoolScalar, array_typing.FloatScalar]:
    """Indicates if a crash is triggered and the radius of the q=1 surface."""


class TriggerConfig(torax_pydantic.BaseModelFrozen):
  """Base config for all trigger models.

  Attributes:
    minimum_radius: The minimum radius of the q=1 surface for triggering.
    suppression_times: Times [s] of events after which sawtooth crashes are
      suppressed. Empty by default.
    suppression_duration: Duration [s] of the suppression window opened by each
      entry of suppression_times. A scalar applies to all of them, a list gives
      a per-event value and must then have the same length.
  """

  minimum_radius: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.05)
  )
  suppression_times: tuple[float, ...] = ()
  suppression_duration: float | tuple[float, ...] = 0.0

  @pydantic.model_validator(mode='after')
  def _validate_suppression_windows(self):
    if isinstance(self.suppression_duration, tuple) and len(
        self.suppression_duration
    ) != len(self.suppression_times):
      raise ValueError(
          'suppression_duration length'
          f' ({len(self.suppression_duration)}) must match suppression_times'
          f' length ({len(self.suppression_times)}).'
      )
    return self

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> sawtooth_runtime_params.TriggerRuntimeParams:
    return sawtooth_runtime_params.TriggerRuntimeParams(
        minimum_radius=self.minimum_radius.get_value(t),
        suppression_times=self.suppression_times,
        suppression_duration=self.suppression_duration,
    )

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
from torax._src import array_typing
from torax._src import static_dataclass
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import core_profiles as core_profiles_lib
from torax._src.geometry import geometry
from torax._src.mhd.sawtooth import runtime_params as sawtooth_runtime_params
from torax._src.torax_pydantic import torax_pydantic


def find_rho_norm_q1(
    q_face: array_typing.FloatVectorFace,
    rho_face_norm: array_typing.FloatVectorFace,
) -> array_typing.FloatScalar:
  """Finds the outermost normalized radius of the q = 1 surface.

  Uses piecewise-linear interpolation of `q_face` on each radial interval that
  straddles `q = 1` (one endpoint strictly below 1 and the other at or above 1)
  so that non-monotonic q profiles are supported while flat or tangent `q >= 1`
  profiles do not produce phantom crossings. Returns `0.0` if no q = 1 crossing
  exists.

  Args:
    q_face: Safety factor profile on the face grid [dimensionless].
    rho_face_norm: Normalized toroidal flux coordinate on the face grid
      [rho_norm].

  Returns:
    Outermost normalized radius where q = 1 [rho_norm], or 0.0 if absent.
  """
  q1 = q_face[:-1]
  q2 = q_face[1:]
  rho_norm1 = rho_face_norm[:-1]
  rho_norm2 = rho_face_norm[1:]
  # `jnp.diff` on a boolean array computes adjacent XOR, so an interval is a
  # crossing iff one endpoint is strictly below 1 and the other is >= 1 (or NaN,
  # propagating NaNs instead of masking them). This guarantees `q2 - q1 != 0` on
  # every valid finite interval, avoids phantom crossings on flat `q = 1` or
  # tangent `q >= 1` profiles, and preserves small-slope downward crossings.
  is_crossing = jnp.diff(q_face < 1.0)
  # Use a unit denominator on non-crossing intervals to avoid 0/0 in both
  # forward evaluation and reverse-mode autodiff (`jax.grad`).
  safe_dq = jnp.where(is_crossing, q2 - q1, 1.0)
  potential_rho_norm_q1 = (
      rho_norm1 + (rho_norm2 - rho_norm1) * (1.0 - q1) / safe_dq
  )
  valid_rho_norm_q1 = jnp.where(is_crossing, potential_rho_norm_q1, 0.0)
  return jnp.max(valid_rho_norm_q1)


@dataclasses.dataclass(frozen=True, eq=False)
class TriggerModel(static_dataclass.StaticDataclass, abc.ABC):
  """Abstract base class for sawtooth trigger models."""

  @abc.abstractmethod
  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: core_profiles_lib.CoreProfiles,
  ) -> tuple[array_typing.BoolScalar, array_typing.FloatScalar]:
    """Indicates if a crash is triggered and the radius of the q=1 surface."""


class TriggerConfig(torax_pydantic.ConfigBase):
  """Base config for all trigger models.

  Attributes:
    minimum_radius: The minimum normalized radius [$rho_norm$] of the `q=1`
      surface required to trigger a crash.
  """

  minimum_radius: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.05)
  )

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> sawtooth_runtime_params.TriggerRuntimeParams:
    return sawtooth_runtime_params.TriggerRuntimeParams(
        minimum_radius=self.minimum_radius.get_value(t),
    )

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

"""Dataclass representing runtime parameter inputs to the pedestal models."""

import dataclasses
import enum

import jax
from torax._src import array_typing

# pylint: disable=invalid-name


@enum.unique
class Mode(enum.Enum):
  """Defines how the pedestal is generated."""

  # The pedestal is set by modifying the transport coefficients.
  ADAPTIVE_TRANSPORT = "ADAPTIVE_TRANSPORT"

  # The pedestal is set by adding a source/sink term.
  ADAPTIVE_SOURCE = "ADAPTIVE_SOURCE"


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class FormationRuntimeParams:
  """Runtime params for pedestal formation models."""

  sharpness: array_typing.FloatScalar
  offset: array_typing.FloatScalar
  base_multiplier: array_typing.FloatScalar


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SaturationRuntimeParams:
  """Runtime params for pedestal saturation models."""

  steepness: array_typing.FloatScalar
  offset: array_typing.FloatScalar
  base_multiplier: array_typing.FloatScalar


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams:
  """Input params for the pedestal model.

  Attributes:
    set_pedestal: Whether to use the pedestal model and set the pedestal.
    mode: Defines how the pedestal is generated.
    use_formation_model_with_adaptive_source: When True and mode is
      ADAPTIVE_SOURCE, enables state-dependent L-H transitions based on P_SOL vs
      P_LH comparison. The formation model is used to check the transition
      condition. When False, ADAPTIVE_SOURCE mode always applies the pedestal
      values directly, whenever set_pedestal is True.
    transition_time_width: Duration of the L-H or H-L transition ramp [s].
      During a transition, pedestal values are linearly interpolated between
      L-mode and H-mode values over this time window. Only used when
      use_formation_model_with_adaptive_source is True.
    P_LH_hysteresis_factor: Hysteresis factor for H-L back transitions. When
      checking for an H-L transition, the L-H threshold power P_LH is
      multiplied by this factor, i.e. the back transition occurs when
      P_SOL < P_LH * P_LH_hysteresis_factor. A value less than 1 means that
      the plasma must lose more power to transition back to L-mode than was
      required to enter H-mode, which is the experimentally observed behavior.
      Must be in [0, 1]. Only used when
      use_formation_model_with_adaptive_source is True.
    formation: Runtime params for the formation model.
    saturation: Runtime params for the saturation model.
    chi_max: Maximum effective thermal diffusion coefficient [m^2/s].
    D_e_max: Maximum effective particle diffusion coefficient [m^2/s].
    V_e_max: Maximum effective particle pinch velocity [m/s].
    V_e_min: Minimum effective particle pinch velocity [m/s].
    pedestal_top_smoothing_width: Width of the smoothing kernel at the pedestal
      top.
  """

  set_pedestal: array_typing.BoolScalar
  mode: Mode = dataclasses.field(metadata={"static": True})
  use_formation_model_with_adaptive_source: bool = dataclasses.field(
      metadata={"static": True}
  )
  transition_time_width: array_typing.FloatScalar
  P_LH_hysteresis_factor: array_typing.FloatScalar
  formation: FormationRuntimeParams
  saturation: SaturationRuntimeParams
  chi_max: array_typing.FloatScalar
  D_e_max: array_typing.FloatScalar
  V_e_max: array_typing.FloatScalar
  V_e_min: array_typing.FloatScalar
  pedestal_top_smoothing_width: array_typing.FloatScalar

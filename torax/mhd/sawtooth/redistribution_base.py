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

"""Base pydantic config and model for sawtooth redistribution."""

import abc
import chex
from torax import array_typing
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.mhd.sawtooth import runtime_params as sawtooth_runtime_params
from torax.torax_pydantic import torax_pydantic


class RedistributionModel(abc.ABC):
  """Abstract base class for sawtooth redistribution models."""

  @abc.abstractmethod
  def __call__(
      self,
      rho_norm_q1: array_typing.ScalarFloat,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles_t: state.CoreProfiles,
      core_profiles_t_plus_crash_dt: state.CoreProfiles,
  ) -> state.CoreProfiles:
    """Returns a redistributed core_profiles if sawtooth has been triggered."""

  @abc.abstractmethod
  def __hash__(self) -> int:
    """Returns a hash of the redistribution model.

    Should be implemented to support jax.jit caching.
    """

  @abc.abstractmethod
  def __eq__(self, other) -> bool:
    """Equality method to be implemented to support jax.jit caching."""


class RedistributionConfig(torax_pydantic.BaseModelFrozen):
  """Base config for all redistribution models.

  Attributes:
    flattening_factor: The factor by which the profile is flattened.
    Default is near 1.0 but not exactly 1.0, to avoid zero-gradients.
  """

  flattening_factor: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.01)
  )

  def build_dynamic_params(
      self, t: chex.Numeric
  ) -> sawtooth_runtime_params.RedistributionDynamicRuntimeParams:
    return sawtooth_runtime_params.RedistributionDynamicRuntimeParams(
        flattening_factor=self.flattening_factor.get_value(t),
    )

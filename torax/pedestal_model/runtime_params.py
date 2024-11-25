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

"""Dataclass representing runtime parameter inputs to the pedestal models.

This is the dataclass runtime config exposed to the user. The actual model gets
a time-interpolated version of this config via the DynamicRuntimeParams.
"""

from __future__ import annotations

import chex
from torax import geometry
from torax.config import base


@chex.dataclass
class RuntimeParams(base.RuntimeParametersConfig['RuntimeParamsProvider']):
  """Runtime parameters for the pedestal model.

  This is the dataclass runtime config exposed to the user. The actual model
  gets a time-interpolated version of this config via the DynamicConfigSlice.
  """

  def make_provider(
      self, torax_mesh: geometry.Grid1D | None = None
  ) -> RuntimeParamsProvider:
    return RuntimeParamsProvider(**self.get_provider_kwargs(torax_mesh))


@chex.dataclass
class RuntimeParamsProvider(
    base.RuntimeParametersProvider['DynamicRuntimeParams']
):
  """Provides a RuntimeParams to use during time t of the sim."""

  runtime_params_config: RuntimeParams

  def build_dynamic_params(self, t: chex.Numeric) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(**self.get_dynamic_params_kwargs(t))


@chex.dataclass(frozen=True)
class DynamicRuntimeParams:
  """Input params for the pedestal model which can be used as compiled args."""

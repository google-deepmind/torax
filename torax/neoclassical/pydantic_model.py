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
"""Pydantic model for the neoclassical package."""

from typing import Annotated

import pydantic
from torax.neoclassical import runtime_params
from torax.neoclassical.bootstrap_current import sauter
from torax.neoclassical.bootstrap_current import zeros
from torax.torax_pydantic import torax_pydantic


def _bootstrap_current_discriminator(config) -> str:
  """Returns the model type of the bootstrap current."""
  if isinstance(config, dict):
    return config.get("model_name", "zeros")
  return getattr(config, "model_name", "zeros")


class Neoclassical(torax_pydantic.BaseModelFrozen):
  """Config for neoclassical models."""

  bootstrap_current: (
      Annotated[zeros.ZerosModelConfig, pydantic.Tag("zeros")]
      | Annotated[sauter.SauterModelConfig, pydantic.Tag("sauter")]
  ) = pydantic.Field(
      discriminator=pydantic.Discriminator(_bootstrap_current_discriminator)
  )

  def build_dynamic_params(self) -> runtime_params.DynamicRuntimeParams:
    return runtime_params.DynamicRuntimeParams(
        bootstrap_current=self.bootstrap_current.build_dynamic_params()
    )

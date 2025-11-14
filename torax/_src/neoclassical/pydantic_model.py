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

import copy
from typing import Any

import pydantic
from torax._src.neoclassical import neoclassical_models
from torax._src.neoclassical import runtime_params as runtime_params_lib
from torax._src.neoclassical.bootstrap_current import sauter as sauter_current
from torax._src.neoclassical.bootstrap_current import zeros as bootstrap_current_zeros
from torax._src.neoclassical.conductivity import sauter as sauter_conductivity
from torax._src.neoclassical.transport import angioni_sauter
from torax._src.neoclassical.transport import zeros as transport_zeros
from torax._src.torax_pydantic import torax_pydantic


class Neoclassical(torax_pydantic.BaseModelFrozen):
  """Config for neoclassical models."""

  bootstrap_current: (
      bootstrap_current_zeros.ZerosModelConfig
      | sauter_current.SauterModelConfig
  ) = pydantic.Field(discriminator="model_name")
  conductivity: sauter_conductivity.SauterModelConfig = (
      torax_pydantic.ValidatedDefault(sauter_conductivity.SauterModelConfig())
  )
  transport: (
      transport_zeros.ZerosModelConfig | angioni_sauter.AngioniSauterModelConfig
  ) = pydantic.Field(discriminator="model_name")

  @pydantic.model_validator(mode="before")
  @classmethod
  def _defaults(cls, data: dict[str, Any]) -> dict[str, Any]:
    configurable_data = copy.deepcopy(data)
    # Set zero models if model not in config dict.
    if "bootstrap_current" not in configurable_data:
      configurable_data["bootstrap_current"] = {"model_name": "zeros"}
    if "transport" not in configurable_data:
      configurable_data["transport"] = {"model_name": "zeros"}
    # Set default model names.
    if "model_name" not in configurable_data["bootstrap_current"]:
      configurable_data["bootstrap_current"]["model_name"] = "sauter"
    if "model_name" not in configurable_data["transport"]:
      configurable_data["transport"]["model_name"] = "angioni_sauter"

    return configurable_data

  def build_runtime_params(self) -> runtime_params_lib.RuntimeParams:
    return runtime_params_lib.RuntimeParams(
        bootstrap_current=self.bootstrap_current.build_runtime_params(),
        conductivity=self.conductivity.build_runtime_params(),
        transport=self.transport.build_runtime_params(),
    )

  def build_models(self) -> neoclassical_models.NeoclassicalModels:
    return neoclassical_models.NeoclassicalModels(
        conductivity=self.conductivity.build_model(),
        bootstrap_current=self.bootstrap_current.build_model(),
        transport=self.transport.build_model(),
    )

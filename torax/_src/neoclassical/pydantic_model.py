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

import abc
import copy
from typing import Annotated, Any, Literal

import pydantic
from torax._src.neoclassical import neoclassical_models
from torax._src.neoclassical import runtime_params as runtime_params_lib
from torax._src.neoclassical.bootstrap_current import redl as redl_current
from torax._src.neoclassical.bootstrap_current import sauter as sauter_current
from torax._src.neoclassical.bootstrap_current import zeros as bootstrap_current_zeros
from torax._src.neoclassical.conductivity import redl as redl_conductivity
from torax._src.neoclassical.conductivity import sauter as sauter_conductivity
from torax._src.neoclassical.poloidal_velocity import kim as kim_poloidal_velocity
from torax._src.neoclassical.poloidal_velocity import zeros as poloidal_velocity_zeros
from torax._src.neoclassical.transport import angioni_sauter
from torax._src.neoclassical.transport import zeros as transport_zeros
from torax._src.torax_pydantic import torax_pydantic


class BaseNeoclassical(torax_pydantic.BaseModelFrozen, abc.ABC):
  """Base config for neoclassical models.

  Subclasses configure either the built-in composite analytical model
  (`AnalyticalNeoclassical`, which delegates to individual analytical
  sub-models) or a custom integrated neoclassical solver that computes all
  neoclassical outputs in a single evaluation.
  """

  @abc.abstractmethod
  def build_runtime_params(self) -> runtime_params_lib.RuntimeParams:
    """Builds runtime params for the neoclassical model."""

  @abc.abstractmethod
  def build_model(self) -> neoclassical_models.NeoclassicalModel:
    """Builds the neoclassical model."""


class AnalyticalNeoclassical(BaseNeoclassical):
  """Config for the composite analytical neoclassical model."""

  # Top-level neoclassical model selector. Built-in support uses 'analytical'
  # (which delegates to modular analytical sub-models). Custom integrated
  # neoclassical solvers that compute all neoclassical outputs in a single
  # evaluation can be registered and selected via this discriminator.
  model_name: Annotated[Literal["analytical"], torax_pydantic.JAX_STATIC] = (
      "analytical"
  )
  bootstrap_current: (
      bootstrap_current_zeros.ZerosModelConfig
      | sauter_current.SauterModelConfig
      | redl_current.RedlModelConfig
  ) = pydantic.Field(discriminator="model_name")
  conductivity: (
      sauter_conductivity.SauterModelConfig
      | redl_conductivity.RedlModelConfig
  ) = pydantic.Field(discriminator="model_name")
  transport: (
      transport_zeros.ZerosModelConfig | angioni_sauter.AngioniSauterModelConfig
  ) = pydantic.Field(discriminator="model_name")
  poloidal_velocity: (
      poloidal_velocity_zeros.ZerosModelConfig
      | kim_poloidal_velocity.KimModelConfig
  ) = pydantic.Field(discriminator="model_name")

  @pydantic.model_validator(mode="before")
  @classmethod
  def _defaults(cls, data: dict[str, Any]) -> dict[str, Any]:
    configurable_data = copy.deepcopy(data)
    if "model_name" not in configurable_data:
      configurable_data["model_name"] = "analytical"
    # Set zero models if model not in config dict.
    if "bootstrap_current" not in configurable_data:
      configurable_data["bootstrap_current"] = {"model_name": "zeros"}
    if "transport" not in configurable_data:
      configurable_data["transport"] = {"model_name": "zeros"}
    # Set default model names.
    configurable_data.setdefault("conductivity", {})
    configurable_data.setdefault("poloidal_velocity", {})
    if "model_name" not in configurable_data["bootstrap_current"]:
      configurable_data["bootstrap_current"]["model_name"] = "sauter"
    if "model_name" not in configurable_data["conductivity"]:
      configurable_data["conductivity"]["model_name"] = "sauter"
    if "model_name" not in configurable_data["transport"]:
      configurable_data["transport"]["model_name"] = "angioni_sauter"
    if "model_name" not in configurable_data["poloidal_velocity"]:
      configurable_data["poloidal_velocity"]["model_name"] = "kim"

    return configurable_data

  def build_runtime_params(self) -> runtime_params_lib.AnalyticalRuntimeParams:
    return runtime_params_lib.AnalyticalRuntimeParams(
        bootstrap_current=self.bootstrap_current.build_runtime_params(),
        conductivity=self.conductivity.build_runtime_params(),
        transport=self.transport.build_runtime_params(),
        poloidal_velocity=self.poloidal_velocity.build_runtime_params(),
    )

  def build_model(self) -> neoclassical_models.AnalyticalNeoclassicalModel:
    return neoclassical_models.AnalyticalNeoclassicalModel(
        conductivity=self.conductivity.build_model(),
        bootstrap_current=self.bootstrap_current.build_model(),
        transport=self.transport.build_model(),
        poloidal_velocity=self.poloidal_velocity.build_model(),
    )


NeoclassicalConfig = Annotated[
    AnalyticalNeoclassical,
    pydantic.Field(discriminator="model_name"),
]

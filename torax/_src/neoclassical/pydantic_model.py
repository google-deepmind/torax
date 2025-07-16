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
from torax._src.neoclassical import neoclassical_models
from torax._src.neoclassical import runtime_params
from torax._src.neoclassical.bootstrap_current import sauter as sauter_current
from torax._src.neoclassical.bootstrap_current import zeros as bootstrap_current_zeros
from torax._src.neoclassical.conductivity import sauter as sauter_conductivity
from torax._src.neoclassical.transport import angioni_sauter
from torax._src.neoclassical.transport import zeros as transport_zeros
from torax._src.torax_pydantic import torax_pydantic

BootstrapCurrentConfig = Annotated[
    bootstrap_current_zeros.ZerosModelConfig | sauter_current.SauterModelConfig,
    pydantic.BeforeValidator(
        torax_pydantic.create_default_model_injector("model_name", "sauter")
    ),
]

TransportConfig = Annotated[
    transport_zeros.ZerosModelConfig | angioni_sauter.AngioniSauterModelConfig,
    pydantic.BeforeValidator(
        torax_pydantic.create_default_model_injector("model_name", "zeros")
    ),
]


class Neoclassical(torax_pydantic.BaseModelFrozen):
  """Config for neoclassical models."""

  bootstrap_current: BootstrapCurrentConfig = pydantic.Field(
      discriminator="model_name",
      default_factory=bootstrap_current_zeros.ZerosModelConfig,
      validate_default=True,
  )
  conductivity: sauter_conductivity.SauterModelConfig = (
      torax_pydantic.ValidatedDefault(sauter_conductivity.SauterModelConfig())
  )
  transport: TransportConfig = pydantic.Field(
      discriminator="model_name",
      default_factory=transport_zeros.ZerosModelConfig,
      validate_default=True,
  )

  def build_dynamic_params(self) -> runtime_params.DynamicRuntimeParams:
    return runtime_params.DynamicRuntimeParams(
        bootstrap_current=self.bootstrap_current.build_dynamic_params(),
        conductivity=self.conductivity.build_dynamic_params(),
        transport=self.transport.build_dynamic_params(),
    )

  def build_models(self) -> neoclassical_models.NeoclassicalModels:
    """Builds and returns a container with instantiated neoclassical model objects."""
    return neoclassical_models.NeoclassicalModels(
        conductivity=self.conductivity.build_model(),
        bootstrap_current=self.bootstrap_current.build_model(),
        transport=self.transport.build_model(),
    )

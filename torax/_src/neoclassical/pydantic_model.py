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
from typing import Annotated, Any, Literal

import pydantic
from torax._src import array_typing
from torax._src.neoclassical import neoclassical_models
from torax._src.neoclassical import runtime_params as runtime_params_lib
from torax._src.neoclassical.bootstrap_current import redl as redl_current
from torax._src.neoclassical.bootstrap_current import sauter as sauter_current
from torax._src.neoclassical.bootstrap_current import zeros as bootstrap_current_zeros
from torax._src.neoclassical.conductivity import redl as redl_conductivity
from torax._src.neoclassical.conductivity import sauter as sauter_conductivity
from torax._src.neoclassical.transport import angioni_sauter
from torax._src.neoclassical.transport import zeros as transport_zeros
from torax._src.torax_pydantic import torax_pydantic


class Neoclassical(torax_pydantic.BaseModelFrozen):
  """Config for neoclassical models.

  Attributes:
    f_trap_model: Trapped-particle fraction approximation used by bootstrap
      current, conductivity, and Angioni-Sauter transport.

      * ``'sauter'``: Sauter FED 2016 Eqs. 33–34 (default).
      * ``'simple'``: :math:`f_t = 1.46\\sqrt{\\epsilon} - 0.46\\epsilon`.
      * ``'LinLiu'``: Lin-Liu & Miller (Phys. Plasmas 2, 1666, 1995)
        :math:`f_t = 0.75 f_{tu} + 0.25 f_{tl}` with circular upper/lower
        bounds.
      * ``'RABBIT'``: M. Weiland et al., Nucl. Fusion 58, 082032 (2018),
        :math:`f_t = 1.4624256\\sqrt{\\epsilon_\\mathrm{eff}} -
        0.46\\epsilon_\\mathrm{eff}^{3/2}` with
        :math:`\\epsilon_\\mathrm{eff}=(R_\\mathrm{max}-R_0)/R_0`.
      * ``'numerical'``: numerical Sauter PoP 1999 Eq. (12). Uses stored
        flux-surface :math:`|B|(\\theta)` and FSA weights when the
        geometry provides them (EQDSK); otherwise a NEO/GACODE Miller
        reconstruction with :math:`|B|=\\sqrt{(F/R)^2+B_p^2}` and
        Jacobian FSA weights.
  """

  bootstrap_current: (
      bootstrap_current_zeros.ZerosModelConfig
      | sauter_current.SauterModelConfig
      | redl_current.RedlModelConfig
  ) = pydantic.Field(discriminator="model_name")
  conductivity: (
      sauter_conductivity.SauterModelConfig | redl_conductivity.RedlModelConfig
  ) = pydantic.Field(
      discriminator="model_name",
      default_factory=sauter_conductivity.SauterModelConfig,
  )
  transport: (
      transport_zeros.ZerosModelConfig | angioni_sauter.AngioniSauterModelConfig
  ) = pydantic.Field(discriminator="model_name")
  poloidal_velocity_multiplier: array_typing.FloatScalar = 1.0
  f_trap_model: Annotated[
      Literal['sauter', 'simple', 'RABBIT', 'LinLiu', 'numerical'],
      torax_pydantic.JAX_STATIC,
  ] = 'sauter'

  @pydantic.model_validator(mode="before")
  @classmethod
  def _defaults(cls, data: dict[str, Any]) -> dict[str, Any]:
    configurable_data = copy.deepcopy(data)
    # Set zero models if model not in config dict.
    if "bootstrap_current" not in configurable_data:
      configurable_data["bootstrap_current"] = {"model_name": "zeros"}
    if "transport" not in configurable_data:
      configurable_data["transport"] = {"model_name": "zeros"}
    if "conductivity" not in configurable_data:
      configurable_data["conductivity"] = {"model_name": "sauter"}
    # Set default model names.
    if "model_name" not in configurable_data["bootstrap_current"]:
      configurable_data["bootstrap_current"]["model_name"] = "sauter"
    if "model_name" not in configurable_data["transport"]:
      configurable_data["transport"]["model_name"] = "angioni_sauter"
    if "model_name" not in configurable_data["conductivity"]:
      configurable_data["conductivity"]["model_name"] = "sauter"

    return configurable_data

  def build_runtime_params(self) -> runtime_params_lib.RuntimeParams:
    return runtime_params_lib.RuntimeParams(
        bootstrap_current=self.bootstrap_current.build_runtime_params(),
        conductivity=self.conductivity.build_runtime_params(),
        transport=self.transport.build_runtime_params(),
        poloidal_velocity_multiplier=self.poloidal_velocity_multiplier,
        f_trap_model=self.f_trap_model,
    )

  def build_models(self) -> neoclassical_models.NeoclassicalModels:
    return neoclassical_models.NeoclassicalModels(
        conductivity=self.conductivity.build_model(),
        bootstrap_current=self.bootstrap_current.build_model(),
        transport=self.transport.build_model(),
    )

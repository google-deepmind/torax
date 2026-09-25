# Copyright 2026 DeepMind Technologies Limited
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
"""Register neoclassical models and sub-models with TORAX."""
from typing import Annotated, Any, get_args

from torax._src.neoclassical import pydantic_model
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.neoclassical.poloidal_velocity import base as poloidal_velocity_base
from torax._src.neoclassical.transport import base as transport_base
from torax._src.torax_pydantic import model_config


def register_neoclassical_model(
    pydantic_model_class: type[pydantic_model.BaseNeoclassical],
) -> None:
  """Registers a top-level neoclassical model with TORAX."""
  inner_union, discriminator = get_args(pydantic_model.NeoclassicalConfig)
  new_union = inner_union | pydantic_model_class
  setattr(
      pydantic_model,
      'NeoclassicalConfig',
      Annotated[new_union, discriminator],
  )
  setattr(
      model_config.ToraxConfig.model_fields['neoclassical'],
      'annotation',
      pydantic_model.NeoclassicalConfig,
  )
  model_config.ToraxConfig.model_rebuild(force=True)


def _register_analytical_submodel(
    field_name: str,
    pydantic_model_class: type[Any],
) -> None:
  """Registers a sub-model on the AnalyticalNeoclassical config."""
  field_info = pydantic_model.AnalyticalNeoclassical.model_fields[field_name]
  field_info.annotation |= pydantic_model_class  # pyrefly: ignore[bad-assignment]
  field_info.discriminator = 'model_name'
  pydantic_model.AnalyticalNeoclassical.model_rebuild(force=True)
  model_config.ToraxConfig.model_rebuild(force=True)


def register_bootstrap_current_model(
    pydantic_model_class: type[
        bootstrap_current_base.BootstrapCurrentModelConfig
    ],
) -> None:
  """Registers a neoclassical bootstrap current sub-model with TORAX."""
  _register_analytical_submodel('bootstrap_current', pydantic_model_class)


def register_conductivity_model(
    pydantic_model_class: type[conductivity_base.ConductivityModelConfig],
) -> None:
  """Registers a neoclassical conductivity sub-model with TORAX."""
  _register_analytical_submodel('conductivity', pydantic_model_class)


def register_neoclassical_transport_model(
    pydantic_model_class: type[
        transport_base.NeoclassicalTransportModelConfig
    ],
) -> None:
  """Registers a neoclassical transport sub-model with TORAX."""
  _register_analytical_submodel('transport', pydantic_model_class)


def register_poloidal_velocity_model(
    pydantic_model_class: type[
        poloidal_velocity_base.PoloidalVelocityModelConfig
    ],
) -> None:
  """Registers a neoclassical poloidal velocity sub-model with TORAX."""
  _register_analytical_submodel('poloidal_velocity', pydantic_model_class)

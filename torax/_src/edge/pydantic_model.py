# Copyright 2025 DeepMind Technologies Limited
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

"""Pydantic configs for edge models."""

from collections.abc import Mapping
import dataclasses
from typing import Annotated, ClassVar, Literal, Self
import chex
import jax
import jax.numpy as jnp
import numpy as np
import pydantic
from torax._src.edge import base
from torax._src.edge import runtime_params as edge_runtime_params
from torax._src.edge.extended_lengyel import pydantic_model as extended_lengyel_pydantic_model
from torax._src.torax_pydantic import torax_pydantic

SubModelConfig = Annotated[
    extended_lengyel_pydantic_model.ExtendedLengyelConfig,
    pydantic.Field(discriminator='model_name'),
]


class CombinedEdgeConfig(base.EdgeModelConfig):
  """Pydantic config for a combined edge model composed of sub-models.

  Each boundary condition is provided by whichever sub-model has the
  corresponding `update_*` flag enabled. At most one sub-model may have a given
  `update_*` flag enabled across all time. The combined model's own `update_*`
  flags are derived from the sub-models (logical OR) and cannot be set directly.

  Attributes:
    model_name: Discriminator field set to `'combined'`.
    sub_models: Mapping from sub-model name to its `EdgeModelConfig`.
  """

  supported_boundary_conditions: ClassVar[base.SupportedBoundaryConditions] = (
      base.SupportedBoundaryConditions()
  )
  model_name: Annotated[Literal['combined'], torax_pydantic.JAX_STATIC] = (
      'combined'
  )
  sub_models: Mapping[str, SubModelConfig] = pydantic.Field(min_length=1)

  @pydantic.model_validator(mode='after')
  def _validate_update_flags(self) -> Self:
    """Overrides the base check to forbid top-level update flags."""
    for field in dataclasses.fields(base.SupportedBoundaryConditions):
      update_flag = f'update_{field.name}'
      if np.any(getattr(self, update_flag).value):
        raise ValueError(
            f"'{update_flag}' cannot be set on a combined edge model. Set it on"
            ' the individual sub_models instead.'
        )
    return self

  def _get_bc_provider(self, update_flag: str) -> str | None:
    """Returns the unique sub-model name with `update_flag` enabled, or None."""
    providers = [
        name
        for name, sub in self.sub_models.items()
        if np.any(getattr(sub, update_flag).value)
    ]
    if len(providers) > 1:
      raise ValueError(
          f"'{update_flag}' may only be enabled on a single sub-model"
          f' across all time, but is enabled on {sorted(providers)}.'
      )
    return providers[0] if providers else None

  @pydantic.model_validator(mode='after')
  def _validate_unique_bc_providers(self) -> Self:
    """Ensures at most one sub-model updates each boundary condition."""
    for field in dataclasses.fields(base.SupportedBoundaryConditions):
      self._get_bc_provider(f'update_{field.name}')
    return self

  def build_edge_model(self) -> base.CombinedEdgeModel:
    """Builds a CombinedEdgeModel from the configured sub-models."""
    return base.CombinedEdgeModel(
        sub_models={
            name: sub_config.build_edge_model()
            for name, sub_config in self.sub_models.items()
        },
        T_e_sub_model=self._get_bc_provider('update_T_e'),
        T_i_sub_model=self._get_bc_provider('update_T_i'),
        n_e_sub_model=self._get_bc_provider('update_n_e'),
        n_impurities_sub_model=self._get_bc_provider('update_n_impurities'),
    )

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> edge_runtime_params.CombinedRuntimeParams:
    """Builds CombinedRuntimeParams for all configured sub-models at time t."""
    sub_params = {
        name: sub_config.build_runtime_params(t)
        for name, sub_config in self.sub_models.items()
    }

    def any_sub_model(update_flag: str) -> jax.Array:
      return jnp.any(
          jnp.asarray(
              [getattr(p, update_flag) for p in sub_params.values()],
              dtype=bool,
          )
      )

    return edge_runtime_params.CombinedRuntimeParams(
        update_T_e=any_sub_model('update_T_e'),
        update_T_i=any_sub_model('update_T_i'),
        update_n_e=any_sub_model('update_n_e'),
        update_n_impurities=any_sub_model('update_n_impurities'),
        sub_models=sub_params,
    )


EdgeConfig = Annotated[
    extended_lengyel_pydantic_model.ExtendedLengyelConfig | CombinedEdgeConfig,
    pydantic.Field(discriminator='model_name'),
]

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
  corresponding `update_*` flag set. The flags are time-varying, but at any time
  at most one sub-model may have a given `update_*` flag set. The combined
  model's own `update_*` flags are derived from the sub-models (logical OR) and
  cannot be set directly.

  Attributes:
    model_name: Discriminator field set to `'combined'`.
    sub_models: Mapping from sub-model name to its `EdgeModelConfig`.
  """

  supported_bcs: ClassVar[base.SupportedBoundaryConditions] = (
      base.SupportedBoundaryConditions()
  )
  model_name: Annotated[Literal['combined'], torax_pydantic.JAX_STATIC] = (
      'combined'
  )
  sub_models: Mapping[str, SubModelConfig] = {}

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

  @pydantic.model_validator(mode='after')
  def _validate_unique_bc_providers(self) -> Self:
    """Ensures at most one sub-model updates each BC at any time."""
    for field in dataclasses.fields(base.SupportedBoundaryConditions):
      update_flag = f'update_{field.name}'
      flags = {
          name: getattr(sub_config, update_flag)
          for name, sub_config in self.sub_models.items()
      }
      if len(flags) < 2:
        continue
      # The flags are step functions, so it is sufficient to check at the union
      # of their time points.
      times = np.unique(
          np.concatenate([np.atleast_1d(f.time) for f in flags.values()])
      )
      for t in times:
        active = sorted(
            name for name, flag in flags.items() if bool(flag.get_value(t))
        )
        if len(active) > 1:
          raise ValueError(
              f"'{update_flag}' is True for multiple sub-models {active} at"
              f' t={float(t)}. At most one sub-model may update each boundary'
              ' condition at any time.'
          )
    return self

  def build_edge_model(self) -> base.CombinedEdgeModel:
    """Builds a CombinedEdgeModel from the configured sub-models."""
    return base.CombinedEdgeModel(
        sub_models={
            name: sub_config.build_edge_model()
            for name, sub_config in self.sub_models.items()
        }
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
        update_impurity=any_sub_model('update_impurity'),
        sub_models=sub_params,
    )


EdgeConfig = Annotated[
    extended_lengyel_pydantic_model.ExtendedLengyelConfig | CombinedEdgeConfig,
    pydantic.Field(discriminator='model_name'),
]

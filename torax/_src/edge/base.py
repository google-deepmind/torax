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

"""Base classes for edge models."""

import abc
from collections.abc import Mapping
import dataclasses
from typing import Annotated, Any, ClassVar, Self
import chex
import jax
import numpy as np
import pydantic
from torax._src import state
from torax._src import static_dataclass
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.edge import runtime_params as edge_runtime_params_lib
from torax._src.geometry import geometry
from torax._src.output_tools import output_grid_context
from torax._src.output_tools import output_keys
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.torax_pydantic import torax_pydantic
import xarray as xr

# pylint: disable=invalid-name


def pack_impurity_mapping(
    key: output_keys.OutputKey | str,
    mapping: Mapping[str, chex.Numeric] | None,
    *,
    dim_name: str,
) -> dict[str, output_grid_context.OutputVar]:
  """Packs an impurity mapping into a dictionary for to_output_dict."""
  if not mapping:
    return {}
  impurities = sorted(list(mapping.keys()))
  data_array = np.stack(
      [np.asarray(mapping[i]) for i in impurities],
      axis=0,
  )
  return {
      key: (
          (dim_name, output_keys.TIME),
          data_array,
          output_keys.get_units(key),
      )
  }


@dataclasses.dataclass(frozen=True)
class SupportedBoundaryConditions:
  """Specifies which boundary conditions an edge model supports computing.

  Attributes:
    T_e: Electron temperature boundary condition (T_e_right_bc).
    T_i: Ion temperature boundary condition (T_i_right_bc).
    n_e: Electron density boundary condition (n_e_right_bc).
    impurity: Impurity boundary conditions (impurity_right_bc).
  """

  T_e: bool = False
  T_i: bool = False
  n_e: bool = False
  impurity: bool = False


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class EdgeModelOutputs:
  """Base class for outputs from an edge model.

  Attributes:
    T_e_right_bc: Electron temperature boundary condition at LCFS [keV].
    T_i_right_bc: Ion temperature boundary condition at LCFS [keV].
    n_e_right_bc: Electron density boundary condition at LCFS [m^-3].
    impurity_right_bc: Mapping from impurity symbol to its right boundary
      condition (n_e_ratio at LCFS).
  """

  T_e_right_bc: jax.Array | None = None
  T_i_right_bc: jax.Array | None = None
  n_e_right_bc: jax.Array | None = None
  impurity_right_bc: Mapping[str, jax.Array] = dataclasses.field(
      default_factory=dict
  )

  def to_output_dict(
      self, context: output_grid_context.OutputGridContext
  ) -> dict[str, output_grid_context.OutputVar]:
    """Returns a dictionary of standard edge output variable tuples."""
    out_dict: dict[str, output_grid_context.OutputVar] = {}
    if self.T_e_right_bc is not None:
      out_dict[output_keys.T_E_RIGHT_BC] = context.pack(
          output_keys.T_E_RIGHT_BC, self.T_e_right_bc
      )
    if self.T_i_right_bc is not None:
      out_dict[output_keys.T_I_RIGHT_BC] = context.pack(
          output_keys.T_I_RIGHT_BC, self.T_i_right_bc
      )
    if self.n_e_right_bc is not None:
      out_dict[output_keys.N_E_RIGHT_BC] = context.pack(
          output_keys.N_E_RIGHT_BC, self.n_e_right_bc
      )
    out_dict.update(
        pack_impurity_mapping(
            output_keys.IMPURITY_RIGHT_BC,
            self.impurity_right_bc,
            dim_name=output_keys.IMPURITY,
        )
    )
    return out_dict

  def to_xr_datatree(
      self, context: output_grid_context.OutputGridContext
  ) -> xr.DataTree:
    """Builds an xr.DataTree of the edge model outputs."""
    coords: dict[str, Any] = {output_keys.TIME: context.times}
    if self.impurity_right_bc:
      coords[output_keys.IMPURITY] = sorted(list(self.impurity_right_bc.keys()))
    return xr.DataTree(
        dataset=context.build_dataset(
            self.to_output_dict(context),
            coords=coords,
        )
    )


@dataclasses.dataclass(frozen=True, eq=False)
class EdgeModel(static_dataclass.StaticDataclass, abc.ABC):
  """Abstract base class for edge models.

  Boundary conditions that a model does not compute should be returned as
  `None` in its `EdgeModelOutputs`.
  """

  @abc.abstractmethod
  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      core_sources: source_profiles_lib.SourceProfiles,
      previous_edge_outputs: EdgeModelOutputs | None = None,
  ) -> EdgeModelOutputs:
    """Evaluates the edge model at the given time."""


class EdgeModelConfig(torax_pydantic.BaseModelFrozen, abc.ABC):
  """Base pydantic configuration for all edge models.

  Subclasses implement the specific model logic, and must override `model_name`
  with a `Literal` to serve as a discriminator in polymorphic unions.

  Attributes:
    model_name: Discriminator field for Pydantic. Subclasses must override with
      a `Literal` value.
    supported_bcs: Class-level declaration of which boundary conditions this
      model is capable of computing. Defaults to all False, so subclasses must
      declare the boundary conditions they support.
    update_T_e: Whether to update electron temperature boundary condition.
      Defaults to `supported_bcs.T_e`.
    update_T_i: Whether to update ion temperature boundary condition. Defaults
      to `supported_bcs.T_i`.
    update_n_e: Whether to update electron density boundary condition. Defaults
      to `supported_bcs.n_e`.
    update_impurity: Whether to update impurity concentrations in the core.
      Defaults to `supported_bcs.impurity`.
  """

  model_name: Annotated[str, torax_pydantic.JAX_STATIC] = ''
  supported_bcs: ClassVar[SupportedBoundaryConditions] = (
      SupportedBoundaryConditions(
          T_e=False,
          T_i=False,
          n_e=False,
          impurity=False,
      )
  )
  update_T_e: torax_pydantic.TimeVaryingScalarStep = (
      torax_pydantic.ValidatedDefault(False)
  )
  update_T_i: torax_pydantic.TimeVaryingScalarStep = (
      torax_pydantic.ValidatedDefault(False)
  )
  update_n_e: torax_pydantic.TimeVaryingScalarStep = (
      torax_pydantic.ValidatedDefault(False)
  )
  update_impurity: torax_pydantic.TimeVaryingScalarStep = (
      torax_pydantic.ValidatedDefault(False)
  )

  @pydantic.model_validator(mode='before')
  @classmethod
  def _default_update_flags(cls, data: Any) -> Any:
    """Defaults any unset `update_<bc>` flag to `supported_bcs.<bc>`."""
    if not isinstance(data, Mapping):
      return data
    configurable_data = dict(data)
    for field in dataclasses.fields(SupportedBoundaryConditions):
      configurable_data.setdefault(
          f'update_{field.name}', getattr(cls.supported_bcs, field.name)
      )
    return configurable_data

  @pydantic.model_validator(mode='after')
  def _validate_update_flags(self) -> Self:
    """Ensures no `update_<bc>` flag is True for an unsupported BC."""
    for field in dataclasses.fields(SupportedBoundaryConditions):
      update_flag = f'update_{field.name}'
      if not getattr(self.supported_bcs, field.name) and np.any(
          getattr(self, update_flag).value
      ):
        raise ValueError(
            f"'{update_flag}' cannot be True because '{self.model_name}'"
            f" does not support the '{field.name}' boundary condition."
        )
    return self

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> edge_runtime_params_lib.RuntimeParams:
    """Builds the runtime parameters for the edge model at time t."""
    return edge_runtime_params_lib.RuntimeParams(
        update_T_e=self.update_T_e.get_value(t),
        update_T_i=self.update_T_i.get_value(t),
        update_n_e=self.update_n_e.get_value(t),
        update_impurity=self.update_impurity.get_value(t),
    )

  @abc.abstractmethod
  def build_edge_model(self) -> EdgeModel:
    """Builds an edge model from the config."""

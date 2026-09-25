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
from collections.abc import Mapping, Sequence
import dataclasses
from typing import Annotated, Any
import chex
import jax
import numpy as np
from torax._src import state
from torax._src import static_dataclass
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.edge import runtime_params as edge_runtime_params
from torax._src.geometry import geometry
from torax._src.output_tools import output_grid_context
from torax._src.output_tools import output_keys
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.torax_pydantic import torax_pydantic
import xarray as xr

# pylint: disable=invalid-name


def pack_impurity_mapping(
    key: output_keys.OutputKey | str,
    mapping: Mapping[str, chex.Numeric],
    *,
    dim_name: str = output_keys.IMPURITY,
    impurities: Sequence[str] | None = None,
) -> output_grid_context.OutputVar:
  """Packs an impurity mapping into an OutputVar tuple."""
  if not mapping:
    raise ValueError(f"Cannot pack empty impurity mapping for '{key}'.")
  if impurities is None:
    impurities = sorted(list(mapping.keys()))
  data_array = np.stack(
      [np.asarray(mapping[i]) for i in impurities],
      axis=0,
  )
  return (
      (dim_name, output_keys.TIME),
      data_array,
      output_keys.get_units(key),
  )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class EdgeModelOutputs:
  """Base class for outputs from an edge model.

  Attributes:
    T_e_right_bc: Electron temperature boundary condition at LCFS [keV].
    T_i_right_bc: Ion temperature boundary condition at LCFS [keV].
    n_e_right_bc: Electron density boundary condition at LCFS [m^-3].
    impurity_right_bc: Mapping from impurity symbol to its right boundary
      condition (n_e_ratio at LCFS).
  """

  T_e_right_bc: jax.Array
  T_i_right_bc: jax.Array
  n_e_right_bc: jax.Array
  impurity_right_bc: Mapping[str, jax.Array]

  def to_output_dict(
      self, context: output_grid_context.OutputGridContext
  ) -> dict[str, output_grid_context.OutputVar]:
    """Returns a dictionary of standard edge output variable tuples."""
    out_dict: dict[str, output_grid_context.OutputVar] = {}

    if self.T_e_right_bc is not None:
      out_dict[str(output_keys.T_E_RIGHT_BC)] = context.pack(
          output_keys.T_E_RIGHT_BC, self.T_e_right_bc
      )
    if self.T_i_right_bc is not None:
      out_dict[str(output_keys.T_I_RIGHT_BC)] = context.pack(
          output_keys.T_I_RIGHT_BC, self.T_i_right_bc
      )
    if self.n_e_right_bc is not None:
      out_dict[str(output_keys.N_E_RIGHT_BC)] = context.pack(
          output_keys.N_E_RIGHT_BC, self.n_e_right_bc
      )
    if self.impurity_right_bc:
      out_dict[str(output_keys.IMPURITY_RIGHT_BC)] = pack_impurity_mapping(
          output_keys.IMPURITY_RIGHT_BC, self.impurity_right_bc
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
  """Abstract base class for edge models."""

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
    update_temperatures: Whether to update temperature boundary conditions.
    update_density: Whether to update electron density boundary condition.
    update_impurities: Whether to update impurity concentrations in the core.
  """

  model_name: Annotated[str, torax_pydantic.JAX_STATIC] = ""
  update_temperatures: torax_pydantic.TimeVaryingScalarStep = (
      torax_pydantic.ValidatedDefault(False)
  )
  update_density: torax_pydantic.TimeVaryingScalarStep = (
      torax_pydantic.ValidatedDefault(False)
  )
  update_impurities: torax_pydantic.TimeVaryingScalarStep = (
      torax_pydantic.ValidatedDefault(False)
  )

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> edge_runtime_params.RuntimeParams:
    """Builds the runtime parameters for the edge model at time t."""
    return edge_runtime_params.RuntimeParams(
        update_temperatures=self.update_temperatures.get_value(t),
        update_density=self.update_density.get_value(t),
        update_impurities=self.update_impurities.get_value(t),
    )

  @abc.abstractmethod
  def build_edge_model(self) -> EdgeModel:
    """Builds an edge model from the config."""

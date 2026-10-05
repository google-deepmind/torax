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
import copy
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

BC_TO_UPDATE_FLAG = (
    ('electron_temperature', 'update_electron_temperature'),
    ('ion_temperature', 'update_ion_temperature'),
    ('electron_density', 'update_electron_density'),
    ('impurities', 'update_impurities'),
)


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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class BoundaryConditions(static_dataclass.StaticDataclass):
  """Specifies which boundary conditions an edge model computes or uses.

  Attributes:
    electron_temperature: Electron temperature boundary condition
      (T_e_right_bc).
    ion_temperature: Ion temperature boundary condition (T_i_right_bc).
    electron_density: Electron density boundary condition (n_e_right_bc).
    impurities: Impurity boundary conditions (impurity_right_bc).
  """

  electron_temperature: bool = dataclasses.field(
      default=False, metadata={'static': True}
  )
  ion_temperature: bool = dataclasses.field(
      default=False, metadata={'static': True}
  )
  electron_density: bool = dataclasses.field(
      default=False, metadata={'static': True}
  )
  impurities: bool = dataclasses.field(
      default=False, metadata={'static': True}
  )


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

  Attributes:
    computed_bcs: Class-level declaration of which boundary conditions this
      model is capable of computing.
    used_bcs: Which of the computed boundary conditions are used from this
      model. Defaults to `computed_bcs`.
  """

  computed_bcs: ClassVar[BoundaryConditions] = BoundaryConditions(
      electron_temperature=True,
      ion_temperature=True,
      electron_density=True,
      impurities=True,
  )
  used_bcs: BoundaryConditions = dataclasses.field(  # pyrefly: ignore[bad-assignment]
      default=None, kw_only=True
  )

  def __post_init__(self):
    if self.used_bcs is None:
      object.__setattr__(self, 'used_bcs', self.computed_bcs)
    super().__post_init__()

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
    computed_bcs: Class-level declaration of which boundary conditions this
      model is capable of computing. Defaults to all True.
    used_bcs: Which of the computed boundary conditions are used from this
      model. Defaults to `computed_bcs`.
    update_electron_temperature: Whether to update electron temperature boundary
      condition.
    update_ion_temperature: Whether to update ion temperature boundary
      condition.
    update_electron_density: Whether to update electron density boundary
      condition.
    update_impurities: Whether to update impurity concentrations in the core.
  """

  model_name: Annotated[str, torax_pydantic.JAX_STATIC] = ''
  computed_bcs: ClassVar[BoundaryConditions] = BoundaryConditions(
      electron_temperature=True,
      ion_temperature=True,
      electron_density=True,
      impurities=True,
  )
  used_bcs: Annotated[BoundaryConditions, torax_pydantic.JAX_STATIC] = (
      BoundaryConditions()
  )
  update_electron_temperature: torax_pydantic.TimeVaryingScalarStep = (
      torax_pydantic.ValidatedDefault(False)
  )
  update_ion_temperature: torax_pydantic.TimeVaryingScalarStep = (
      torax_pydantic.ValidatedDefault(False)
  )
  update_electron_density: torax_pydantic.TimeVaryingScalarStep = (
      torax_pydantic.ValidatedDefault(False)
  )
  update_impurities: torax_pydantic.TimeVaryingScalarStep = (
      torax_pydantic.ValidatedDefault(False)
  )

  @pydantic.model_validator(mode='before')
  @classmethod
  def _default_used_bcs(cls, data: Any) -> Any:
    if not isinstance(data, Mapping):
      return data
    configurable_data = copy.deepcopy(dict(data))
    if 'used_bcs' not in configurable_data:
      configurable_data['used_bcs'] = cls.computed_bcs
    elif isinstance(configurable_data['used_bcs'], Mapping):
      configurable_data['used_bcs'] = BoundaryConditions(
          **configurable_data['used_bcs']
      )
    used_bcs = configurable_data['used_bcs']
    if (
        isinstance(used_bcs, BoundaryConditions)
        and cls.computed_bcs != BoundaryConditions()
    ):
      for bc_name, update_flag in BC_TO_UPDATE_FLAG:
        if (
            not getattr(used_bcs, bc_name)
            and update_flag not in configurable_data
        ):
          configurable_data[update_flag] = False
    return configurable_data

  @pydantic.model_validator(mode='after')
  def _validate_bcs_and_update_flags(self) -> Self:
    for bc_name, update_flag in BC_TO_UPDATE_FLAG:
      if getattr(self.used_bcs, bc_name) and not getattr(
          self.computed_bcs, bc_name
      ):
        raise ValueError(
            f"Boundary condition '{bc_name}' cannot be True in 'used_bcs'"
            f" because '{self.model_name}' does not compute it."
        )
      if (
          self.computed_bcs != BoundaryConditions()
          and not getattr(self.used_bcs, bc_name)
          and np.any(getattr(self, update_flag).value)
      ):
        raise ValueError(
            f"'{update_flag}' cannot be True when '{bc_name}' is False in"
            " 'used_bcs'."
        )
    return self

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> edge_runtime_params_lib.RuntimeParams:
    """Builds the runtime parameters for the edge model at time t."""
    return edge_runtime_params_lib.RuntimeParams(
        update_electron_temperature=self.update_electron_temperature.get_value(
            t
        ),
        update_ion_temperature=self.update_ion_temperature.get_value(t),
        update_electron_density=self.update_electron_density.get_value(t),
        update_impurities=self.update_impurities.get_value(t),
    )

  @abc.abstractmethod
  def build_edge_model(self) -> EdgeModel:
    """Builds an edge model from the config."""

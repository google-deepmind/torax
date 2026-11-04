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
from jax import numpy as jnp
import numpy as np
import pydantic
from torax._src import static_dataclass
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import core_profiles as core_profiles_lib
from torax._src.edge import edge_state as edge_state_lib
from torax._src.edge import runtime_params as edge_runtime_params_lib
from torax._src.geometry import geometry
from torax._src.output_tools import output_grid_context
from torax._src.output_tools import output_keys
from torax._src.sources import core_sources as core_sources_lib
from torax._src.torax_pydantic import torax_pydantic
import xarray as xr

# pylint: disable=invalid-name


def pack_impurity_mapping(
    key: output_keys.OutputKey | str,
    mapping: Mapping[str, chex.Numeric] | None,
    *,
    context: output_grid_context.OutputGridContext,
    dim_name: str,
) -> dict[str, output_grid_context.OutputVar]:
  """Packs an impurity mapping into a dictionary for to_output_dict."""
  if not mapping:
    return {}
  impurities = sorted(list(mapping.keys()))
  context.register_coord(dim_name, np.asarray(impurities))
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
    n_impurities: Impurity boundary conditions (impurity_right_bc).
  """

  T_e: bool = False
  T_i: bool = False
  n_e: bool = False
  n_impurities: bool = False


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
    valid_for_core_update: Whether the edge outputs are valid for updating core
      boundary conditions.
  """

  T_e_right_bc: jax.Array | None = None
  T_i_right_bc: jax.Array | None = None
  n_e_right_bc: jax.Array | None = None
  impurity_right_bc: Mapping[str, jax.Array] = dataclasses.field(
      default_factory=dict
  )
  valid_for_core_update: jax.Array = dataclasses.field(
      default_factory=lambda: jnp.array(True, dtype=jnp.bool_),
      kw_only=True,
  )

  def to_output_dict(
      self, context: output_grid_context.OutputGridContext
  ) -> dict[str, output_grid_context.OutputVar]:
    """Packs time-stacked edge model outputs into xarray-ready variables.

    Args:
      context: Grid context providing time coordinates for shape validation and
        `(dims, data, attrs)` packing. Mutated in-place to register the
        `impurity` coordinate when `impurity_right_bc` is non-empty.

    Returns:
      Mapping from boundary condition variable names to `(dims, data, attrs)`
      `OutputVar` tuples (omitting unset or empty fields):
        - `T_e_right_bc`, `T_i_right_bc`, `n_e_right_bc`: `('time',)` with shape
          `(n_time,)`.
        - `impurity_right_bc`: `('impurity', 'time')` with shape
          `(n_impurities, n_time)`.
    """
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
            context=context,
            dim_name=output_keys.IMPURITY,
        )
    )
    return out_dict

  def to_xr_datatree(
      self, context: output_grid_context.OutputGridContext
  ) -> xr.DataTree:
    """Builds an xr.DataTree of the edge model outputs."""
    return xr.DataTree(
        dataset=context.build_dataset(self.to_output_dict(context))
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class CombinedEdgeOutputs(EdgeModelOutputs):
  """Outputs from a CombinedEdgeModel.

  Attributes:
    sub_models: Mapping from sub-model name to its EdgeModelOutputs.
  """

  sub_models: Mapping[str, EdgeModelOutputs] = dataclasses.field(
      default_factory=dict
  )

  def to_xr_datatree(
      self, context: output_grid_context.OutputGridContext
  ) -> xr.DataTree:
    """Builds an xr.DataTree of the combined edge model and sub-models."""
    base_tree = super().to_xr_datatree(context)
    children = {
        name: sub_out.to_xr_datatree(context)
        for name, sub_out in self.sub_models.items()
    }
    return xr.DataTree(
        dataset=base_tree.dataset,
        children=children,
    )


@dataclasses.dataclass(frozen=True, eq=False)
class EdgeModel(static_dataclass.StaticDataclass, abc.ABC):
  """Abstract base class for edge models."""

  def initial_state(
      self,
      edge_runtime_params: edge_runtime_params_lib.RuntimeParams,
  ) -> edge_state_lib.EdgeState:
    """Returns the initial cross-timestep EdgeState for this model.

    Stateless edge models inherit this default returning an empty EdgeState(),
    while stateful subclasses override it with their initial state.

    Args:
      edge_runtime_params: Runtime parameters for the edge model.

    Returns:
      Initial EdgeState.
    """
    del edge_runtime_params
    return edge_state_lib.EdgeState()

  def initial_state_from_datatree(
      self,
      datatree: xr.DataTree | None,
      edge_runtime_params: edge_runtime_params_lib.RuntimeParams,
  ) -> edge_state_lib.EdgeState:
    """Reconstructs EdgeState at restart time from a single-time-slice DataTree.

    During simulation restart from file, this method reconstructs the initial
    `EdgeState` at the restart time from an xr.DataTree with the `time`
    dimension already selected (e.g. via `.sel(time=...)`), or `None` if no
    edge state was saved in the file.

    Stateless edge models inherit this default implementation returning an empty
    `EdgeState()`. Stateful subclasses (e.g. `ExtendedLengyelModel`,
    `CombinedEdgeModel`) override this method to deserialize their state.

    Args:
      datatree: Single-time-slice DataTree containing edge state, or None.
      edge_runtime_params: Runtime parameters for the edge model.

    Returns:
      Reconstructed EdgeState at restart time.
    """
    del datatree, edge_runtime_params
    return edge_state_lib.EdgeState()

  @abc.abstractmethod
  def __call__(
      self,
      edge_runtime_params: edge_runtime_params_lib.RuntimeParams,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: core_profiles_lib.CoreProfiles,
      core_sources: core_sources_lib.CoreSources,
      edge_state: edge_state_lib.EdgeState,
  ) -> tuple[EdgeModelOutputs, edge_state_lib.EdgeState]:
    """Evaluates the edge model at the given time.

    Args:
      edge_runtime_params: Runtime parameters for the edge model.
      runtime_params: Runtime parameters for the simulation.
      geo: The magnetic geometry of the tokamak.
      core_profiles: Core plasma profiles.
      core_sources: Core plasma sources.
      edge_state: Cross-timestep state for the edge model.

    Returns:
      A tuple of (EdgeModelOutputs, updated_edge_state).
    """


@dataclasses.dataclass(frozen=True, eq=False)
class CombinedEdgeModel(EdgeModel):
  """Edge model composed of multiple sub-models.

  Each boundary condition is taken from the single sub-model configured to
  update it (if any).
  """

  sub_models: Mapping[str, EdgeModel] = dataclasses.field(default_factory=dict)
  T_e_sub_model: str | None = None
  T_i_sub_model: str | None = None
  n_e_sub_model: str | None = None
  n_impurities_sub_model: str | None = None

  def initial_state(
      self,
      edge_runtime_params: edge_runtime_params_lib.RuntimeParams,
  ) -> edge_state_lib.CombinedEdgeState:
    """Returns the initial cross-timestep CombinedEdgeState."""
    assert isinstance(
        edge_runtime_params, edge_runtime_params_lib.CombinedRuntimeParams
    )
    return edge_state_lib.CombinedEdgeState(
        sub_models={
            name: sub_model.initial_state(
                edge_runtime_params.sub_models[name],
            )
            for name, sub_model in self.sub_models.items()
        }
    )

  def initial_state_from_datatree(
      self,
      datatree: xr.DataTree | None,
      edge_runtime_params: edge_runtime_params_lib.RuntimeParams,
  ) -> edge_state_lib.CombinedEdgeState:
    """Reconstructs CombinedEdgeState from a single-time-slice DataTree.

    During simulation restart from file, this method reconstructs the initial
    `CombinedEdgeState` at the restart time by delegating reconstruction of each
    sub-model's state to that sub-model's `initial_state_from_datatree` method
    using child nodes of the single-time-slice `datatree`.

    Args:
      datatree: Single-time-slice DataTree containing edge state, or None.
      edge_runtime_params: Runtime parameters for the combined edge model.

    Returns:
      Reconstructed CombinedEdgeState at restart time.
    """
    assert isinstance(
        edge_runtime_params, edge_runtime_params_lib.CombinedRuntimeParams
    )
    children = datatree.children if datatree is not None else {}
    return edge_state_lib.CombinedEdgeState(
        sub_models={
            name: sub_model.initial_state_from_datatree(
                children.get(name),
                edge_runtime_params.sub_models[name],
            )
            for name, sub_model in self.sub_models.items()
        }
    )

  def __call__(
      self,
      edge_runtime_params: edge_runtime_params_lib.RuntimeParams,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: core_profiles_lib.CoreProfiles,
      core_sources: core_sources_lib.CoreSources,
      edge_state: edge_state_lib.EdgeState,
  ) -> tuple[CombinedEdgeOutputs, edge_state_lib.CombinedEdgeState]:
    """Evaluates the edge sub-models and merges their BCs."""
    assert isinstance(
        edge_runtime_params, edge_runtime_params_lib.CombinedRuntimeParams
    )
    edge_params: edge_runtime_params_lib.CombinedRuntimeParams = (
        edge_runtime_params
    )
    assert isinstance(edge_state, edge_state_lib.CombinedEdgeState)
    sub_outputs = {}
    new_sub_states = {}
    for name, sub_model in self.sub_models.items():
      out, sub_state = sub_model(
          edge_params.sub_models[name],
          runtime_params,
          geo,
          core_profiles,
          core_sources,
          edge_state=edge_state.sub_models[name],
      )
      sub_outputs[name] = out
      new_sub_states[name] = sub_state

    outputs = CombinedEdgeOutputs(
        T_e_right_bc=(
            sub_outputs[self.T_e_sub_model].T_e_right_bc
            if self.T_e_sub_model is not None
            else None
        ),
        T_i_right_bc=(
            sub_outputs[self.T_i_sub_model].T_i_right_bc
            if self.T_i_sub_model is not None
            else None
        ),
        n_e_right_bc=(
            sub_outputs[self.n_e_sub_model].n_e_right_bc
            if self.n_e_sub_model is not None
            else None
        ),
        impurity_right_bc=(
            sub_outputs[self.n_impurities_sub_model].impurity_right_bc
            if self.n_impurities_sub_model is not None
            else {}
        ),
        sub_models=sub_outputs,
    )
    return outputs, edge_state_lib.CombinedEdgeState(sub_models=new_sub_states)


class EdgeModelConfig(torax_pydantic.ConfigBase):
  """Base pydantic configuration for all edge models.

  Subclasses implement the specific model logic, and must override `model_name`
  with a `Literal` to serve as a discriminator in polymorphic unions.

  Attributes:
    model_name: The edge model discriminator tag. Set to `'extended_lengyel'`
      for the built-in extended Lengyel model, `'combined'` when combining
      multiple edge sub-models under `sub_models`, or the registered model name
      for a custom edge model.
    supported_boundary_conditions: Class-level `SupportedBoundaryConditions`
      instance whose fields (`T_e`, `T_i`, `n_e`, `n_impurities`) declare which
      boundary conditions the subclass can compute and set the defaults for the
      corresponding `update_*` flags below.
    update_T_e: If `True`, update core electron temperature boundary condition
      based on edge model results. Defaults to
      `supported_boundary_conditions.T_e`, and may only be `True` if supported.
    update_T_i: If `True`, update core ion temperature boundary condition based
      on edge model results. Defaults to `supported_boundary_conditions.T_i`,
      and may only be `True` if supported.
    update_n_e: If `True`, update core electron density boundary condition based
      on edge model results. Defaults to `supported_boundary_conditions.n_e`,
      and may only be `True` if supported.
    update_n_impurities: If `True`, update core impurity profiles based on edge
      model results. Defaults to `supported_boundary_conditions.n_impurities`,
      and may only be `True` if supported.
  """

  model_name: Annotated[str, torax_pydantic.JAX_STATIC]
  supported_boundary_conditions: ClassVar[SupportedBoundaryConditions]
  update_T_e: torax_pydantic.TimeVaryingScalarStep
  update_T_i: torax_pydantic.TimeVaryingScalarStep
  update_n_e: torax_pydantic.TimeVaryingScalarStep
  update_n_impurities: torax_pydantic.TimeVaryingScalarStep

  @pydantic.model_validator(mode='before')
  @classmethod
  def _default_update_flags(cls, data: Any) -> Any:
    """Defaults any unset `update_<bc>` flag from `supported_boundary_conditions`."""
    if not hasattr(cls, 'supported_boundary_conditions'):
      raise ValueError('supported_boundary_conditions must be set.')
    if not isinstance(data, Mapping):
      return data
    configurable_data = dict(data)
    for field in dataclasses.fields(SupportedBoundaryConditions):
      configurable_data.setdefault(
          f'update_{field.name}',
          getattr(cls.supported_boundary_conditions, field.name),
      )
    return configurable_data

  @pydantic.model_validator(mode='after')
  def _validate_update_flags(self) -> Self:
    """Ensures no `update_<bc>` flag is True for an unsupported BC."""
    for field in dataclasses.fields(SupportedBoundaryConditions):
      update_flag = f'update_{field.name}'
      if not getattr(self.supported_boundary_conditions, field.name) and np.any(
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
    """Builds the runtime parameters for the edge model at time `t`."""
    return edge_runtime_params_lib.RuntimeParams(
        update_T_e=self.update_T_e.get_value(t),
        update_T_i=self.update_T_i.get_value(t),
        update_n_e=self.update_n_e.get_value(t),
        update_n_impurities=self.update_n_impurities.get_value(t),
    )

  @abc.abstractmethod
  def build_edge_model(self) -> EdgeModel:
    """Builds an edge model from the config."""

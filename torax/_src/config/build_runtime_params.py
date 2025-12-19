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
"""Methods for building simulation parameters.

 - `RuntimeParamsProvider` which provides a the `RuntimeParams` to
  use during time t of the sim.
 - `get_consistent_params_and_geometry` which returns a
`RuntimeParams` and a corresponding `Geometry` with consistent `Ip`. Also
  optionally updates temperature boundary conditions and impurity
  concentrations based on the edge model outputs.
"""
import dataclasses
from typing import Any, Callable, Mapping, Sequence, TypeAlias

import chex
import equinox as eqx
import jax
from jax import numpy as jnp
from torax._src.config import numerics as numerics_lib
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import profile_conditions as profile_conditions_lib
from torax._src.core_profiles.plasma_composition import plasma_composition as plasma_composition_lib
from torax._src.edge import base as edge_base
from torax._src.edge import updaters as edge_updaters
from torax._src.geometry import geometry
from torax._src.geometry import geometry_provider as geometry_provider_lib
from torax._src.mhd import pydantic_model as mhd_pydantic_model
from torax._src.neoclassical import pydantic_model as neoclassical_pydantic_model
from torax._src.pedestal_model import pydantic_model as pedestal_pydantic_model
from torax._src.solver import pydantic_model as solver_pydantic_model
from torax._src.sources import pydantic_model as sources_pydantic_model
from torax._src.time_step_calculator import pydantic_model as time_step_calculator_pydantic_model
from torax._src.torax_pydantic import interpolated_param_1d
from torax._src.torax_pydantic import interpolated_param_2d
from torax._src.torax_pydantic import model_config
from torax._src.transport_model import pydantic_model as transport_pydantic_model
import typing_extensions

# pylint: disable=invalid-name


ReplaceablePytreeNodes: TypeAlias = (
    interpolated_param_1d.TimeVaryingScalar
    | interpolated_param_2d.TimeVaryingArray
    | chex.Numeric
)
ValidUpdates: TypeAlias = (
    interpolated_param_1d.TimeVaryingScalarUpdate
    | interpolated_param_2d.TimeVaryingArrayUpdate
    | chex.Numeric
)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParamsProvider:
  """Provides a RuntimeParamsSlice to use during time t of the sim.

  The RuntimeParams may change from time step to time step, so this
  class interpolates any time-dependent params in the input config to the values
  they should be at time t.

  NOTE: In order to maintain consistency between the RuntimeParams
  and the geometry, `get_consistent_runtime_params_and_geometry`
  should be used to get a slice of the RuntimeParams and a
  corresponding geometry.
  """

  sources: sources_pydantic_model.Sources
  numerics: numerics_lib.Numerics
  profile_conditions: profile_conditions_lib.ProfileConditions
  plasma_composition: plasma_composition_lib.PlasmaComposition
  transport_model: transport_pydantic_model.TransportConfig
  solver: solver_pydantic_model.SolverConfig
  pedestal: pedestal_pydantic_model.PedestalConfig
  mhd: mhd_pydantic_model.MHD
  edge: edge_base.EdgeModelConfig | None
  neoclassical: neoclassical_pydantic_model.Neoclassical
  time_step_calculator: time_step_calculator_pydantic_model.TimeStepCalculator

  @classmethod
  def from_config(
      cls,
      config: model_config.ToraxConfig,
  ) -> typing_extensions.Self:
    """Constructs a RuntimeParamsProvider from a ToraxConfig."""
    return cls(
        sources=config.sources,
        numerics=config.numerics,
        profile_conditions=config.profile_conditions,
        plasma_composition=config.plasma_composition,
        transport_model=config.transport,
        solver=config.solver,
        pedestal=config.pedestal,
        mhd=config.mhd,
        edge=config.edge,
        neoclassical=config.neoclassical,
        time_step_calculator=config.time_step_calculator,
    )

  # TODO(b/460347309): investigate effect of jit here on overall compile time.
  def __call__(
      self,
      t: chex.Numeric,
  ) -> runtime_params_lib.RuntimeParams:
    """Returns a runtime_params_slice.RuntimeParams to use during time t of the sim."""
    return runtime_params_lib.RuntimeParams(
        transport=self.transport_model.build_runtime_params(t),
        solver=self.solver.build_runtime_params,
        sources={
            source_name: source_config.build_runtime_params(t)
            for source_name, source_config in dict(self.sources).items()
            if source_config is not None
        },
        plasma_composition=self.plasma_composition.build_runtime_params(t),
        profile_conditions=self.profile_conditions.build_runtime_params(t),
        numerics=self.numerics.build_runtime_params(t),
        neoclassical=self.neoclassical.build_runtime_params(),
        pedestal=self.pedestal.build_runtime_params(t),
        mhd=self.mhd.build_runtime_params(t),
        time_step_calculator=self.time_step_calculator.build_runtime_params(),
        edge=None if self.edge is None else self.edge.build_runtime_params(t),
    )

  def update_provider(
      self,
      get_nodes_to_replace: Callable[
          [typing_extensions.Self],
          Sequence[ReplaceablePytreeNodes],
      ],
      replacement_values: Sequence[ValidUpdates],
  ) -> typing_extensions.Self:
    """Updates a provider with new values. Works under `jax.jit`.

    Example usage:
    ```
    ip_update = interpolated_param_1d.TimeVaryingScalarReplace(
        value=new_ip_value,
    )
    T_e_update = interpolated_param_2d.TimeVaryingArrayReplace(
        value=T_e_cell_value * 3.0,
        rho_norm=T_e.grid.cell_centers,
    )
    new_provider = provider.update_provider(
        # ordering of the replace values and return values must match.
        lambda x: (x.profile_conditions.Ip, x.profile_conditions.T_e),
        (ip_update, T_e_update),
    )
    ```

    Args:
      get_nodes_to_replace: A function that takes a provider and returns a tuple
        of nodes to replace. See above for an example. The returned nodes must
        be one of the following types: `TimeVaryingScalar`, `TimeVaryingArray`,
        `chex.Numeric`.
      replacement_values: A tuple of values to replace the nodes with.

    Returns:
      A new provider with the updated values.
    """
    # Parse `TimeVaryingArrayReplace` -> `TimeVaryingArray` and
    # `TimeVaryingScalarReplace` -> `TimeVaryingScalar`.
    new_provider_values = []
    for leaf, replace_value in zip(
        get_nodes_to_replace(self), replacement_values, strict=True
    ):
      new_provider_values.append(
          _get_provider_value_from_replace_value(leaf, replace_value)
      )

    return eqx.tree_at(get_nodes_to_replace, self, replace=new_provider_values)

  def get_node_from_path(self, path: str) -> Any:
    """Iteratively call `getattr` on `self` from dot-separated path of attrs."""
    x = self
    attributes = path.split(".")
    for attr in attributes:
      try:
        x = getattr(x, attr)
      except AttributeError as exc:
        raise ValueError(f"Attribute {attr} of {path} not found.") from exc
    return x

  def update_provider_from_mapping(
      self, replacements: Mapping[str, ValidUpdates]
  ) -> typing_extensions.Self:
    """Update a provider from a mapping of replacements.

    Example usage:
    ```
    ip_update = interpolated_param_1d.TimeVaryingScalarReplace(
        value=new_ip_value,
    )
    T_e_update = interpolated_param_2d.TimeVaryingArrayReplace(
        cell_value=T_e_cell_value * 3.0,
        rho_norm=T_e.grid.cell_centers,
    )
    new_provider = provider.update_provider_from_mapping(
        {
            'profile_conditions.Ip': ip_update,
            'profile_conditions.T_e': T_e_update,
            'sources.ei_exchange.Qei_multiplier': 2.0,
        }
    )
    ```

    Args:
      replacements: A mapping of node paths to replacement values. Paths are of
        the form `'some.path.to.field_name'` and the `value` is the new value
        depending on the type of the node. The path can be dictionary keys or
        attribute names with field_name pointing to one of the following types:
        {`TimeVaryingScalar`, `TimeVaryingArray`, `chex.Numeric`}.

    Returns:
      A new provider with the updated values.
    """

    def get_replacements(
        provider: typing_extensions.Self,
    ) -> list[ReplaceablePytreeNodes]:
      """Returns the nodes to replace."""
      nodes_to_replace: list[ReplaceablePytreeNodes] = []

      for key in replacements.keys():
        x = provider.get_node_from_path(key)
        nodes_to_replace.append(x)
      return nodes_to_replace

    return self.update_provider(get_replacements, tuple(replacements.values()))


def _get_provider_value_from_replace_value(
    leaf: ReplaceablePytreeNodes,
    replace_value: ValidUpdates,
) -> ReplaceablePytreeNodes:
  """Validate and convert any replacement value to the correct type."""
  match leaf:
    case interpolated_param_1d.TimeVaryingScalar():
      if not isinstance(
          replace_value, interpolated_param_1d.TimeVaryingScalarUpdate
      ):
        raise ValueError(
            "To replace a `TimeVaryingScalar` use a"
            f" `TimeVaryingScalarReplace`, got {type(replace_value)} instead."
        )
      return leaf.update(replace_value)
    case interpolated_param_2d.TimeVaryingArray():
      if not isinstance(
          replace_value, interpolated_param_2d.TimeVaryingArrayUpdate
      ):
        raise ValueError(
            "To replace a `TimeVaryingArray` use a `TimeVaryingArrayReplace`,"
            f" got {type(replace_value)} instead."
        )
      return leaf.update(replace_value)
    case _ if isinstance(leaf, (chex.Array, float)):
      if not isinstance(replace_value, (chex.Array, float, jax.core.Tracer)):
        raise ValueError(
            "To replace a scalar or `Array` pass a scalar or `Array`,"
            f" got {type(replace_value)} instead."
        )
      leaf = jnp.asarray(leaf)
      replace_value = jnp.asarray(replace_value)
      if leaf.shape != replace_value.shape or leaf.dtype != replace_value.dtype:
        raise ValueError(
            "The shape of the replacement value must match the shape of the"
            f" leaf, Got leaf: shape={leaf.shape}, dtype={leaf.dtype},"
            f" replace_value: shape={replace_value.shape},"
            f" dtype={replace_value.dtype}."
        )
      return replace_value
    case _:
      raise ValueError(
          "Only a scalar, `TimeVaryingScalar` or `TimeVaryingArray` can be"
          f" replaced, got {type(leaf)} instead."
      )


def get_consistent_runtime_params_and_geometry(
    *,
    t: chex.Numeric,
    runtime_params_provider: RuntimeParamsProvider,
    geometry_provider: geometry_provider_lib.GeometryProvider,
    edge_outputs: edge_base.EdgeModelOutputs | None = None,
) -> tuple[runtime_params_lib.RuntimeParams, geometry.Geometry]:
  """Returns the runtime params and geometry for a given time."""
  geo = geometry_provider(t)
  runtime_params_from_provider = runtime_params_provider(t=t)
  runtime_params = edge_updaters.update_runtime_params(
      runtime_params_from_provider, edge_outputs
  )
  return runtime_params_lib.make_ip_consistent(runtime_params, geo)


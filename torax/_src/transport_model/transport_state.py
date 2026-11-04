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

"""Cross-timestep state for transport models."""

from collections.abc import Mapping
import dataclasses
import jax
from torax._src.output_tools import output_grid_context
from torax._src.output_tools import output_keys
import xarray as xr


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ComponentTransportState:
  """Base class for component transport model cross-timestep state."""

  def to_xr_datatree(
      self,
      context: output_grid_context.OutputGridContext,
  ) -> xr.DataTree | None:
    """Builds an xr.DataTree of component transport state, or None.

    Args:
      context: Output grid context for building datasets.

    Returns:
      An xr.DataTree of component state, or None if stateless.
    """
    del context
    return None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TransportState:
  """Cross-timestep state for the transport model."""

  core_transport_states: Mapping[str, ComponentTransportState] = (
      dataclasses.field(default_factory=dict)
  )
  pedestal_transport_states: Mapping[str, ComponentTransportState] = (
      dataclasses.field(default_factory=dict)
  )

  def to_xr_datatree(
      self,
      context: output_grid_context.OutputGridContext,
  ) -> xr.DataTree | None:
    """Builds an xr.DataTree of transport state, or None if stateless.

    Args:
      context: Output grid context for building datasets.

    Returns:
      An xr.DataTree of transport state, or None if stateless.
    """
    core_children = {
        name: tree
        for name, s in self.core_transport_states.items()
        if (tree := s.to_xr_datatree(context)) is not None
    }
    pedestal_children = {
        name: tree
        for name, s in self.pedestal_transport_states.items()
        if (tree := s.to_xr_datatree(context)) is not None
    }
    children = {}
    if core_children:
      children[output_keys.CORE] = xr.DataTree(children=core_children)
    if pedestal_children:
      children[output_keys.PEDESTAL] = xr.DataTree(children=pedestal_children)
    if not children:
      return None
    return xr.DataTree(children=children)

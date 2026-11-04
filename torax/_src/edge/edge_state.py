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

"""Cross-timestep state for edge models."""

from collections.abc import Mapping
import dataclasses
import jax
from torax._src.output_tools import output_grid_context
import xarray as xr


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class EdgeState:
  """Base class for cross-timestep state of an edge model."""

  def to_xr_datatree(
      self,
      context: output_grid_context.OutputGridContext,
  ) -> xr.DataTree | None:
    """Builds an xr.DataTree of edge state, or None if stateless.

    Args:
      context: Output grid context for building datasets.

    Returns:
      An xr.DataTree of edge state, or None if the edge model is stateless.
    """
    del context
    return None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class CombinedEdgeState(EdgeState):
  """Cross-timestep state for a CombinedEdgeModel."""

  sub_models: Mapping[str, EdgeState] = dataclasses.field(default_factory=dict)

  def to_xr_datatree(
      self,
      context: output_grid_context.OutputGridContext,
  ) -> xr.DataTree | None:
    """Builds an xr.DataTree of sub-model edge states, or None if stateless.

    Args:
      context: Output grid context for building datasets.

    Returns:
      An xr.DataTree of combined sub-model states, or None if all are stateless.
    """
    children = {
        name: tree
        for name, sub in self.sub_models.items()
        if (tree := sub.to_xr_datatree(context)) is not None
    }
    if not children:
      return None
    return xr.DataTree(children=children)

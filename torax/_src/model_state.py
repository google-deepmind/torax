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

"""ModelState dataclass for holding cross-timestep physics model states."""

from collections.abc import Mapping
import dataclasses
import jax
from torax._src.edge import edge_state
from torax._src.output_tools import output_grid_context
from torax._src.pedestal_model import pedestal_state
from torax._src.sources import source_state
from torax._src.transport_model import transport_state
import xarray as xr


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ModelState:
  """Cross-timestep state for stateful physics models."""

  pedestal: pedestal_state.PedestalState
  transport: transport_state.TransportState
  sources: Mapping[str, source_state.SourceState]
  edge: edge_state.EdgeState | None = None

  def to_xr_datatree(
      self,
      context: output_grid_context.OutputGridContext,
  ) -> xr.DataTree | None:
    """Builds an xr.DataTree of model state, or None.

    Args:
      context: Output grid context for building datasets.

    Returns:
      An xr.DataTree containing model states, or None if all models are
      stateless.
    """
    del context
    children: dict[str, xr.DataTree] = {}
    if not children:
      return None
    return xr.DataTree(children=children)

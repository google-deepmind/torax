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
"""A container for all models."""

import dataclasses

from torax._src import model_state as model_state_lib
from torax._src import static_dataclass
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.edge import base as edge_model_lib
from torax._src.internal_boundary_conditions import base_model as internal_boundary_conditions_base_model
from torax._src.mhd import base as mhd_model_lib
from torax._src.neoclassical import neoclassical_model as neoclassical_model_lib
from torax._src.output_tools import output_keys
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.sources import source_models as source_models_lib
from torax._src.time_step_calculator.time_step_calculator import TimeStepCalculator
from torax._src.transport_model import transport_model as transport_model_lib
import xarray as xr


@dataclasses.dataclass(frozen=True, eq=False)
class Models(static_dataclass.StaticDataclass):
  """A container for all models used within TORAX.

  This class is used as a static argument to Jax functions. It is therefore
  designed to be immutable and support comparison and hashing by value.
  Because this class does not use polymorphism, it does not need to hash
  the class id, so the default frozen dataset hashing works.
  """

  source_models: source_models_lib.SourceModels
  transport_model: transport_model_lib.TransportModel
  pedestal_model: pedestal_model_lib.PedestalModel
  neoclassical_model: neoclassical_model_lib.NeoclassicalModel
  mhd_models: mhd_model_lib.MHDModels
  edge_model: edge_model_lib.EdgeModel | None
  time_step_calculator: TimeStepCalculator
  internal_boundary_condition_model: (
      internal_boundary_conditions_base_model.InternalBoundaryConditionModel
  )

  def initial_state_from_datatree(
      self,
      datatree: xr.DataTree | None,
      runtime_params: runtime_params_lib.RuntimeParams,
  ) -> model_state_lib.ModelState:
    """Reconstructs ModelState across all physics models from a DataTree.

    Args:
      datatree: Time-sliced `/model_state` DataTree, or None if all models were
        stateless.
      runtime_params: Runtime parameters at the restart time.

    Returns:
      Reconstructed ModelState.
    """
    children = datatree.children if datatree is not None else {}
    if self.edge_model is not None:
      assert runtime_params.edge is not None
      edge_state = self.edge_model.initial_state_from_datatree(
          children.get(output_keys.EDGE),
          runtime_params.edge,
      )
    else:
      edge_state = None
    return model_state_lib.ModelState(
        pedestal=self.pedestal_model.initial_state_from_datatree(
            children.get(output_keys.PEDESTAL),
            runtime_params,
        ),
        transport=self.transport_model.initial_state_from_datatree(
            children.get(output_keys.TRANSPORT),
            runtime_params,
        ),
        sources=self.source_models.initial_state_from_datatree(
            children.get(output_keys.SOURCES),
            runtime_params,
        ),
        edge=edge_state,
    )

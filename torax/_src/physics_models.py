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
"""A container for all physics models."""
import dataclasses

import jax
from torax._src.mhd import base as mhd_model_lib
from torax._src.neoclassical import neoclassical_models as neoclassical_models_lib
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.sources import source_models as source_models_lib
from torax._src.transport_model import transport_model as transport_model_lib


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PhysicsModels:
  """A container for all physics models."""

  source_models: source_models_lib.SourceModels = dataclasses.field(
      metadata=dict(static=True)
  )
  transport_model: transport_model_lib.TransportModel = dataclasses.field(
      metadata=dict(static=True)
  )
  pedestal_model: pedestal_model_lib.PedestalModel = dataclasses.field(
      metadata=dict(static=True)
  )
  neoclassical_models: neoclassical_models_lib.NeoclassicalModels = (
      dataclasses.field(metadata=dict(static=True))
  )
  mhd_models: mhd_model_lib.MHDModels = dataclasses.field(
      metadata=dict(static=True)
  )

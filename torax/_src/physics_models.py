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
"""A container for all physics models."""
import dataclasses

from torax._src.edge import base as edge_model_lib
from torax._src.mhd import base as mhd_model_lib
from torax._src.neoclassical import neoclassical_models as neoclassical_models_lib
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.sources import source_models as source_models_lib
from torax._src.transport_model import transport_model as transport_model_lib


@dataclasses.dataclass(frozen=True)
class PhysicsModels:
  """A container for all physics models.

  This class is used as a static argument to Jax functions. It is therefore
  designed to be immutable and support comparison and hashing by value.
  Because this class does not use polymorphism, it does not need to hash
  the class id, so the default frozen dataset hashing works.
  """

  source_models: source_models_lib.SourceModels
  transport_model: transport_model_lib.TransportModel
  pedestal_model: pedestal_model_lib.PedestalModel
  neoclassical_models: neoclassical_models_lib.NeoclassicalModels
  mhd_models: mhd_model_lib.MHDModels
  edge_model: edge_model_lib.EdgeModelBase | None

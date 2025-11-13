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


"""Base classes for Neoclassical models."""
import dataclasses

from torax._src import static_dataclass
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.neoclassical.transport import base as transport_base


@dataclasses.dataclass(frozen=True, eq=False)
class NeoclassicalModels(static_dataclass.StaticDataclass):
  """Container for instantiated Neoclassical model objects.

  This class is intended for use as a static argument to jitted jax functions.
  It is therefore immutable and supports comparison and hashing by value.
  Because this class is not polymorphic, it does not need to hash the class
  id, so the default frozen dataclass hashing works.
  """

  conductivity: conductivity_base.ConductivityModel
  bootstrap_current: bootstrap_current_base.BootstrapCurrentModel
  transport: transport_base.NeoclassicalTransportModel

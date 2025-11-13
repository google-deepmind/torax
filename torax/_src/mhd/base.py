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

"""Base classes for MHD models."""

import dataclasses

from torax._src import static_dataclass
from torax._src.mhd.sawtooth import sawtooth_models as sawtooth_models_lib


@dataclasses.dataclass(frozen=True, eq=False)
class MHDModels(static_dataclass.StaticDataclass):
  """Container for instantiated MHD model objects.

  This class is designed to be used as a static argument to jitted Jax
  functions, so it is immutable and supports comparison and hashing by value.
  """

  sawtooth_models: sawtooth_models_lib.SawtoothModels | None

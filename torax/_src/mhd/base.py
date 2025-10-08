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

import jax
from torax._src.mhd.sawtooth import sawtooth_models as sawtooth_models_lib


@jax.tree_util.register_dataclass
@dataclasses.dataclass(eq=False)
class MHDModels:
  """Container for instantiated MHD model objects."""

  sawtooth_models: sawtooth_models_lib.SawtoothModels | None = None

  def __eq__(self, other: 'MHDModels') -> bool:
    return self.sawtooth_models == other.sawtooth_models

  def _hash(self) -> int:
    return hash((self.sawtooth_models,))

  def __hash__(self) -> int:
    return self._hash()

  def __post_init__(self):
    """Run post-init checks."""

    # Make sure our custom hash hasn't been overwritten by the dataclasses
    # decorator
    assert hash(self) == self._hash()

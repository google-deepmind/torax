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

"""Base runtime_params for edge models."""
import dataclasses
import jax
from torax._src import array_typing


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class RuntimeParams:
  """Base for edge model runtime parameters.

  Attributes:
    update_temperatures: Whether to update temperature boundary conditions.
    update_density: Whether to update electron density boundary condition.
    update_impurities: Whether to update impurity concentrations in the core.
  """

  # Not static to allow rapid sensitivity checking of edge-model impact.
  update_temperatures: array_typing.BoolScalar
  update_density: array_typing.BoolScalar
  update_impurities: array_typing.BoolScalar

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

"""Runtime parameters for geometry models."""

# pylint: disable=invalid-name

import dataclasses
import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import jax_utils
from torax._src.geometry import geometry as geometry_lib


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False, kw_only=True)
class RuntimeParams:
  """Base class for geometry model runtime parameters.

  Attributes:
    calc_Phi_b_dot: Whether to calculate Phi_b_dot across time steps.
  """

  calc_Phi_b_dot: bool = jax.tree.static(default=False)


# Alias for backward/forward compatibility.
GeometryRuntimeParams = RuntimeParams


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False, kw_only=True)
class PrescribedGeometryRuntimeParams(RuntimeParams):
  """Runtime parameters for prescribed geometry evaluated at time `t`.

  Attributes:
    geometry: Single-slice `Geometry` evaluated at time `t`.
    initial_Phi_b_dot: Initial time derivative of boundary toroidal magnetic
      flux at time `t` [Wb/s].
  """

  geometry: geometry_lib.Geometry
  initial_Phi_b_dot: array_typing.FloatScalar = dataclasses.field(
      default_factory=lambda: jnp.zeros((), dtype=jax_utils.get_dtype())
  )

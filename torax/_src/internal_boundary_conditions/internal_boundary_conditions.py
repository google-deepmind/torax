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

"""Internal boundary conditions."""

import dataclasses

import chex
import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import jax_utils
from torax._src.geometry import geometry
from torax._src.torax_pydantic import torax_pydantic
# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class InternalBoundaryConditions:
  """Container for internal boundary conditions.

  Internal boundary conditions are set on the cell grid only, as they are met
  using an adaptive source (which by nature is on the cell grid).
  Zero values in the array are ignored, i.e. they are not treated as boundary
  conditions. This allows us to use a single object to set multiple boundary
  conditions simultaneously. For example, setting
  `T_e=jnp.array([0, 20, 0, 10, 0])` will set the electron temperature to 20
  and 10 at the 2nd and 4th cells respectively.

  Attributes:
    T_i: Ion temperature boundary conditions on the cell grid.
    T_e: Electron temperature boundary condition on the cell grid.
    n_e: Electron density boundary condition on the cell grid.
  """

  T_i: array_typing.FloatVectorCell
  T_e: array_typing.FloatVectorCell
  n_e: array_typing.FloatVectorCell

  def update(
      self,
      other: 'InternalBoundaryConditions',
  ) -> 'InternalBoundaryConditions':
    """Combine with another InternalBoundaryCondition object, with values from `other` taking precedence."""
    return dataclasses.replace(
        self,
        T_i=jnp.where(other.T_i != 0.0, other.T_i, self.T_i),
        T_e=jnp.where(other.T_e != 0.0, other.T_e, self.T_e),
        n_e=jnp.where(other.n_e != 0.0, other.n_e, self.n_e),
    )

  @classmethod
  def empty(cls, geo: geometry.Geometry) -> 'InternalBoundaryConditions':
    """Return an empty InternalBoundaryConditions object."""
    nx = geo.torax_mesh.nx
    return cls(
        T_i=jnp.zeros(nx, dtype=jax_utils.get_dtype()),
        T_e=jnp.zeros(nx, dtype=jax_utils.get_dtype()),
        n_e=jnp.zeros(nx, dtype=jax_utils.get_dtype()),
    )


class InternalBoundaryConditionsConfig(torax_pydantic.BaseModelFrozen):
  """Pydantic model for internal boundary conditions."""

  # Set to zero by default, which is ignored by the adaptive source.
  T_i: torax_pydantic.TimeVaryingPoints = torax_pydantic.ValidatedDefault(0.0)
  T_e: torax_pydantic.TimeVaryingPoints = torax_pydantic.ValidatedDefault(0.0)
  n_e: torax_pydantic.TimeVaryingPoints = torax_pydantic.ValidatedDefault(0.0)

  def build_runtime_params(self, t: chex.Numeric) -> InternalBoundaryConditions:
    """Builds the runtime params for the internal boundary conditions."""
    kwargs = {
        field.name: getattr(self, field.name).get_value(t)
        for field in dataclasses.fields(InternalBoundaryConditions)
    }
    return InternalBoundaryConditions(**kwargs)

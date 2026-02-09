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

import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import jax_utils
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry

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


def apply_adaptive_source(
    *,
    source_T_i: array_typing.FloatVectorCell,
    source_T_e: array_typing.FloatVectorCell,
    source_n_e: array_typing.FloatVectorCell,
    source_mat_ii: array_typing.FloatVectorCell,
    source_mat_ee: array_typing.FloatVectorCell,
    source_mat_nn: array_typing.FloatVectorCell,
    runtime_params: runtime_params_lib.RuntimeParams,
    internal_boundary_conditions: InternalBoundaryConditions,
) -> tuple[
    array_typing.FloatVectorCell,
    array_typing.FloatVectorCell,
    array_typing.FloatVectorCell,
    array_typing.FloatVectorCell,
    array_typing.FloatVectorCell,
    array_typing.FloatVectorCell,
]:
  """Applies an adaptive source to the source profiles to set internal boundary conditions."""

  # Ion temperature
  source_T_i += (
      runtime_params.numerics.adaptive_T_source_prefactor
      * internal_boundary_conditions.T_i
  )
  source_mat_ii -= jnp.where(
      internal_boundary_conditions.T_i != 0.0,
      runtime_params.numerics.adaptive_T_source_prefactor,
      0.0,
  )

  # Electron temperature
  source_T_e += (
      runtime_params.numerics.adaptive_T_source_prefactor
      * internal_boundary_conditions.T_e
  )
  source_mat_ee -= jnp.where(
      internal_boundary_conditions.T_e != 0.0,
      runtime_params.numerics.adaptive_T_source_prefactor,
      0.0,
  )

  # Density
  source_n_e += (
      runtime_params.numerics.adaptive_n_source_prefactor
      * internal_boundary_conditions.n_e
  )
  source_mat_nn -= jnp.where(
      internal_boundary_conditions.n_e != 0.0,
      runtime_params.numerics.adaptive_n_source_prefactor,
      0.0,
  )

  return (
      source_T_i,
      source_T_e,
      source_n_e,
      source_mat_ii,
      source_mat_ee,
      source_mat_nn,
  )

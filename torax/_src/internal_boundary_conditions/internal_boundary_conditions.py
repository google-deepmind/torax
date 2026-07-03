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
from torax._src.core_profiles import convertors
from torax._src.geometry import geometry
from torax._src.torax_pydantic import interpolated_param_2d
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class InternalBoundaryConditions:
  """Container for internal boundary conditions.

  Internal boundary conditions are set on the cell grid. They are enforced
  via direct matrix row replacement in the solver.
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

  def merge(
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

  def to_solver_coeffs(
      self,
      evolving_names: tuple[str, ...],
      nx: int,
  ) -> tuple[jax.Array, jax.Array]:
    """Build stacked internal boundary condition mask and target arrays.

    For each evolving variable, produces a boolean mask (True where the
    internal boundary condition is active, i.e. the target is nonzero)
    and the corresponding target value scaled from their physical units in
    CoreProfiles, to solver units (e.g. with scaling factors to maintain
    similar order of magnitude between evolved variables in the state vector).

    Args:
      evolving_names: Ordered tuple of evolving variable names (e.g. ('T_i',
        'T_e', 'psi', 'n_e')).
      nx: Number of cells in the radial grid.

    Returns:
      A (mask, target) tuple where:
        mask: Bool array of shape (nx, num_channels).
        target: Float array of shape (nx, num_channels) in solver-scaled units.
    """
    var_map = {
        'T_i': self.T_i,
        'T_e': self.T_e,
        'n_e': self.n_e,
    }
    mask_parts = []
    target_parts = []
    for var in evolving_names:
      if var in var_map:
        target_original_units = var_map[var]
        # Only apply where the target is nonzero.
        mask_parts.append(target_original_units != 0.0)
        target_parts.append(
            target_original_units / convertors.SCALING_FACTORS[var]
        )
      else:
        # Variables like psi have no internal boundary conditions.
        mask_parts.append(jnp.zeros(nx, dtype=jnp.bool_))
        target_parts.append(jnp.zeros(nx, dtype=jax_utils.get_dtype()))
    # Stack along axis=-1 to produce shape (nx, num_channels), i.e.
    # grid-major ordering. This matches the block tridiagonal solver layout
    # where the leading axis indexes spatial cells and the trailing axis
    # indexes the evolving channels within each cell's block.
    return jnp.stack(mask_parts, axis=-1), jnp.stack(target_parts, axis=-1)


class InternalBoundaryConditionsConfig(torax_pydantic.BaseModelFrozen):
  """Pydantic model for internal boundary conditions."""

  T_i: interpolated_param_2d.SparseTimeVaryingArray = (
      torax_pydantic.ValidatedDefault(0.0)
  )
  T_e: interpolated_param_2d.SparseTimeVaryingArray = (
      torax_pydantic.ValidatedDefault(0.0)
  )
  n_e: interpolated_param_2d.SparseTimeVaryingArray = (
      torax_pydantic.ValidatedDefault(0.0)
  )

  def build_runtime_params(self, t: chex.Numeric) -> InternalBoundaryConditions:
    """Builds the runtime params for the internal boundary conditions."""
    return InternalBoundaryConditions(
        T_i=self.T_i.get_value(t),
        T_e=self.T_e.get_value(t),
        n_e=self.n_e.get_value(t),
    )


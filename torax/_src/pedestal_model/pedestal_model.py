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

"""The PedestalModel abstract base class.

The pedestal model calculates quantities relevant to the pedestal.
"""
import abc
import dataclasses

import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import state
from torax._src import static_dataclass
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.internal_boundary_conditions import internal_boundary_conditions as internal_boundary_conditions_lib

# pylint: disable=invalid-name
# Using physics notation naming convention


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PedestalModelOutput:
  """Output of the PedestalModel."""

  # The location of the pedestal.
  rho_norm_ped_top: array_typing.FloatScalar
  # The index of the pedestal in rho_norm.
  rho_norm_ped_top_idx: array_typing.IntScalar
  # The ion temperature at the pedestal.
  T_i_ped: array_typing.FloatScalar
  # The electron temperature at the pedestal.
  T_e_ped: array_typing.FloatScalar
  # The electron density at the pedestal in units 10^-3.
  n_e_ped: array_typing.FloatScalar

  def to_internal_boundary_conditions(
      self,
      geo: geometry.Geometry,
  ) -> internal_boundary_conditions_lib.InternalBoundaryConditions:
    """Convert the pedestal model output to internal boundary conditions."""
    pedestal_mask = (
        jnp.zeros_like(geo.rho, dtype=bool)
        .at[self.rho_norm_ped_top_idx]
        .set(True)
    )
    return internal_boundary_conditions_lib.InternalBoundaryConditions(
        T_i=jnp.where(pedestal_mask, self.T_i_ped, 0.0),
        T_e=jnp.where(pedestal_mask, self.T_e_ped, 0.0),
        n_e=jnp.where(pedestal_mask, self.n_e_ped, 0.0),
    )


@dataclasses.dataclass(frozen=True, eq=False)
class PedestalModel(static_dataclass.StaticDataclass, abc.ABC):
  """Calculates temperature and density of the pedestal."""

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> PedestalModelOutput:
    return jax.lax.cond(
        runtime_params.pedestal.set_pedestal,
        lambda: self._call_implementation(runtime_params, geo, core_profiles),
        # Set the pedestal location to infinite to indicate that the pedestal is
        # not present.
        # Set the index to outside of bounds of the mesh to indicate that the
        # pedestal is not present.
        lambda: PedestalModelOutput(
            rho_norm_ped_top=jnp.inf,
            T_i_ped=0.0,
            T_e_ped=0.0,
            n_e_ped=0.0,
            rho_norm_ped_top_idx=geo.torax_mesh.nx,
        ),
    )

  @abc.abstractmethod
  def _call_implementation(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> PedestalModelOutput:
    """Calculate the pedestal values."""

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

"""Output of the pedestal model."""

import dataclasses
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src.geometry import geometry
from torax._src.internal_boundary_conditions import internal_boundary_conditions as internal_boundary_conditions_lib

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class TransportMultipliers:
  """Transport multipliers for the pedestal."""

  chi_e_multiplier: array_typing.FloatScalar = 1.0
  chi_i_multiplier: array_typing.FloatScalar = 1.0
  D_e_multiplier: array_typing.FloatScalar = 1.0
  v_e_multiplier: array_typing.FloatScalar = 1.0


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PedestalModelOutput:
  """Output of a PedestalModel.

  Attributes:
    rho_norm_ped_top: The requested location of the pedestal top in rho_norm,
      not quantized to either the cell or face grid.
    rho_norm_ped_top_idx: The nearest cell index of the pedestal top.
    T_i_ped: The ion temperature at the pedestal top in keV.
    T_e_ped: The electron temperature at the pedestal top in keV.
    n_e_ped: The electron density at the pedestal top in m^-3.
    transport_multipliers: Multipliers for the transport coefficients in the
      pedestal region. Only used if the pedestal is in ADAPTIVE_TRANSPORT mode.
  """

  rho_norm_ped_top: array_typing.FloatScalar
  # TODO(b/434175938): Can we remove rho_norm_ped_top_idx?
  rho_norm_ped_top_idx: array_typing.IntScalar
  T_i_ped: array_typing.FloatScalar
  T_e_ped: array_typing.FloatScalar
  n_e_ped: array_typing.FloatScalar
  transport_multipliers: TransportMultipliers = TransportMultipliers()

  def to_internal_boundary_conditions(
      self,
      geo: geometry.Geometry,
  ) -> internal_boundary_conditions_lib.InternalBoundaryConditions:
    """Convert the pedestal model output to internal boundary conditions."""
    # In this case, the mask is only the pedestal top, not the whole pedestal
    # region. This is because we are adding a source/sink term only at the
    # pedestal top.
    # We are using the cell grid here, since internal boundary conditions are
    # applied using an adaptive source (which acts on the cell grid).
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

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

"""GeometryState dataclass for holding cross-timestep geometry model states."""

import dataclasses
from typing import Self

import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import jax_utils
from torax._src.geometry import geometry
from torax._src.output_tools import output_grid_context
from torax._src.output_tools import output_keys
import xarray as xr

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class GeometryState:
  """Cross-timestep state for geometry models.

  Attributes:
    Phi_b: Toroidal magnetic flux at the plasma boundary (LCFS) [Wb].
    Phi_b_dot: Time derivative of the toroidal magnetic flux at the plasma
      boundary (LCFS) [Wb/s].
  """

  Phi_b: array_typing.FloatScalar
  Phi_b_dot: array_typing.FloatScalar = dataclasses.field(
      default_factory=lambda: jnp.zeros((), dtype=jax_utils.get_dtype())
  )

  @classmethod
  def from_geometry(
      cls,
      geo: geometry.Geometry,
      *,
      Phi_b_dot: array_typing.FloatScalar | None = None,
  ) -> Self:
    """Initializes `GeometryState` from an equilibrium `Geometry`."""
    if Phi_b_dot is None:
      Phi_b_dot = jnp.zeros((), dtype=jax_utils.get_dtype())
    return cls(
        Phi_b=jnp.asarray(geo.Phi_b, dtype=jax_utils.get_dtype()),
        Phi_b_dot=jnp.asarray(Phi_b_dot, dtype=jax_utils.get_dtype()),
    )

  def to_output_dict(
      self, context: output_grid_context.OutputGridContext
  ) -> dict[str, output_grid_context.OutputVar]:
    """Packs geometry state outputs into an output dictionary."""
    return {
        output_keys.PHI_B_DOT: context.pack(
            output_keys.PHI_B_DOT, self.Phi_b_dot
        ),
    }

  def to_xr_datatree(
      self,
      context: output_grid_context.OutputGridContext,
  ) -> xr.DataTree | None:
    """Serializes geometry state to an xarray DataTree node, or None."""
    del context
    return None


def update_geometry_state(
    *,
    geometry_state: GeometryState,
    geo: geometry.Geometry,
    dt: array_typing.FloatScalar,
    calc_Phi_b_dot: bool,
) -> GeometryState:
  """Computes the updated `GeometryState` across `[t, t + dt]`.

  Args:
    geometry_state: Previous geometry state from timestep `t`.
    geo: Equilibrium geometry at current timestep `t + dt`.
    dt: Time step duration [s].
    calc_Phi_b_dot: Whether to compute `Phi_b_dot` across the step interval.

  Returns:
    Updated `GeometryState` for `[t, t + dt]`.
  """
  phi_b = geo.Phi_b
  if not calc_Phi_b_dot:
    phi_b_dot = jnp.zeros((), dtype=jax_utils.get_dtype())
  else:
    phi_b_dot = (phi_b - geometry_state.Phi_b) / dt
  return GeometryState(Phi_b=phi_b, Phi_b_dot=phi_b_dot)

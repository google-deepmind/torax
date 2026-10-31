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

"""Flux profiles dataclass and constructor for magnetic and current profiles."""

import dataclasses
from typing import Self
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import jax_utils
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.physics import psi_calculations

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class FluxProfiles:
  """Magnetic flux and current profiles.

  Attributes:
    psi: Poloidal flux [Wb].
    q_face: Safety factor on face grid.
    s_face: Magnetic shear on face grid.
    j_toroidal_total: Total current density on cell grid [A/m^2].
    j_toroidal_total_face: Total current density on face grid [A/m^2].
    I_p_profile_face: Plasma current profile on face grid [A].
    v_loop_lcfs: Loop voltage at LCFS [V].
  """

  psi: cell_variable.CellVariable
  q_face: array_typing.FloatVectorFace
  s_face: array_typing.FloatVectorFace
  j_toroidal_total: array_typing.FloatVectorCell
  j_toroidal_total_face: array_typing.FloatVectorFace
  I_p_profile_face: array_typing.FloatVectorFace
  v_loop_lcfs: array_typing.FloatScalar

  @classmethod
  def from_psi(
      cls,
      geo: geometry.Geometry,
      psi: cell_variable.CellVariable,
      min_rho_norm: array_typing.FloatScalar,
      v_loop_lcfs: array_typing.FloatScalar | None = None,
  ) -> Self:
    """Constructs FluxProfiles derived from poloidal flux and geometry."""
    j_toroidal_total, j_toroidal_total_face, I_p_profile_face = (
        psi_calculations.calc_j_toroidal_total(
            geo,
            psi,
            min_rho_norm,
        )
    )
    if v_loop_lcfs is None:
      v_loop_lcfs = jnp.asarray(0.0, dtype=jax_utils.get_dtype())
    return cls(
        psi=psi,
        q_face=psi_calculations.calc_q_face(geo, psi),
        s_face=psi_calculations.calc_s_face(geo, psi),
        j_toroidal_total=j_toroidal_total,
        j_toroidal_total_face=j_toroidal_total_face,
        I_p_profile_face=I_p_profile_face,
        v_loop_lcfs=v_loop_lcfs,
    )

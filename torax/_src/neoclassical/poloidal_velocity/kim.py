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
"""Kim model for neoclassical poloidal velocity."""
from typing import Annotated, Literal
import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical.formulas import formulas
from torax._src.neoclassical.poloidal_velocity import base
from torax._src.physics import psi_calculations
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name


@jax.jit
def _calculate_poloidal_velocity(
    T_i: cell_variable.CellVariable,
    k_neo: array_typing.FloatVectorFace,
    Z_i: array_typing.FloatVectorFace,
    B_tor: array_typing.FloatVectorFace,
    B_total_squared: array_typing.FloatVectorFace,
    geo: geometry_lib.Geometry,
    poloidal_velocity_multiplier: array_typing.FloatScalar = 1.0,
) -> cell_variable.CellVariable:
  """Computes the neoclassical ion poloidal velocity profile.

  Implementing eq.33 from
  Y. B. Kim , P. H. Diamond , R. J. Groebner.
  "Neoclassical poloidal and toroidal rotation in tokamaks"
  Phys. Fluids B 3, 2050–2060 (1991)
  https://doi.org/10.1063/1.859671

  Eq. 33 can be simplified to the following form in SI units:
  v_pol = k_neo * (dT/dr) * (B_tor / <B^2>) / (Z * e)

  Args:
    T_i: Ion temperature as a cell variable [keV].
    k_neo: Neoclassical coefficient on the face grid.
    Z_i: Main ion charge on the face grid.
    B_tor: Toroidal magnetic field on the face grid [T].
    B_total_squared: Total magnetic field (toroidal + poloidal) on the face grid
      [T].
    geo: Geometry.
    poloidal_velocity_multiplier: A multiplier to apply to the poloidal
      velocity.

  Returns:
    v_pol : Poloidal velocity profile [m/s].
  """
  # Note: all computations are performed on the face grid.
  # Calculate Radial Temperature Gradient (dT/dr)
  grad_Ti = (
      T_i.face_grad(
          x=geo.r_mid, x_left=geo.r_mid_face[0], x_right=geo.r_mid_face[-1]
      )
      * constants.CONSTANTS.keV_to_J
  )  # [J/m]

  # Calculate Poloidal Velocity
  # v_pol = k_i * (dT/dr) * (B_tor / <B^2>) / (Z * e)
  B_total_squared_safe = jnp.maximum(B_total_squared, constants.CONSTANTS.eps)
  v_pol = (
      k_neo
      * grad_Ti
      * (B_tor / B_total_squared_safe)
      / (constants.CONSTANTS.q_e * Z_i)
  )

  v_pol = poloidal_velocity_multiplier * v_pol

  return cell_variable.CellVariable(
      value=geometry_lib.face_to_cell(v_pol),
      face_centers=geo.rho_face_norm,
      right_face_constraint=v_pol[-1],
      right_face_grad_constraint=None,
  )


class KimModel(base.PoloidalVelocityModel):
  """Kim (1991) model for neoclassical poloidal velocity."""

  def calculate_poloidal_velocity(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
      neoclassical_intermediates: formulas.NeoclassicalIntermediates,
  ) -> base.PoloidalVelocity:
    """Calculates poloidal velocity according to the Kim (1991) model."""
    B_tor_face = geometry.F_face / geometry.R_major_profile_face
    B_pol_squared_face = psi_calculations.calc_bpol_squared(
        geometry, core_profiles.psi
    )
    B_total_squared_face = B_pol_squared_face + B_tor_face**2

    poloidal_velocity_params = runtime_params.neoclassical.poloidal_velocity
    v_pol = _calculate_poloidal_velocity(
        T_i=core_profiles.T_i,
        k_neo=neoclassical_intermediates.k_neo,
        Z_i=core_profiles.Z_i_face,
        B_tor=B_tor_face,
        B_total_squared=B_total_squared_face,
        geo=geometry,
        poloidal_velocity_multiplier=poloidal_velocity_params.poloidal_velocity_multiplier,
    )
    return base.PoloidalVelocity(v_pol=v_pol)

  def __eq__(self, other) -> bool:
    return isinstance(other, self.__class__)

  def __hash__(self) -> int:
    return hash(self.__class__)


class KimModelConfig(base.PoloidalVelocityModelConfig):
  """Config for the Kim (1991) model implementation of poloidal velocity."""

  model_name: Annotated[Literal['kim'], torax_pydantic.JAX_STATIC] = 'kim'

  def build_model(self) -> KimModel:
    return KimModel()

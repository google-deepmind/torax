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

"""Transport coefficient data structures."""

import dataclasses

import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src.geometry import geometry
from torax._src.output_tools import output_grid_context
from torax._src.output_tools import output_keys
import typing_extensions


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TransportCoeffs:
  """Base 4-channel transport coefficients on the face grid.

  Attributes:
    chi_face_ion: Ion heat conductivity, on the face grid [m^2/s].
    chi_face_el: Electron heat conductivity, on the face grid [m^2/s].
    d_face_el: Diffusivity of electron density, on the face grid [m^2/s].
    v_face_el: Convection strength of electron density, on the face grid [m/s].
  """

  chi_face_ion: array_typing.FloatVectorFace
  chi_face_el: array_typing.FloatVectorFace
  d_face_el: array_typing.FloatVectorFace
  v_face_el: array_typing.FloatVectorFace

  @classmethod
  def zeros(cls, geo: geometry.Geometry) -> typing_extensions.Self:
    """Returns a TransportCoeffs with all zeros."""
    zeros = jnp.zeros_like(geo.rho_face_norm)
    return cls(
        chi_face_ion=zeros,
        chi_face_el=zeros,
        d_face_el=zeros,
        v_face_el=zeros,
    )

  def __add__(self, other: typing_extensions.Self) -> typing_extensions.Self:
    """Adds two TransportCoeffs channel-by-channel."""
    return self.__class__(
        chi_face_ion=self.chi_face_ion + other.chi_face_ion,
        chi_face_el=self.chi_face_el + other.chi_face_el,
        d_face_el=self.d_face_el + other.d_face_el,
        v_face_el=self.v_face_el + other.v_face_el,
    )

  def chi_max(self, geo: geometry.Geometry) -> jax.Array:
    """Calculates the maximum value of chi across ion and electron channels.

    Args:
      geo: Geometry of the torus.

    Returns:
      chi_max: Maximum value of chi.
    """
    return jnp.maximum(
        jnp.max(self.chi_face_ion * geo.g1_over_vpr2_face),
        jnp.max(self.chi_face_el * geo.g1_over_vpr2_face),
    )

  def to_output_dict(
      self,
      context: output_grid_context.OutputGridContext,
  ) -> dict[str, output_grid_context.OutputVar]:
    """Converts the 4 standard channels to an OutputVar mapping."""
    return {
        output_keys.CHI_TURB_I: context.pack(
            output_keys.CHI_TURB_I, self.chi_face_ion
        ),
        output_keys.CHI_TURB_E: context.pack(
            output_keys.CHI_TURB_E, self.chi_face_el
        ),
        output_keys.D_TURB_E: context.pack(
            output_keys.D_TURB_E, self.d_face_el
        ),
        output_keys.V_TURB_E: context.pack(
            output_keys.V_TURB_E, self.v_face_el
        ),
    }

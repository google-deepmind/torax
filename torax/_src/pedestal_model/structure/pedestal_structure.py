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

"""Pedestal structure data container."""

import dataclasses
from typing import Self

import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import jax_utils

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PedestalStructure:
  """Output of a PedestalStructureModel.

  Here "pedestal top" follows the naming convention in Snyder et al., Nucl.
  Fusion 44 (2004) 320, which was designed for modified-tanh (mtanh) profiles.
  In this convention, a pedestal width of Δ corresponds to a pedestal 'top'
  location of ψ_top = 1 - 1.5Δ and a pedestal 'knee' location of ψ_ped = 1 - Δ.

  Attributes:
    rho_norm_top: The requested location of the pedestal top (corresponding to
      ψ_top = 1 - 1.5Δ) in rho_norm, not quantized to either the cell or face
      grid.
    T_i_top: The ion temperature at the pedestal top [keV].
    T_e_top: The electron temperature at the pedestal top [keV].
    n_e_top: The electron density at the pedestal top [m^-3].
  """

  rho_norm_top: array_typing.FloatScalar
  T_i_top: array_typing.FloatScalar
  T_e_top: array_typing.FloatScalar
  n_e_top: array_typing.FloatScalar

  @classmethod
  def no_pedestal(cls) -> Self:
    """Returns a PedestalStructure representing an inactive or absent pedestal.

    Uses `rho_norm_top=jnp.inf` and `0.0` pedestal values so that radial
    masks are False while maintaining a concrete PyTree structure compatible
    with `jax.lax.cond`.
    """
    return cls(
        rho_norm_top=jnp.array(jnp.inf, dtype=jax_utils.get_dtype()),
        T_i_top=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        T_e_top=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        n_e_top=jnp.array(0.0, dtype=jax_utils.get_dtype()),
    )

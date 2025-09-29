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

"""Impurity content defined by ratios of impurity to electron density."""
import dataclasses
from typing import Annotated, Literal, Mapping
import chex
import jax
from jax import numpy as jnp
import jaxtyping as jt
import pydantic
from torax._src import array_typing
from torax._src import constants
from torax._src.torax_pydantic import torax_pydantic
import typing_extensions

# pylint: disable=invalid-name


def calculate_fractions_from_ratios(
    ratios: array_typing.FloatVector,
) -> array_typing.FloatVector:
  """Calculates fractions from ratios, handling the all-zero case."""
  # Ratios can be 1D (n_species,) or 2D (n_species, n_grid).
  # Sum over the species axis.
  total_ratio = jnp.sum(ratios, axis=0)

  is_positive = total_ratio > 0.0

  # Avoid division by zero by replacing zeros in total_ratio with 1.0.
  # The result of this division will be masked out by jnp.where anyway.
  safe_total_ratio = jnp.where(is_positive, total_ratio, 1.0)

  # Calculate fractions where total_ratio is positive.
  calculated_fractions = ratios / safe_total_ratio

  # For the zero impurity case, return uniform fractions to avoid NaNs.
  # The choice is arbitrary as it will be multiplied by zero impurity density.
  num_species = ratios.shape[0]
  # num_species guaranteed to be > 0 since no empty impurity dict is allowed.
  uniform_fractions = jnp.ones_like(ratios) / num_species

  return jnp.where(is_positive, calculated_fractions, uniform_fractions)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams:
  """Analogous to ion_mixture.RuntimeParams but for n_e_ratio inputs."""

  n_e_ratios: jt.Float[array_typing.Array, 'ion_symbol rhon']
  n_e_ratios_face: jt.Float[array_typing.Array, 'ion_symbol rhon+1']
  A_avg: array_typing.FloatVectorCell
  A_avg_face: array_typing.FloatVectorFace
  Z_override: array_typing.FloatScalar | None = None

  @property
  def fractions(self) -> array_typing.FloatVector:
    """Returns the impurity fractions calculated from the n_e_ratios."""
    return calculate_fractions_from_ratios(self.n_e_ratios)

  @property
  def fractions_face(self) -> array_typing.FloatVectorFace:
    """Returns the impurity fractions calculated from the n_e_ratios."""
    return calculate_fractions_from_ratios(self.n_e_ratios_face)


class ElectronDensityRatios(torax_pydantic.BaseModelFrozen):
  """Impurity content defined by ratios of impurity to electron density.

  Attributes:
    species: A dictionary of TimeVaryingArray objects, where the keys are ion
      symbols and the values are TimeVaryingArray objects representing the
      ratios of impurity to electron density at the defined rho_norm points.
    Z_override: A TimeVaryingScalar object representing the charge override.
    A_override: A TimeVaryingScalar object representing the A override.
  """

  species: Mapping[str, torax_pydantic.NonNegativeTimeVaryingArray]
  Z_override: torax_pydantic.TimeVaryingScalar | None = None
  A_override: torax_pydantic.TimeVaryingScalar | None = None
  impurity_mode: Annotated[Literal['n_e_ratios'], torax_pydantic.JAX_STATIC] = (
      'n_e_ratios'
  )

  @pydantic.model_validator(mode='after')
  def _validate_species_not_empty(self) -> typing_extensions.Self:
    if not self.species:
      raise ValueError('The species dictionary cannot be empty.')
    return self

  def build_runtime_params(self, t: chex.Numeric) -> RuntimeParams:
    """Creates a RuntimeParams object at a given time."""
    ions = self.species.keys()
    n_e_ratios_arr = jnp.array(
        [ratio.get_value(t) for ratio in self.species.values()]
    )
    n_e_ratios_face_arr = jnp.array([
        ratio.get_value(t, grid_type='face') for ratio in self.species.values()
    ])
    Z_override = None if not self.Z_override else self.Z_override.get_value(t)
    fractions = calculate_fractions_from_ratios(n_e_ratios_arr)
    fractions_face = calculate_fractions_from_ratios(n_e_ratios_face_arr)

    if not self.A_override:
      As = jnp.array([constants.ION_PROPERTIES_DICT[ion].A for ion in ions])
      A_avg = jnp.sum(As[..., jnp.newaxis] * fractions, axis=0)
      A_avg_face = jnp.sum(As[..., jnp.newaxis] * fractions_face, axis=0)
    else:
      A_override = self.A_override.get_value(t)
      A_avg = jnp.ones_like(n_e_ratios_arr[0]) * A_override
      A_avg_face = jnp.ones_like(n_e_ratios_face_arr[0]) * A_override

    return RuntimeParams(
        n_e_ratios=n_e_ratios_arr,
        n_e_ratios_face=n_e_ratios_face_arr,
        A_avg=A_avg,
        Z_override=Z_override,
        A_avg_face=A_avg_face,
    )

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
from jax import lax
from jax import numpy as jnp
import pydantic
from torax._src import array_typing
from torax._src import constants
from torax._src.torax_pydantic import torax_pydantic
import typing_extensions


# pylint: disable=invalid-name


def calculate_fractions_from_ratios(
    ratios: Mapping[str, chex.Array],
) -> Mapping[str, chex.Array]:
  """Calculates fractions from ratios, handling the all-zero case."""
  # Ratios can be 1D (n_species,) or 2D (n_species, n_grid).
  # Sum over the species axis.
  total_ratio = jax.tree.reduce(lax.add, ratios)

  is_positive = total_ratio > 0.0

  # Avoid division by zero by replacing zeros in total_ratio with 1.0.
  # The result of this division will be masked out by jnp.where anyway.
  safe_total_ratio = jnp.where(is_positive, total_ratio, 1.0)

  # For the zero impurity case, return uniform fractions to avoid NaNs.
  # The choice is arbitrary as it will be multiplied by zero impurity density.
  num_species = len(ratios)

  def f(leaf):
    # Calculate fractions where total_ratio is positive.
    calculated_fractions = leaf / safe_total_ratio
    uniform_fractions = jnp.ones_like(leaf) / num_species
    return jnp.where(is_positive, calculated_fractions, uniform_fractions)

  return jax.tree.map(f, ratios)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams:
  """Analogous to ion_mixture.RuntimeParams but for n_e_ratio inputs."""

  n_e_ratios: Mapping[str, array_typing.FloatVectorCell]
  n_e_ratios_face: Mapping[str, array_typing.FloatVectorFace]
  A_avg: array_typing.FloatVectorCell
  A_avg_face: array_typing.FloatVectorFace
  Z_override: array_typing.FloatScalar | None = None

  @property
  def fractions(self) -> Mapping[str, array_typing.FloatVector]:
    """Returns the impurity fractions calculated from the n_e_ratios."""
    return calculate_fractions_from_ratios(self.n_e_ratios)

  @property
  def fractions_face(self) -> Mapping[str, array_typing.FloatVectorFace]:
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
    n_e_ratios = {
        ion_symbol: ratio.get_value(t)
        for ion_symbol, ratio in self.species.items()
    }
    n_e_ratios_face = {
        ion_symbol: ratio.get_value(t, grid_type='face')
        for ion_symbol, ratio in self.species.items()
    }
    Z_override = None if not self.Z_override else self.Z_override.get_value(t)
    fractions = calculate_fractions_from_ratios(n_e_ratios)
    fractions_face = calculate_fractions_from_ratios(n_e_ratios_face)

    if not self.A_override:
      A_avg = jnp.sum(
          jnp.array([
              constants.ION_PROPERTIES_DICT[ion].A * fraction
              for ion, fraction in fractions.items()
          ]),
          axis=0,
      )
      A_avg_face = jnp.sum(
          jnp.array([
              constants.ION_PROPERTIES_DICT[ion].A * fraction
              for ion, fraction in fractions_face.items()
          ]),
          axis=0,
      )
    else:
      A_override = self.A_override.get_value(t)
      A_avg = jnp.ones_like(list(fractions.values())[0]) * A_override
      A_avg_face = jnp.ones_like(list(fractions_face.values())[0]) * A_override

    return RuntimeParams(
        n_e_ratios=n_e_ratios,
        n_e_ratios_face=n_e_ratios_face,
        A_avg=A_avg,
        Z_override=Z_override,
        A_avg_face=A_avg_face,
    )

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
  """Analogous to IonMixture but for n_e_ratio inputs."""

  n_e_ratios: array_typing.FloatVector
  A_avg: array_typing.FloatScalar

  @property
  def fractions(self) -> array_typing.FloatVector:
    """Returns the impurity fractions calculated from the n_e_ratios."""
    return calculate_fractions_from_ratios(self.n_e_ratios)


class ELectronDensityRatios(torax_pydantic.BaseModelFrozen):
  """Impurity content defined by ratios of impurity to electron density."""

  species: Mapping[str, torax_pydantic.NonNegativeTimeVaryingScalar]
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
    fractions = calculate_fractions_from_ratios(n_e_ratios_arr)

    if not self.A_override:
      As = jnp.array([constants.ION_PROPERTIES_DICT[ion].A for ion in ions])
      A_avg = jnp.sum(As * fractions)
    else:
      A_avg = self.A_override.get_value(t)

    return RuntimeParams(
        n_e_ratios=n_e_ratios_arr,
        A_avg=A_avg,
    )

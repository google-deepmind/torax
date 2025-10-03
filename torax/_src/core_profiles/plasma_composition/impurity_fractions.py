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

"""Ion mixture model and impurity fractions model for plasma composition."""
from collections.abc import Mapping
import dataclasses
from typing import Annotated, Any, Literal, TypeAlias

import chex
import jax
from jax import numpy as jnp
import numpy as np
import pydantic
from torax._src import array_typing
from torax._src import constants
from torax._src.torax_pydantic import torax_pydantic
from typing_extensions import Final


# pylint: disable=invalid-name
_IMPURITY_MODE_FRACTIONS: Final[str] = 'fractions'


def _impurity_before_validator(value: Any) -> Any:
  """Validates the input for the ImpurityMapping."""
  if isinstance(value, str):
    return {value: 1.0}
  return value


def _impurity_after_validator(
    value: Mapping[str, torax_pydantic.TimeVaryingArray],
) -> Mapping[str, torax_pydantic.TimeVaryingArray]:
  """Validates a dictionary of TimeVaryingArray objects that form a composition.

  This validator checks for three conditions:
  1. All TimeVaryingArray objects have the same time points.
  2. For each time point, all TimeVaryingArray objects have the same `rho_norm`
     array.
  3. For each time point, the `values` arrays across all TimeVaryingArray
     objects have the same shape and sum to 1.0 along the axis of the dictionary
     keys. This is useful for ensuring that fractions of a quantity sum to 1.

  Args:
    value: The dictionary of TimeVaryingArray objects to validate.

  Returns:
    The validated dictionary.

  Raises:
    ValueError: If any of the validation checks fail.
  """
  if not value:
    raise ValueError('The species dictionary cannot be empty.')

  first_key = next(iter(value))
  first_tva = value[first_key]
  reference_times = first_tva.value.keys()

  for species_key, tva in value.items():
    if tva.value.keys() != reference_times:
      raise ValueError(
          f'Inconsistent times for key "{species_key}". Expected'
          f' {sorted(list(reference_times))}, got'
          f' {sorted(list(tva.value.keys()))}'
      )

  for t in reference_times:
    reference_rho_norm, _ = first_tva.value[t]
    for species_key, tva in value.items():
      current_rho_norm, _ = tva.value[t]
      if not np.array_equal(current_rho_norm, reference_rho_norm):
        raise ValueError(
            f'Inconsistent rho_norm for key "{species_key}" at time {t}.'
        )

    # Check for each time point, `values` have the same shape and sum to 1.0.
    values_at_t = [tva.value[t][1] for tva in value.values()]
    reference_shape = values_at_t[0].shape
    for i, v_arr in enumerate(values_at_t):
      if v_arr.shape != reference_shape:
        species_key = list(value.keys())[i]
        raise ValueError(
            f'Inconsistent value shape for key "{species_key}" at time {t}.'
            f' Expected {reference_shape}, got {v_arr.shape}.'
        )
    sum_of_values = np.sum(np.stack(values_at_t, axis=0), axis=0)
    if not np.allclose(sum_of_values, 1.0):
      raise ValueError(
          f'Values do not sum to 1 at time {t}. Sum is {sum_of_values}.'
      )
  return value


ImpurityMapping: TypeAlias = Annotated[
    Mapping[str, torax_pydantic.NonNegativeTimeVaryingArray],
    pydantic.BeforeValidator(_impurity_before_validator),
    pydantic.AfterValidator(_impurity_after_validator),
]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams:
  """Represents a fixed mixture of ion species at a specific time.

  Information on ion names are not stored here, but rather as static attributes,
  to simplify JAX logic and performance in source functions for fusion power and
  radiation which are species-dependent.

  Attributes:
    fractions: Impurity fractions for a time slice.
    fractions_face: Impurity fractions for a time slice for the face grid.
    A_avg: Average atomic mass of the mixture.
    A_avg_face: Average atomic mass of the mixture for the face grid.
    Z_override: Typically, the average Z is calculated according to the
      temperature dependent charge-state-distribution, or for low-Z cases by the
      atomic numbers of the ions assuming full ionization. If Z_override is
      provided, it is used instead for the average Z.
  """

  fractions: Mapping[str, array_typing.FloatVectorCell]
  fractions_face: Mapping[str, array_typing.FloatVectorFace]
  A_avg: array_typing.FloatVectorCell
  A_avg_face: array_typing.FloatVectorFace
  Z_override: array_typing.FloatScalar | None = None


class ImpurityFractions(torax_pydantic.BaseModelFrozen):
  """Impurity content defined by fractional abundances."""

  impurity_mode: Annotated[Literal['fractions'], torax_pydantic.JAX_STATIC] = (
      'fractions'
  )
  species: ImpurityMapping = torax_pydantic.ValidatedDefault({'Ne': 1.0})
  Z_override: torax_pydantic.TimeVaryingScalar | None = None
  A_override: torax_pydantic.TimeVaryingScalar | None = None

  def build_runtime_params(self, t: chex.Numeric) -> RuntimeParams:
    """Builds a RuntimeParams object at a given time."""
    fractions = {ion: value.get_value(t) for ion, value in self.species.items()}
    fractions_face = {
        ion: value.get_value(t, grid_type='face')
        for ion, value in self.species.items()
    }
    Z_override = None if not self.Z_override else self.Z_override.get_value(t)

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
      ion_key = list(self.species.keys())[0]
      A_avg = jnp.full_like(fractions[ion_key], self.A_override.get_value(t))
      A_avg_face = jnp.full_like(
          fractions_face[ion_key], self.A_override.get_value(t)
      )

    return RuntimeParams(
        fractions=fractions,
        fractions_face=fractions_face,
        A_avg=A_avg,
        A_avg_face=A_avg_face,
        Z_override=Z_override,
    )

  @pydantic.model_validator(mode='before')
  @classmethod
  def _conform_impurity_data(cls, data: dict[str, Any]) -> dict[str, Any]:
    """Ensures backward compatibility if infered that data in legacy format."""

    # Maps legacy inputs to the new API format and convert what would have been
    # TimeVaryingScalar inputs to an equivalent TimeVaryingArray.
    # TODO(b/434175938): Remove this once V1 API is deprecated.
    # This branch is hit when calling from the outer `PlasmaComposition`.
    if 'legacy' in data:
      del data['legacy']
      if 'species' in data and isinstance(data['species'], dict):
        new_species = {}
        for species, value in data['species'].items():
          new_species[species] = (
              torax_pydantic.TimeVaryingScalar.model_validate(value)
              .to_time_varying_array()
          )
        data['species'] = new_species

    # This branch is typically hit on using `update_fields`.
    if 'species' not in data and 'impurity_mode' not in data:
      if isinstance(data, dict):
        new_species = {}
        for species, value in data.items():
          new_species[species] = (
              torax_pydantic.TimeVaryingScalar.model_validate(value)
              .to_time_varying_array()
          )
      else:
        new_species = data
      return {'species': new_species, 'impurity_mode': _IMPURITY_MODE_FRACTIONS}

    return data

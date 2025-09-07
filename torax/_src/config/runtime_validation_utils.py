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

"""Utilities for validating the config inputs."""

from collections.abc import Mapping
import functools
import logging
from typing import Annotated, Any, Final, TypeAlias

import numpy as np
import pydantic
from torax._src import constants
from torax._src.torax_pydantic import torax_pydantic

_TOLERANCE: Final[float] = 1e-6


def time_varying_array_defined_at_1(
    time_varying_array: torax_pydantic.TimeVaryingArray,
) -> torax_pydantic.TimeVaryingArray:
  """Validates the input for the TimeVaryingArray."""
  if not time_varying_array.right_boundary_conditions_defined:
    logging.debug("""Not defined at rho=1.0.""")
  return time_varying_array


def time_varying_array_bounded(
    time_varying_array: torax_pydantic.TimeVaryingArray,
    lower_bound: float = -np.inf,
    upper_bound: float = np.inf,
) -> torax_pydantic.TimeVaryingArray:
  """Validates the input for the TimeVaryingArray."""
  for t, (_, values) in time_varying_array.value.items():
    if not np.all(values >= lower_bound):
      raise ValueError(
          f'Some values are smaller than lower bound {lower_bound} at time'
          f' {t}: {values}'
      )
    if not np.all(values <= upper_bound):
      raise ValueError(
          f'Some values are larger than upper bound {upper_bound} at time'
          f' {t}: {values}'
      )
  return time_varying_array


TimeVaryingArrayDefinedAtRightBoundaryAndBounded: TypeAlias = Annotated[
    torax_pydantic.TimeVaryingArray,
    pydantic.AfterValidator(time_varying_array_defined_at_1),
    pydantic.AfterValidator(
        functools.partial(
            time_varying_array_bounded,
            lower_bound=1.0,
        )
    ),
]


def _ion_mixture_before_validator(value: Any) -> Any:
  """Validates the input for the IonMixtureType."""
  if isinstance(value, str):
    return {value: 1.0}
  return value


def _ion_mixture_after_validator(
    value: Mapping[str, torax_pydantic.TimeVaryingScalar],
) -> Mapping[str, torax_pydantic.TimeVaryingScalar]:
  """Validates the input for the IonMixtureType."""
  if not value:
    raise ValueError('The species dictionary cannot be empty.')

  # Check if all species keys are in the allowed list.
  invalid_ion_symbols = set(value.keys()) - constants.ION_SYMBOLS
  if invalid_ion_symbols:
    raise ValueError(
        f'Invalid ion symbols: {invalid_ion_symbols}. Allowed symbols are:'
        f' {constants.ION_SYMBOLS}'
    )

  time_arrays = [v.time for v in value.values()]
  fraction_arrays = [v.value for v in value.values()]

  # Check if all time arrays are equal
  if not all(np.array_equal(time_arrays[0], x) for x in time_arrays[1:]):
    raise ValueError(
        'All time indices for ion mixture fractions must be equal.'
    )

  # Check if the ion fractions sum to 1 at all times
  fraction_sum = np.sum(fraction_arrays, axis=0)
  if not np.allclose(fraction_sum, 1.0, rtol=_TOLERANCE):
    raise ValueError(
        'Fractional concentrations in an IonMixture must sum to 1 at all times.'
    )
  return value


IonMapping: TypeAlias = Annotated[
    Mapping[str, torax_pydantic.TimeVaryingScalar],
    pydantic.BeforeValidator(_ion_mixture_before_validator),
    pydantic.AfterValidator(_ion_mixture_after_validator),
]


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

  # 1. Check that all TimeVaryingArray objects have the same time points.
  for species_key, tva in value.items():
    if tva.value.keys() != reference_times:
      raise ValueError(
          f'Inconsistent times for key "{species_key}". Expected'
          f' {sorted(list(reference_times))}, got'
          f' {sorted(list(tva.value.keys()))}'
      )

  for t in reference_times:
    # 2. Check that for each time point, all TimeVaryingArray objects have the
    # same `rho_norm` array.
    reference_rho_norm, _ = first_tva.value[t]
    for species_key, tva in value.items():
      current_rho_norm, _ = tva.value[t]
      if not np.array_equal(current_rho_norm, reference_rho_norm):
        raise ValueError(
            f'Inconsistent rho_norm for key "{species_key}" at time {t}.'
        )

    # 3. Check that for each time point, the `values` arrays across all
    # TimeVaryingArray objects have the same shape and sum to 1.0.
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
    Mapping[str, torax_pydantic.TimeVaryingArray],
    pydantic.BeforeValidator(_ion_mixture_before_validator),
    pydantic.AfterValidator(_impurity_after_validator),
]


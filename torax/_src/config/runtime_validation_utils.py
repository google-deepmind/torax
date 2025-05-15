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
from torax import constants
from torax.torax_pydantic import torax_pydantic


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

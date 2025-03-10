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

"""Pydantic custom types."""

from typing import Annotated, TypeAlias

import numpy as np
import pydantic


DataTypes: TypeAlias = float | int | bool
DtypeName: TypeAlias = str

NestedList: TypeAlias = (
    DataTypes
    | list[DataTypes]
    | list[list[DataTypes]]
    | list[list[list[DataTypes]]]
)

NumpySerialized: TypeAlias = tuple[DtypeName, NestedList]


def _numpy_array_before_validator(
    x: np.ndarray | NumpySerialized,
) -> np.ndarray:
  """Validates and converts a serialized NumPy array."""

  if isinstance(x, np.ndarray):
    return x
  # This can be either a tuple or a list. The list case is if this is coming
  # from JSON, which doesn't have a tuple type.
  elif isinstance(x, tuple) or isinstance(x, list) and len(x) == 2:
    dtype, data = x
    return np.array(data, dtype=np.dtype(dtype))
  else:
    raise ValueError(
        'Expected NumPy or a tuple representing a serialized NumPy array, but'
        f' got a {type(x)}'
    )


def _numpy_array_serializer(x: np.ndarray) -> NumpySerialized:
  return (x.dtype.name, x.tolist())


def _numpy_array_is_rank_1(x: np.ndarray) -> np.ndarray:
  if x.ndim != 1:
    raise ValueError(f'NumPy array is not 1D, rather of rank {x.ndim}')
  return x


NumpyArray = Annotated[
    np.ndarray,
    pydantic.BeforeValidator(_numpy_array_before_validator),
    pydantic.PlainSerializer(
        _numpy_array_serializer, return_type=NumpySerialized
    ),
]

NumpyArray1D = Annotated[
    NumpyArray, pydantic.AfterValidator(_numpy_array_is_rank_1)
]


def _array_is_unit_interval(array: np.ndarray) -> np.ndarray:
  """Checks if the array is in the unit interval."""
  if not np.all((array >= 0.0) & (array <= 1.0)):
    raise ValueError(
        f'Some array elements are not in the unit interval: {array}'
    )
  return array


NumpyArray1DUnitInterval = Annotated[
    NumpyArray1D,
    pydantic.AfterValidator(_array_is_unit_interval),
]

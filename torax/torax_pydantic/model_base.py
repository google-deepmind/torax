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

"""Pydantic utilities and base classes."""

from collections.abc import Mapping
from typing import Annotated, Any, TypeAlias
import jax
import numpy as np
import pydantic
from torax.geometry import geometry
from typing_extensions import Self

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

  if isinstance(x, np.ndarray):
    return x
  else:
    dtype, data = x
    return np.array(data, dtype=np.dtype(dtype))


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


class BaseModelFrozen(pydantic.BaseModel):
  """Base config class. Any custom config classes should inherit from this.

  No model fields are allowed to be assigned to after construction.

  See https://docs.pydantic.dev/latest/ for documentation on pydantic.

  This class is compatible with JAX, so can be used as an argument to a JITted
  function.
  """

  model_config = pydantic.ConfigDict(
      frozen=True,
      # Do not allow attributes not defined in pydantic model.
      extra='forbid',
      arbitrary_types_allowed=True,
  )

  def __new__(cls, *unused_args, **unused_kwargs):
    try:
      registered_cls = jax.tree_util.register_pytree_node_class(cls)
    except ValueError:
      registered_cls = cls  # Already registered.
    return super().__new__(registered_cls)

  def tree_flatten(self):

    children = tuple(getattr(self, k) for k in self.model_fields.keys())
    aux_data = None
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data

    init = {
        key: value
        for key, value in zip(cls.model_fields, children, strict=True)
    }
    # The model needs to be reconstructed without validation, as init can
    # contain JAX tracers inside a JIT, which will fail Pydantic validation. In
    # addition, validation is unecessary overhead.
    return cls.model_construct(**init)

  @classmethod
  def from_dict(cls: type[Self], cfg: Mapping[str, Any]) -> Self:
    return cls.model_validate(cfg)

  def to_dict(self) -> dict[str, Any]:
    return self.model_dump()

  def set_rho_norm_grid(self, grid: NumpyArray | geometry.Grid1D):
    """Sets the rho_norm_grid field in all TimeVaryingArray fields.

    This will set the grid to all sub-models as well.

    This function can only be called if the rho_norm_grid field is None.

    Args:
      grid: The grid to use for interpolation, either as a NumPy array or a
        geometry.Grid1D object.

    Raises:
      RuntimeError: If the rho_norm_grid field is not None.
    Returns:
      No return value.
    """

    for name in self.model_fields.keys():
      attr = getattr(self, name)
      if hasattr(attr, 'set_rho_norm_grid'):
        attr.set_rho_norm_grid(grid)

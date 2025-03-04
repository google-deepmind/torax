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

from collections.abc import Set
from typing import Annotated, Any, Final, Mapping, Sequence, TypeAlias
import jax
import numpy as np
import pydantic
import treelib
from typing_extensions import Self


TIME_INVARIANT: Final[str] = '_pydantic_time_invariant_field'

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


class BaseModelMutable(pydantic.BaseModel):
  """Base config class. Any custom config classes should inherit from this.

  See https://docs.pydantic.dev/latest/ for documentation on pydantic.

  This class is compatible with JAX, so can be used as an argument to a JITted
  function.
  """

  model_config = pydantic.ConfigDict(
      frozen=False,
      # Do not allow attributes not defined in pydantic model.
      extra='forbid',
      # Re-run validation if the model is updated.
      validate_assignment=True,
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

  @classmethod
  def time_invariant_fields(cls) -> tuple[str, ...]:
    """Returns the names of the time invariant fields in the model."""
    return tuple(
        k for k, v in cls.model_fields.items() if TIME_INVARIANT in v.metadata
    )

  def _get_direct_submodels(self) -> tuple[Self, ...]:
    """Return all direct submodels in the model."""

    def is_leaf(x):
      if isinstance(x, (Mapping, Sequence, Set)):
        return False
      return True

    leaves = jax.tree.flatten(self.__dict__, is_leaf=is_leaf)[0]
    return tuple(i for i in leaves if isinstance(i, BaseModelMutable))

  def _get_submodels(self) -> tuple[pydantic.BaseModel, ...]:
    """Return all submodels in the model.

    This will return all Pydantic models directly inside model fields, and
    inside container types: mappings, sequences, and sets.

    Returns:
      A tuple of all submodels in the model.
    """

    all_submodels = []
    new_submodels = self._get_direct_submodels()
    while new_submodels:
      new_submodels_temp = []
      for model in new_submodels:
        # assert isinstance(model, BaseModelMutable)
        all_submodels.append(model)
        new_submodels_temp += model._get_direct_submodels()  # pylint: disable=protected-access
      new_submodels = new_submodels_temp
    return tuple(all_submodels)

  def _has_unique_submodels(self) -> bool:
    """Returns True if all submodels are different instances of models."""
    submodels = self._get_submodels()
    unique_ids = set(id(m) for m in submodels)
    return len(submodels) == len(unique_ids)

  def tree_build(self) -> treelib.Tree:
    """Returns a treelib.Tree representation of a nested Pydantic model."""

    # The tree nodes are object IDs, which also allows easy node lookup. This
    # causes problems when the user creates a Pydantic model with shared objects
    # for different fields. As this cannot happen with the standard dict
    # constructor, we simply disallow this case. An alternative is to
    # automatically 'fix' the user model by making copies of duplicated
    # submodels. This could be implemented if a need arises.
    if not self._has_unique_submodels():
      raise ValueError(
          'Cannot build a `treelib.Tree` for a model with non-unique submodels.'
      )

    model_tree = treelib.Tree()
    model_tree.create_node(
        tag=self.__class__.__name__,
        identifier=id(self),
        data=self,
    )
    for model in self._get_direct_submodels():
      if model.__class__ is not BaseModelMutable:
        model_tree.paste(id(self), model.tree_build())
    return model_tree


class BaseModelFrozen(BaseModelMutable, frozen=True):
  """Base config with frozen fields.

  See https://docs.pydantic.dev/latest/ for documentation on pydantic.

  This class is compatible with JAX, so can be used as an argument to a JITted
  function.
  """

  ...

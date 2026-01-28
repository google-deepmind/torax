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
import functools
import inspect
from typing import Any, Final, Mapping, Sequence, TypeAlias

import jax
import pydantic
import treelib
from typing_extensions import Self

TIME_INVARIANT: Final[str] = '_pydantic_time_invariant_field'
JAX_STATIC: Final[str] = '_pydantic_jax_static_field'

StaticKwargs: TypeAlias = dict[str, Any]
DynamicArgs: TypeAlias = list[Any]


class BaseModelFrozen(pydantic.BaseModel):
  """Base config with frozen fields.

  See https://docs.pydantic.dev/latest/ for documentation on pydantic.

  This class is compatible with JAX, so can be used as an argument to a JITted
  function. Static fields can be annotated via
  `typing.Annotated[dtype, torax_pydantic.JAX_STATIC`] to make them static in
  the JAX tree. These fields must be hashable.
  """

  model_config = pydantic.ConfigDict(
      frozen=True,
      # Do not allow attributes not defined in pydantic model.
      extra='forbid',
      arbitrary_types_allowed=True,
      validate_default=True,
  )

  def __new__(cls, *unused_args, **unused_kwargs):
    try:
      registered_cls = jax.tree_util.register_pytree_node_class(cls)
    except ValueError:
      registered_cls = cls  # Already registered.
    return super().__new__(registered_cls)

  @classmethod
  @functools.cache
  def _jit_dynamic_kwarg_names(cls) -> tuple[str, ...]:
    return tuple(
        name
        for name in cls.model_fields.keys()
        if JAX_STATIC not in cls.model_fields[name].metadata
    )

  @classmethod
  @functools.cache
  def _jit_static_kwarg_names(cls) -> tuple[str, ...]:
    return tuple(
        name
        for name in cls.model_fields.keys()
        if JAX_STATIC in cls.model_fields[name].metadata
    )

  def tree_flatten(self) -> tuple[DynamicArgs, StaticKwargs]:
    """Flattens the model into a JAX dynamic and static argument tuple.

    Static arguments are model fields annotated via
    `typing.Annotated[dtype, torax_pydantic.JAX_STATIC]`. Dynamic arguments are
    all other fields.

    Required by the use of `jax.tree_util.register_pytree_node_class`.

    Returns:
      A tuple of the dynamic and static arguments. Dynamic arguments are a list
      of numeric values compatible with `jax.jit`. Static arguments are a
      dictionary of hashable values considered `static_argnames` by `jax.jit`.
    """
    static_names = self._jit_static_kwarg_names()
    dynamic_names = self._jit_dynamic_kwarg_names()
    static_children = {name: getattr(self, name) for name in static_names}
    dynamic_children = [getattr(self, name) for name in dynamic_names]

    return (dynamic_children, static_children)

  @classmethod
  def tree_unflatten(
      cls, aux_data: StaticKwargs, children: DynamicArgs
  ) -> Self:
    """Reconstructs a model from a JAX dynamic and static argument tuple.

    Required by the use of `jax.tree_util.register_pytree_node_class`.

    Args:
      aux_data: A dictionary of static arguments.
      children: A list of dynamic arguments.

    Returns:
      A model instance.
    """
    dynamic_kwargs = {
        name: value
        for name, value in zip(
            cls._jit_dynamic_kwarg_names(), children, strict=True
        )
    }
    # The model needs to be reconstructed without validation, as init can
    # contain JAX tracers inside a JIT, which will fail Pydantic validation. In
    # addition, validation is unnecessary overhead.
    return cls.model_construct(**(dynamic_kwargs | aux_data))

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

  @property
  def _direct_submodels(self) -> tuple[Self, ...]:
    """Direct submodels in the model."""

    def is_leaf(x):
      if isinstance(x, (Mapping, Sequence, Set)):
        return False
      return True

    # Exclude non-field values in __dict__, such as cached_properties.
    leaves = {k: self.__dict__[k] for k in self.__class__.model_fields.keys()}
    # Some Pydantic models are values of a dict. We flatten the tree to access
    # them.
    leaves = jax.tree.flatten(leaves, is_leaf=is_leaf)[0]
    return tuple(i for i in leaves if isinstance(i, BaseModelFrozen))

  @property
  def submodels(self) -> tuple[Self, ...]:
    """A tuple of the model and all submodels.

    This will return all Pydantic models directly inside model fields, and
    inside container types: mappings, sequences, and sets.

    Returns:
      A tuple of the model and all model submodels.
    """

    all_submodels = [self]
    new_submodels = self._direct_submodels
    while new_submodels:
      new_submodels_temp = []
      for model in new_submodels:
        all_submodels.append(model)
        new_submodels_temp += model._direct_submodels  # pylint: disable=protected-access
      new_submodels = new_submodels_temp
    return tuple(all_submodels)

  @property
  def _has_unique_submodels(self) -> bool:
    """Returns True if all submodels are different instances of models."""
    submodels = self.submodels
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
    if not self._has_unique_submodels:
      raise ValueError(
          'Cannot build a `treelib.Tree` for a model with non-unique submodels.'
      )

    model_tree = treelib.Tree()
    model_tree.create_node(
        tag=self.__class__.__name__,
        identifier=id(self),
        data=self,
    )
    for model in self._direct_submodels:
      if model.__class__ is not BaseModelFrozen:
        model_tree.paste(id(self), model.tree_build())
    return model_tree

  def clear_cached_properties(self, exceptions: Sequence[str] | None = None):
    """Clears all `functools.cached_property` caches in the model.

    Args:
      exceptions: A sequence of property names to exclude from clearing.
    """
    cached_properties = []
    for name, value in inspect.getmembers(self.__class__):
      if isinstance(value, functools.cached_property):
        cached_properties.append(name)
    exceptions = {} if exceptions is None else exceptions
    cached_properties = set(cached_properties) - set(exceptions)
    for p in cached_properties:
      # Note: this is not the idiomatic way to clear a cached property, which
      # is `del self.some_property`. This doesn't work with `del getattr(...)`
      # and Pydantic frozen models banned `delattr`.
      if p in self.__dict__:
        del self.__dict__[p]

  def _update_fields(self, x: Mapping[str, Any]):
    """Safely update fields a nested BaseModelFrozen.

    This method will invalidate all `functools.cached_property` caches of
    all ancestral models in the nested tree, as these could have a dependency
    on the updated model. In addition, these ancestral models will be
    re-validated.

    If the value to be updated is a field of a Pydantic model, any value
    conformable to the field type will be accepted.

    Args:
      x: A dictionary whose key is a path `'some.path.to.field_name'` and the
        `value` is the new value for `field_name`. The path can be dictionary
        keys or attribute names, but `field_name` must be an attribute of a
        Pydantic model.

    Raises:
      ValueError: all submodels must be unique object instances. A `ValueError`
        will be raised if this is not the case.
    """
    model_tree = self.tree_build()
    mutated_models = []

    for path, value in x.items():
      path_split = path.split('.')
      value_name = path_split.pop()
      model = self._lookup_path(path_split)
      mutated_models.append(model)

      if not isinstance(model, BaseModelFrozen) or (
          value_name not in model.__class__.model_fields
      ):
        raise ValueError(
            f'The path {path} does not refer to a field of a Pydantic'
            ' BaseModelFrozen model.'
        )
      assert value_name in model.__dict__
      field_type = model.__class__.model_fields[value_name].annotation

      # TypeAdapter does not allow a config arg if the value is a Pydantic
      # model, as this has its own config.
      cfg = pydantic.ConfigDict(arbitrary_types_allowed=True)
      try:
        cfg = None if issubclass(field_type, pydantic.BaseModel) else cfg
      except TypeError:
        pass
      value = pydantic.TypeAdapter(field_type, config=cfg).validate_python(
          value
      )
      model.__dict__[value_name] = value

    for model in mutated_models:
      for model_ancestral in model_tree.rsearch(id(model)):
        m = model_tree.get_node(model_ancestral).data
        m.clear_cached_properties()
        # Re-validate all ancestral models.
        m.__class__.from_dict(m.to_dict())

  def _lookup_path(self, paths: Sequence[str]) -> Self:
    """Returns the model at the given path."""
    value = self
    for path in paths:
      if isinstance(value, BaseModelFrozen):
        if path not in value.__class__.model_fields:
          raise ValueError(
              f'The path {".".join(paths)} does not refer to a field of a'
              ' Pydantic BaseModelFrozen model.'
          )
        value = getattr(value, path)
      elif isinstance(value, dict):
        value = value[path]
      else:
        raise ValueError(f'Cannot look up path {path} in {value}')
    if not isinstance(value, BaseModelFrozen):
      raise ValueError(f'The value at path {paths} is not a Pydantic model.')
    return value

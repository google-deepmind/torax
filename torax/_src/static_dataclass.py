# Copyright 2025 DeepMind Technologies Limited
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
"""The StaticDataclass class.

This is a base class to be used to define classes to be used as static
arguments to jitted JAX functions. Ideally, all classes in TORAX used as
static arguments should inherit from this class. Static arguments need
to be immutable and need to support comparison and hashing by value in
a way that works with the persistent cache. This process has many
"footguns" and this class has been designed to help avoid them.
"""

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True, eq=False)
class StaticDataclass:
  """Empty base class used to define static arguments to JAX.

  This class and its subclasses are to be used as static arguments to jitted
  JAX functions. Specifically, inheriting from this class ensures that a
  new class will be immutable and support comparison and hashing by
  value.

  Subclass responsibilities:
  - (This will cause silent wrong computations if the wrong thing is done) The
    __hash__ method here assumes that the fields of this class and any
   subclasses are immutable and support comparison and hashing by value. If a
   subclass needs to introduce special fields (that shouldn't be hashed, or
   be hashed with a special procedure) the subclass must implement its own
   __hash__ method. For example, the subclass should not have a Python
   function as a field, unless the subclass implements its own __hash__
   function that has some way of correctly hashing that field.
   Implementing a custom hashing strategy also requires overriding
   __post_init__ to turn off the checks that the default strategy is in use.
   While StaticDataclass does its best to automatically detect and prevent
   problems, this is impossible to do in the general case; you must still
   think about how your class should be hashed.
  - (This will be automatically detected if not done) The subclass must also
   be decorated with @dataclasses.dataclass(frozen=True,eq=False)
  """

  def as_tuple(self) -> tuple[Any, ...]:
    """Returns a tuple containing the value of each field.

    This is different from dataclasses.as_tuple, which recursively crawls
    nested dataclasses to return a tree-structured tuple containing no
    dataclasses. `self.as_tuple` is always a flat tuple and may contain
    dataclasses.
    """

    return tuple(
        getattr(self, field.name) for field in dataclasses.fields(self)
    )

  def _full_tuple(self) -> tuple[Any, ...]:
    """Returns a tuple containing everything needed to hash the class.

    As of this writing, that is:
    - The fields of the class (which will recursively include the analogous
      information in their own hash when hash is called on the resulting
      tuple)
    - A representation of the class ID

    self.as_tuple() may not return everything that needs to be hashed.
    In particular, polymorphic classes need to hash a class ID.

    Returns:
      full_tuple: Contents of self.as_tuple() augmented with everything
                  else that neeeds to be hashed.
    """
    # Create a fully qualified class name string (e.g., 'my_module.MyClass')
    class_id = f"{self.__class__.__module__}.{self.__class__.__qualname__}"

    return self.as_tuple() + (class_id,)

  def __eq__(self, other: Any) -> bool:
    """One __eq__ method for the whole class hierarchy.

    Mostly rely on dataclass behavior for comparison, but also enforce that
    two objects are not the same if they are not the same class.

    Args:
      other: The object to compare to.

    Returns:
      True if the objects are equal, False otherwise.
    """

    # The role of this if statement is not to make sure the class id matches,
    # it is to make sure that `other` supports the `_full_tuple` interface.
    # The class id will be in _full_tuple and will be checked next.
    if not isinstance(other, StaticDataclass):
      return False

    return self._full_tuple() == other._full_tuple()

  def __hash__(self) -> int:
    """One __hash__ method for the whole class hierarchy.

    Mostly rely on dataclass behavior for hashing, but also hash the class
    id.

    Returns:
      The hash of the object.
    """

    return self._hash()

  def _hash(self) -> int:
    """Hash function implementation."""
    return hash(self._full_tuple())

  def __post_init__(self):

    # Make sure the StaticDataclass method is used
    if hash(self) != self._hash():
      raise TypeError(
          "A subclass of StaticDataclass is not using the base "
          "StaticDataclass.__hash__ method. This is usually "
          "accidental and caused by forgetting to pass eq=False "
          "to the dataclass decorator on the subclass. If this is"
          "intentional then override the __post_init__ method too"
          " to avoid this check."
      )

    # Attempt to make sure hashing by id is not used. In general this is
    # not possible as users may hash a variety of functions of the id and
    # after Python 3.3 the mapping from id to hash is randomized
    values = self.as_tuple()
    hashes = tuple(hash(v) for v in values)
    bad_hashes = tuple(
        (
            id(v) // 16,  # Python 2.7-3.2 default hash
            hash(id(v)),
        )  # Naive user-provided hash by id
        for v in values
    )
    names = [field.name for field in dataclasses.fields(self)]
    for h, bh, n in zip(hashes, bad_hashes, names):
      if h in bh:
        raise TypeError(
            f"{self}.{n} hashes by id when it should hash by value."
        )

    # Make sure nested dataclasses are StaticDataclass
    for v, n in zip(values, names):
      if dataclasses.is_dataclass(v) and not isinstance(v, StaticDataclass):
        raise TypeError(
            f"{self}.{n} is a dataclass but not a "
            "StaticDataclass. The whole hierarchy must be "
            "StaticDataclass objects to ensure that all hash "
            "functions hash the class id."
        )

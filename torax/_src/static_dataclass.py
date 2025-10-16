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
arguments to jitted Jax functions. Ideally, all classes in Torax used as
static arguments should inherit from this class. Static arguments need
to be immutable and need to support comparison and hashing by value in
a way that works with the persistent cache. This process has many
"footguns" and this class has been designed to help avoid them.
"""

import dataclasses


@dataclasses.dataclass(frozen=True, eq=False)
class StaticDataclass:
  """Empty base class used to define static arguments to Jax.

  This class and its subclasses are to be used as static arguments to jitted
  Jax functions. Specifically, inheriting from this class ensures that a
  new class will be immutable and support comparison and hashing by
  value.

  Subclass responsibilities:
  - (This will cause silent wrong computations if the wrong thing is done) The
    __hash__ method here assumes that the fields of this class and any
   subclasses are immutable and support comparison and hashing by value in a
   way that is compatible with the persistent cache. If a subclass needs to
   introduce special fields (that shouldn't be hashed, or be hashed with
   a special procedure) the subclass must implement its own __hash__ method.
   For example, the subclass should not have a Python function as a field,
   unless the subclass implements its own __hash__ function that has some
   way of correctly hashing that field.
  - (This will be automatically detected if not done) The subclass must also
   be decorated with @dataclasses.dataclass(frozen=True,eq=False)
  """

  def _full_tuple(self):
    """Returns a description of the object as a tuple including class."""
    # Create a fully qualified class name string (e.g., 'my_module.MyClass').
    # This string is stable across interpreter sessions.
    class_id = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
    return dataclasses.astuple(self) + (class_id,)

  def __eq__(self, other):
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

  def __hash__(self):
    """One __hash__ method for the whole class hierarchy.

    Mostly rely on dataclass behavior for hashing, but also hash the class
    id.

    Returns:
      The hash of the object.
    """

    return self._hash()

  def _hash(self):
    """Hash function implementation."""
    return hash(self._full_tuple())

  def __post_init__(self):
    # Subclasses must remember to pass eq=False to the dataclasses decorator,
    # or the decorator will overwrite our custom __hash__ method.
    # This assert guards against that.
    # If subclasses need to implement a different __hash__ function
    # intentionally they can override __post_init__ to turn off this check.
    assert hash(self) == self._hash()

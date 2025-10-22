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
from __future__ import annotations

import dataclasses

from absl.testing import absltest
from torax._src import static_dataclass


@dataclasses.dataclass(frozen=True, eq=False)
class MyData(static_dataclass.StaticDataclass):
  x: int
  y: str


@dataclasses.dataclass(frozen=True, eq=False)
class MySubclass(MyData):
  pass


@dataclasses.dataclass(frozen=True, eq=False)
class MyDataWithDefaults(static_dataclass.StaticDataclass):
  x: int = 10
  y: str = "hello"


@dataclasses.dataclass(frozen=True, eq=False)
class AnotherData(static_dataclass.StaticDataclass):
  x: int
  y: str


class HashesById:
  """A test class that uses default Python hashing by id.

  This object should not be allowed as a field of StaticDataclass.
  """

  def __hash__(self):

    # Explicitly implement the Python 2.7-3.2 behavior so the test
    # will work for later Pythons.
    # If we didn't do this, we would need to raise a skiptest.
    # The test makes sure StaticDataset can automatically reject
    # hash by id in Python 2.7-3.2. For Python 3.3 and beyond,
    # the relationship between id and hash may be randomized so
    # it is up to the user to ensure that fields do not use
    # hashing by id.
    return id(self) // 16


@dataclasses.dataclass(frozen=True, eq=False)
class NotAStaticDataclass:
  """A test class that is a dataclass and not a static dataclass.

  This object should not be allowed as a field of StaticDataclass
  because it doesn't hash its class id.
  """

  pass


@dataclasses.dataclass(frozen=True, eq=False)
class TreeLeaf(static_dataclass.StaticDataclass):
  """A test class for building hierarchical dataclasses."""


class TreeLeafA(TreeLeaf):
  """A test class for testing polymorphism in dataclass trees."""


class TreeLeafB(TreeLeaf):
  """A test class for testing polymorphism in dataclass trees."""


@dataclasses.dataclass(frozen=True, eq=False)
class TreeInternal(static_dataclass.StaticDataclass):
  """A test class for building hierarchical dataclasses."""

  children: tuple[TreeLeaf | TreeInternal, ...]


class StaticDataclassTest(absltest.TestCase):

  def test_frozen(self):
    d = MyData(x=1, y="a")
    with self.assertRaises(dataclasses.FrozenInstanceError):
      d.x = 2
    with self.assertRaises(dataclasses.FrozenInstanceError):
      d.z = 3

  def test_defaults(self):
    d = MyDataWithDefaults()
    self.assertEqual(d.x, 10)
    self.assertEqual(d.y, "hello")
    d2 = MyDataWithDefaults(x=20)
    self.assertEqual(d2.x, 20)
    self.assertEqual(d2.y, "hello")

  def test_eq(self):
    d1 = MyData(x=1, y="a")
    d2 = MyData(x=1, y="a")
    d3 = MyData(x=2, y="a")
    d4 = MyData(x=1, y="b")
    d5 = AnotherData(x=1, y="a")
    d6 = MySubclass(x=1, y="a")
    self.assertEqual(d1, d2)
    self.assertNotEqual(d1, d3)
    self.assertNotEqual(d1, d4)
    self.assertNotEqual(d1, d5)  # Different class
    self.assertNotEqual(d1, "completely different type")
    self.assertNotEqual(d1, d6)  # Same base class, different subclass

  def test_hash(self):
    d1 = MyData(x=1, y="a")
    d2 = MyData(x=1, y="a")
    d3 = MyData(x=2, y="a")
    d5 = AnotherData(x=1, y="a")
    d6 = MySubclass(x=1, y="a")
    self.assertEqual(hash(d1), hash(d2))
    self.assertNotEqual(hash(d1), hash(d3))
    self.assertNotEqual(hash(d1), hash(d5))  # Different class
    self.assertNotEqual(
        hash(d1), hash(d6)
    )  # Same base class, different subclass

  def test_eq_true(self):
    with self.assertRaises(TypeError):

      @dataclasses.dataclass(frozen=True)  # eq=False is missing
      class MissingEqFalse(static_dataclass.StaticDataclass):
        z: int

      MissingEqFalse(z=1)

  def test_subclass_missing_decorator(self):
    with self.assertRaises(TypeError):

      class NotADataclass(static_dataclass.StaticDataclass):
        z: int

      NotADataclass(z=1)  # pytype: disable=wrong-keyword-args

  def test_field_hashes_by_id(self):

    @dataclasses.dataclass(frozen=True, eq=False)
    class HasBadField(static_dataclass.StaticDataclass):
      hashes_by_id: HashesById

    hashes_by_id = HashesById()

    with self.assertRaises(TypeError):
      HasBadField(hashes_by_id=hashes_by_id)

  def test_dataclass_field_is_static(self):

    not_a_static_dataclass = NotAStaticDataclass()

    @dataclasses.dataclass(frozen=True, eq=False)
    class HasBadField(static_dataclass.StaticDataclass):
      not_a_static_dataclass: NotAStaticDataclass

    with self.assertRaises(TypeError):
      HasBadField(not_a_static_dataclass=not_a_static_dataclass)

  def test_equivalent_trees_hash_same(self):

    def make_tree():
      return TreeInternal(
          children=(
              TreeLeafA(),
              TreeInternal(children=(TreeLeafA(), TreeLeafB())),
          )
      )

    first = make_tree()
    second = make_tree()
    self.assertEqual(hash(first), hash(second))

  def test_different_leaf_class_changes_hash(self):

    # This class makes sure that static dataclasses hash their class id and
    # that the hashed class id propagates up to affect the root of the nested
    # dataclass hierarchy.

    def make_tree(cls):
      """Makes a nested tree of dataclasses.

      Args:
        cls: The class to use for one leaf dataclass.

      Returns:
        tree: A tree of dataclasses, with one particular leaf node set to
              use `cls`.
      """
      return TreeInternal(
          children=(TreeLeafA(), TreeInternal(children=(TreeLeafA(), cls())))
      )

    tree_a = make_tree(TreeLeafA)
    tree_b = make_tree(TreeLeafB)

    self.assertNotEqual(hash(tree_a), hash(tree_b))


if __name__ == "__main__":
  absltest.main()

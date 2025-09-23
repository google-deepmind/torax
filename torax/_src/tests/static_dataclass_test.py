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

  def test_subclass_missing_decorator(self):
    with self.assertRaises(AssertionError):

      @dataclasses.dataclass(frozen=True)  # eq=False is missing
      class MissingEqFalse(static_dataclass.StaticDataclass):
        z: int

      MissingEqFalse(z=1)

    with self.assertRaises(TypeError):

      class NotADataclass(static_dataclass.StaticDataclass):
        z: int

      NotADataclass(z=1)  # pytype: disable=wrong-keyword-args


if __name__ == "__main__":
  absltest.main()

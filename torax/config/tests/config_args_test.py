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

import dataclasses
from absl.testing import absltest
from absl.testing import parameterized
from torax.config import config_args


class ConfigArgsTest(parameterized.TestCase):

  def test_recursive_replace(self):
    """Basic test of recursive replace."""

    # Make an nested dataclass with 3 layers of nesting, to make sure we can
    # handle root, internal, and leaf nodes. Make sure we can run a
    # recursive_replace call that is able to update some values but leave other
    # values untouched, at every level of the nested hierarchy.

    instance = A()

    # Change all values so we notice if `replace` fills them in with constructor
    # defaults
    instance.a1 = 7
    instance.a2 = 8
    instance.a3.b1 = 9
    instance.a3.b2 = 10
    instance.a3.b3.c1 = 11
    instance.a3.b3.c2 = 12
    instance.a3.b4.c1 = 13
    instance.a3.b4.c2 = 14
    instance.a4.b1 = 15
    instance.a4.b2 = 16
    instance.a4.b3.c1 = 17
    instance.a4.b3.c2 = 18
    instance.a4.b4.c1 = 19
    instance.a4.b4.c2 = 20

    changes = {
        'a1': -1,
        # Don't update a2, to test that it is untouched
        'a3': {
            # Don't update b1, to test that it is untouched
            'b2': -2,
            # Don't update b3, to test that it is untouched
            'b4': {
                'c1': -3,
                # Don't update c2, to test that it is untouched
            },
        },
        # Don't update a4, to test that it is untouched
    }

    result = config_args.recursive_replace(instance, **changes)

    self.assertIsInstance(result, A)
    self.assertEqual(result.a1, -1)
    self.assertEqual(result.a2, 8)
    self.assertIsInstance(result.a3, B)
    self.assertEqual(result.a3.b1, 9)
    self.assertEqual(result.a3.b2, -2)
    self.assertIsInstance(result.a3.b3, C)
    self.assertEqual(result.a3.b3.c1, 11)
    self.assertEqual(result.a3.b3.c2, 12)
    self.assertIsInstance(result.a3.b4, C)
    self.assertEqual(result.a3.b4.c1, -3)
    self.assertEqual(result.a3.b4.c2, 14)
    self.assertIsInstance(result.a4, B)
    self.assertEqual(result.a4.b1, 15)
    self.assertEqual(result.a4.b2, 16)
    self.assertIsInstance(result.a4.b3, C)
    self.assertEqual(result.a4.b3.c1, 17)
    self.assertEqual(result.a4.b3.c2, 18)
    self.assertIsInstance(result.a4.b4, C)
    self.assertEqual(result.a4.b4.c1, 19)
    self.assertEqual(result.a4.b4.c2, 20)


@dataclasses.dataclass
class C:
  c1: int = 1
  c2: int = 2


@dataclasses.dataclass
class B:
  b1: int = 3
  b2: int = 4
  b3: C = dataclasses.field(default_factory=C)
  b4: C = dataclasses.field(default_factory=C)


@dataclasses.dataclass
class A:
  a1: int = 5
  a2: int = 6
  a3: B = dataclasses.field(default_factory=B)
  a4: B = dataclasses.field(default_factory=B)


if __name__ == '__main__':
  absltest.main()

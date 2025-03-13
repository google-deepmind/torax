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

"""Unit tests for the `torax.torax_pydantic.model_base` module."""

import functools
from typing import Annotated, Any
from absl.testing import absltest
from absl.testing import parameterized
import jax
import pydantic
from torax.torax_pydantic import model_base
from torax.torax_pydantic import torax_pydantic


class PydanticBaseTest(parameterized.TestCase):

  def test_model_base_frozen(self):

    class TestModel(model_base.BaseModelFrozen):
      x: float
      y: float

    m = TestModel(y=4.0, x=2.0)

    with self.subTest('frozen_model_cannot_be_updated'):
      with self.assertRaises(ValueError):
        m.x = 2.0

  @parameterized.parameters(True, False)
  def test_model_base_map_pytree(self, frozen: bool):

    if frozen:

      class TestModel(model_base.BaseModelFrozen):
        x: float
        y: float

    else:

      class TestModel(model_base.BaseModelFrozen):
        x: float
        y: float

    m = TestModel(x=2.0, y=4.0)
    m2 = jax.tree_util.tree_map(lambda x: x**2, m)

    self.assertEqual(m2.x, 4.0)
    self.assertEqual(m2.y, 16.0)

    @jax.jit
    def f(data):
      return data.x * data.y

    with self.subTest('jit_works'):
      self.assertEqual(f(m), m.x * m.y)

  def test_model_field_metadata(self):

    class TestModel(model_base.BaseModelFrozen):
      x: torax_pydantic.Second
      y: Annotated[
          torax_pydantic.Meter, 'other_metadata', model_base.TIME_INVARIANT
      ]
      z: Annotated[
          Annotated[torax_pydantic.OpenUnitInterval, model_base.TIME_INVARIANT],
          'other_metadata',
      ]

    m = TestModel(x=2.0, y=4.0, z=0.1)

    with self.subTest('time_invariant_fields'):
      self.assertEqual(('y', 'z'), m.time_invariant_fields())

    with self.subTest('invalid_meter'):
      with self.assertRaises(ValueError):
        TestModel(x=2.0, y=-4.0, z=0.1)

    with self.subTest('invalid_open_unit_interval'):
      with self.assertRaises(ValueError):
        TestModel(x=2.0, y=4.0, z=1.0)

  def test_nested_model_graph(self):

    class Test1(model_base.BaseModelFrozen):
      x: bool = False

    class Test2(model_base.BaseModelFrozen):
      x: dict[str, Any]
      y: int
      z: list[tuple[Test1, Test1, int]]  # pytype: disable=invalid-annotation

    class Test3(model_base.BaseModelFrozen):
      x: tuple[Test1, Test2, Test1]  # pytype: disable=invalid-annotation
      y: dict[str, int]

    t1 = Test1(x=True)
    t2 = Test2(x=dict(t1=Test1(), t2='dd'), y=2, z=[(Test1(), Test1(), 34)])
    t3 = Test3(x=(t1, t2, Test1()), y={'test': 2})

    model_tree_1 = t1.tree_build()
    model_tree_2 = t2.tree_build()
    model_tree_3 = t3.tree_build()

    with self.subTest('tree_size'):
      self.assertEqual(model_tree_1.size(), 1)
      self.assertEqual(model_tree_2.size(), 4)
      self.assertEqual(model_tree_3.size(), 7)

    with self.subTest('tree_consistency_get_submodels'):
      self.assertTrue(model_tree_3.size(), len(t3.submodels))

    with self.subTest('tree_depth'):
      self.assertEqual(model_tree_1.depth(), 0)
      self.assertEqual(model_tree_2.depth(), 1)
      self.assertEqual(model_tree_3.depth(), 2)

    with self.subTest('tree_depth'):
      self.assertEqual(model_tree_1.depth(), 0)
      self.assertEqual(model_tree_2.depth(), 1)
      self.assertEqual(model_tree_3.depth(), 2)

    # Test that data is correctly associated with nodes.
    with self.subTest('tree_3_leaves'):
      leaves = model_tree_3.leaves()
      self.assertLen(leaves, 5)
      leaf_ids = set(id(l.data) for l in leaves)
      leaf_ids_expected = {
          id(t1),
          id(t3.x[2]),
          id(t2.x['t1']),
          id(t2.z[0][0]),
          id(t2.z[0][1]),
      }
      self.assertEqual(leaf_ids, leaf_ids_expected)

    with self.subTest('tree_json'):
      json_1 = '"Test1"'
      json_2 = '{"Test2": {"children": ["Test1", "Test1", "Test1"]}}'
      json_3 = (
          '{"Test3": {"children": ["Test1", "Test1", {"Test2": {"children":'
          ' ["Test1", "Test1", "Test1"]}}]}}'
      )

      self.assertEqual(model_tree_1.to_json(), json_1)
      self.assertEqual(model_tree_2.to_json(), json_2)
      self.assertEqual(model_tree_3.to_json(), json_3)

  def test_nested_model_non_unique_submodels(self):

    class Test1(model_base.BaseModelFrozen):
      x: bool = False

    class Test2(model_base.BaseModelFrozen):
      x: Test1  # pytype: disable=invalid-annotation
      y: Test1  # pytype: disable=invalid-annotation

    t1 = Test1(x=True)
    t2 = Test2(x=t1, y=t1)

    with self.assertRaisesRegex(ValueError, 'model with non-unique submodels'):
      t2.tree_build()

  def test_update_fields(self):
    class Test1(model_base.BaseModelFrozen):
      x: float

      @functools.cached_property
      def get_x(self):
        return self.x

    class Test2(model_base.BaseModelFrozen):
      x: pydantic.PositiveFloat
      y: Test1
      z: Test1

      @functools.cached_property
      def get_yx(self):
        return self.y.x

    class Test3(model_base.BaseModelFrozen):
      x: Test1
      y: Test2

      @functools.cached_property
      def get_square(self):
        return self.x.x**2

    class Test4(model_base.BaseModelFrozen):
      x: Test1
      y: Test2
      z: Test3

      @functools.cached_property
      def get_yyx(self):
        return self.y.y.x

    x_ref = 4.0
    model_2 = Test2(x=0.1, y=Test1(x=x_ref), z=Test1(x=-1.0))
    model_3 = Test3(
        x=Test1(x=0.3),
        y=Test2(x=0.4, y=Test1(x=5.0), z=Test1(x=-4.0)),
    )
    model = Test4(x=Test1(x=1.0), y=model_2, z=model_3)

    with self.subTest('model_3_cache'):
      self.assertNotIn('get_square', model.z.__dict__)
      self.assertEqual(model.z.get_square, model.z.x.x**2)
      self.assertIn('get_square', model.z.__dict__)

    with self.subTest('check_getters_correct'):
      # This also sets the cache.
      self.assertEqual(model.y.y.get_x, x_ref)
      self.assertEqual(model.y.get_yx, x_ref)
      self.assertEqual(model.get_yyx, x_ref)

    new_x = 99.0
    model._update_fields({'y.y.x': new_x})

    with self.subTest('check_cache_invalidated'):
      self.assertEqual(model.y.y.x, new_x)
      self.assertEqual(model.y.y.get_x, new_x)
      self.assertEqual(model.y.get_yx, new_x)
      self.assertEqual(model.get_yyx, new_x)

    # The field update should not have invalidated the cache of Test3.
    with self.subTest('check_test_3_cache_not_invalidated'):
      self.assertIn('get_square', Test3.__dict__)

    with self.subTest('updates_trigger_validation'):
      with self.assertRaises(pydantic.ValidationError):
        model_2._update_fields({'x': -1.0})

    with self.subTest('invalid_path'):
      with self.assertRaisesRegex(ValueError, 'Cannot update field'):
        model._update_fields({'x.zz': -1.0})

  def test_update_fields_dict(self):
    class Test1(model_base.BaseModelFrozen):
      x: float

    class Test2(model_base.BaseModelFrozen):
      y: dict[str, dict[str, Test1]]  # pytype: disable=invalid-annotation

    model_1 = Test1(x=1.0)
    model = Test2(y={'test1': {'test2': model_1}})
    new_val = 9.0
    model._update_fields({'y.test1.test2.x': new_val})
    self.assertEqual(model.y['test1']['test2'].x, new_val)

  def test_unique_submodels(self):
    class Test1(model_base.BaseModelFrozen):
      x: float

    class Test2(model_base.BaseModelFrozen):
      x: list[Test1]  # pytype: disable=invalid-annotation
      y: float
      z: Test1  # pytype: disable=invalid-annotation

    t1_1 = Test1(x=1.0)
    t1_2 = Test1(x=2.0)
    t2 = Test2(x=[t1_1, t1_2], y=3.0, z=t1_2)

    with self.subTest('number_of_submodels'):
      self.assertLen(t2.submodels, 4)

    with self.subTest('unique_submodels'):
      submodels_set = set(id(m) for m in t2.submodels)
      self.assertSetEqual(submodels_set, {id(t2), id(t1_2), id(t1_1)})

  def test_update_fields_conform(self):
    class Test1(model_base.BaseModelFrozen):
      x: float
      y: float

      @pydantic.model_validator(mode='before')
      @classmethod
      def conform_string(cls, data: Any) -> Any:
        if isinstance(data, dict):
          return data
        x, y = data.split(';')
        return dict(x=float(x), y=float(y))

    class Test2(model_base.BaseModelFrozen):
      x: dict[int, Test1]
      y: Test1

    t = Test2(x={1: Test1(x=1.0, y=2.0)}, y=Test1(x=3.0, y=4.0))
    t._update_fields({'x': {1: '10.;11.0'}, 'y': Test1(x=12.0, y=13.0)})

    self.assertEqual(t.x[1].x, 10.0)
    self.assertEqual(t.x[1].y, 11.0)
    self.assertEqual(t.y.x, 12.0)
    self.assertEqual(t.y.y, 13.0)


if __name__ == '__main__':
  absltest.main()

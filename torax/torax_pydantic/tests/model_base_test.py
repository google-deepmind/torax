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
import numpy as np
import pydantic
from torax.torax_pydantic import model_base
from torax.torax_pydantic import torax_pydantic


class PydanticBaseTest(parameterized.TestCase):

  def test_numpy_array_serializer(self):
    """Tests that interpolated vars are only constructed once."""

    class TestModel(pydantic.BaseModel):
      x: model_base.NumpyArray
      y: model_base.NumpyArray
      z: tuple[model_base.NumpyArray1D, float]

      model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    model = TestModel(
        x=np.random.rand(2, 5, 1).astype(np.float64),
        y=np.array(2.3479, dtype=np.float32),
        z=(np.array([True, False, False, True], dtype=np.bool_), -304.5),
    )

    model_dict = model.model_dump()
    model_from_dict = TestModel.model_validate(model_dict)

    with self.subTest('dict_dump_and_load'):
      np.testing.assert_array_equal(model.x, model_from_dict.x, strict=True)
      np.testing.assert_array_equal(model.y, model_from_dict.y, strict=True)
      np.testing.assert_array_equal(
          model.z[0], model_from_dict.z[0], strict=True
      )

    with self.subTest('json_dump_and_load'):
      model_json = model.model_dump_json()
      model_from_json = model.model_validate_json(model_json)
      np.testing.assert_array_equal(model.x, model_from_json.x, strict=True)
      np.testing.assert_array_equal(model.y, model_from_json.y, strict=True)
      np.testing.assert_array_equal(
          model.z[0], model_from_json.z[0], strict=True
      )

  def test_1d_array(self):
    array = pydantic.TypeAdapter(
        model_base.NumpyArray1D,
        config=pydantic.ConfigDict(arbitrary_types_allowed=True),
    )

    # Fail with 2D array.
    with self.assertRaises(ValueError):
      array.validate_python(np.array([[1.0, 2.0], [3.0, 4.0]]))

  def test_model_base_frozen(self):

    class TestModel(model_base.BaseModelFrozen):
      x: float
      y: float

    m = TestModel(y=4.0, x=2.0)

    with self.subTest('frozen_model_cannot_be_updated'):
      with self.assertRaises(ValueError):
        m.x = 2.0

  def test_model_base(self):

    class Test(model_base.BaseModelMutable, validate_assignment=True):
      name: str

      @functools.cached_property
      def computed(self):
        return self.name + '_test'  # pytype: disable=attribute-error

      @pydantic.model_validator(mode='after')
      def validate(self):
        if hasattr(self, 'computed'):
          del self.computed
        return self

    m = Test(name='test_string')
    self.assertEqual(m.computed, 'test_string_test')

    with self.subTest('field_is_mutable'):
      m.name = 'new_test_string'

    with self.subTest('after_model_validator_is_called_on_update'):
      self.assertEqual(m.computed, 'new_test_string_test')

  @parameterized.parameters(True, False)
  def test_model_base_map_pytree(self, frozen: bool):

    if frozen:

      class TestModel(model_base.BaseModelFrozen):
        x: float
        y: float

    else:

      class TestModel(model_base.BaseModelMutable):
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

    class Test1(model_base.BaseModelMutable):
      x: bool = False

    class Test2(model_base.BaseModelMutable):
      x: dict[str, Any]
      y: int
      z: list[tuple[Test1, Test1, int]]  # pytype: disable=invalid-annotation

    class Test3(model_base.BaseModelMutable):
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

    class Test1(model_base.BaseModelMutable):
      x: bool = False

    class Test2(model_base.BaseModelMutable):
      x: Test1  # pytype: disable=invalid-annotation
      y: Test1  # pytype: disable=invalid-annotation

    t1 = Test1(x=True)
    t2 = Test2(x=t1, y=t1)

    with self.assertRaisesRegex(ValueError, 'model with non-unique submodels'):
      t2.tree_build()


if __name__ == '__main__':
  absltest.main()

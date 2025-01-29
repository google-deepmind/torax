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

"""Unit tests for the `torax.config.pydantic_base` module."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pydantic
from torax.config import pydantic_base


class PydanticBaseTest(parameterized.TestCase):

  def test_numpy_array_serializer(self):
    """Tests that interpolated vars are only constructed once."""

    class TestModel(pydantic.BaseModel):
      x: pydantic_base.NumpyArray
      y: pydantic_base.NumpyArray
      z: tuple[pydantic_base.NumpyArray1D, float]

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
        pydantic_base.NumpyArray1D,
        config=pydantic.ConfigDict(arbitrary_types_allowed=True),
    )

    # Fail with 2D array.
    with self.assertRaises(ValueError):
      array.validate_python(np.array([[1.0, 2.0], [3.0, 4.0]]))

  def test_pydantic_base_frozen(self):

    class TestModel(pydantic_base.BaseFrozen):
      x: float
      y: float

    m = TestModel(y=4.0, x=2.0)

    with self.subTest('frozen_model_cannot_be_updated'):
      with self.assertRaises(ValueError):
        m.x = 2.0

  def test_pydantic_base(self):

    class Test(pydantic_base.Base, validate_assignment=True):
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


if __name__ == '__main__':
  absltest.main()

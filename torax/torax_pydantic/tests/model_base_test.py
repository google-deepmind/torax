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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
import pydantic
from torax.torax_pydantic import interpolated_param_2d
from torax.torax_pydantic import model_base


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

  def test_model_base_map_pytree(self):

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

  def test_model_set_grid(self):

    class LowerModel(model_base.BaseModelFrozen):
      x: float
      y: interpolated_param_2d.TimeVaryingArray

    class TestModel(model_base.BaseModelFrozen):
      x: int
      y: interpolated_param_2d.TimeVaryingArray
      z: LowerModel  # pytype: disable=invalid-annotation

    m = TestModel(
        x=1,
        y=interpolated_param_2d.TimeVaryingArray.model_validate(1.0),
        z=LowerModel(
            x=1.0, y=interpolated_param_2d.TimeVaryingArray.model_validate(2.0)
        ),
    )

    grid = np.array([1.0, 2.0, 3.0])
    m.set_rho_norm_grid(grid)
    # This test ensures that the grid is correctly set, and that no copies of
    # the grid are made.
    self.assertIs(m.y.rho_norm_grid, grid)
    self.assertIs(m.z.y.rho_norm_grid, grid)

    with self.subTest('cannot_set_grid_twice'):
      with self.assertRaises(RuntimeError):
        m.y.set_rho_norm_grid(grid)


if __name__ == '__main__':
  absltest.main()

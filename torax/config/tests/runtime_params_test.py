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
from torax.config import profile_conditions as profile_conditions_lib
from torax.config import runtime_params as general_runtime_params
from torax.geometry import pydantic_model as geometry_pydantic_model


# pylint: disable=invalid-name
class RuntimeParamsTest(parameterized.TestCase):

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
        "a1": -1,
        # Don't update a2, to test that it is untouched
        "a3": {
            # Don't update b1, to test that it is untouched
            "b2": -2,
            # Don't update b3, to test that it is untouched
            "b4": {
                "c1": -3,
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

  def test_runtime_params_raises_for_invalid_temp_boundary_conditions(
      self,
  ):
    """Tests that runtime params validate boundary conditions."""
    with self.assertRaises(ValueError):
      general_runtime_params.GeneralRuntimeParams(
          profile_conditions=profile_conditions_lib.ProfileConditions(
              Ti={0.0: {0.0: 12.0, 0.95: 2.0}}
          )
      )

  @parameterized.parameters(
      (
          {0.0: {0.0: 12.0, 1.0: 2.0}},
          None,
      ),  # Ti includes 1.0.
      (
          {0.0: {0.0: 12.0, 1.0: 2.0}},
          1.0,
      ),  # Both provided.
      (
          {0.0: {0.0: 12.0, 0.95: 2.0}},
          1.0,
      ),  # Ti_bound_right provided.
  )
  def test_runtime_params_constructs_with_valid_profile_conditions(
      self,
      Ti,
      Ti_bound_right,
  ):
    """Tests that runtime params validate boundary conditions."""
    general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            Ti=Ti,
            Ti_bound_right=Ti_bound_right,
        )
    )

  def test_runtime_params_make_provider(self):
    """Test that runtime params can make a provider and build dynamic params."""
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions()
    )
    torax_mesh = (
        geometry_pydantic_model.CircularConfig().build_geometry().torax_mesh
    )
    runtime_params_provider = runtime_params.make_provider(torax_mesh)
    runtime_params_provider.build_dynamic_params(0.0)


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


if __name__ == "__main__":
  absltest.main()

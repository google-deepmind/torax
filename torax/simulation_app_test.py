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

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from jax import tree_util
from torax import simulation_app


SequenceKey = tree_util.SequenceKey
GetAttrKey = tree_util.GetAttrKey
DictKey = tree_util.DictKey


class SimulationAppTest(parameterized.TestCase):

  @parameterized.parameters(
      # Three GetAttrKeys.
      dict(
          path=(SequenceKey(idx=0), GetAttrKey("currents"), GetAttrKey("I")),
          expected="I",
      ),
      # Two GetAttrKeys.
      dict(
          path=(SequenceKey(idx=0), GetAttrKey("d_face_el")),
          expected="d_face_el",
      ),
      # DictKey.
      dict(
          path=(
              SequenceKey(idx=0),
              GetAttrKey("profiles"),
              DictKey(key="source"),
          ),
          expected="source",
      ),
      # Cases for `[temp_ion, temp_el, ne, ni, psi].value`.
      dict(
          path=(
              SequenceKey(idx=0),
              GetAttrKey("temp_ion"),
              GetAttrKey("value"),
          ),
          expected="temp_ion",
      ),
      dict(
          path=(SequenceKey(idx=0), GetAttrKey("temp_el"), GetAttrKey("value")),
          expected="temp_el",
      ),
      dict(
          path=(SequenceKey(idx=0), GetAttrKey("ne"), GetAttrKey("value")),
          expected="ne",
      ),
      dict(
          path=(SequenceKey(idx=0), GetAttrKey("ni"), GetAttrKey("value")),
          expected="ni",
      ),
      dict(
          path=(SequenceKey(idx=0), GetAttrKey("psi"), GetAttrKey("value")),
          expected="psi",
      ),
      # Cases for `[temp_ion, temp_el, ne, ni, psi] boundary conditions.
      dict(
          path=(
              SequenceKey(idx=0),
              GetAttrKey("temp_ion"),
              GetAttrKey("right_face_constraint"),
          ),
          expected="temp_ion_right_bc",
      ),
      dict(
          path=(
              SequenceKey(idx=0),
              GetAttrKey("temp_el"),
              GetAttrKey("right_face_constraint"),
          ),
          expected="temp_el_right_bc",
      ),
      dict(
          path=(
              SequenceKey(idx=0),
              GetAttrKey("ne"),
              GetAttrKey("right_face_constraint"),
          ),
          expected="ne_right_bc",
      ),
      dict(
          path=(
              SequenceKey(idx=0),
              GetAttrKey("ni"),
              GetAttrKey("right_face_constraint"),
          ),
          expected="ni_right_bc",
      ),
      dict(
          path=(
              SequenceKey(idx=0),
              GetAttrKey("psi"),
              GetAttrKey("right_face_grad_constraint"),
          ),
          expected="psi_right_grad_bc",
      ),
      # Case for psi non grad boundary condition not being saved.
      dict(
          path=(
              SequenceKey(idx=0),
              GetAttrKey("psi"),
              GetAttrKey("right_face_constraint"),
          ),
          expected="right_face_constraint",
      ),
  )
  def test_path_to_name(self, path: tuple[Any, ...], expected: str):
    name = simulation_app._path_to_name(path)
    self.assertEqual(name, expected)


if __name__ == "__main__":
  absltest.main()

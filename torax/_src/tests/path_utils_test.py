# Copyright 2026 DeepMind Technologies Limited
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

import pathlib
from absl.testing import absltest
from torax._src import path_utils


class PathUtilsTest(absltest.TestCase):

  def test_torax_path(self):
    path = path_utils.torax_path()
    self.assertIsInstance(path, pathlib.Path)
    self.assertTrue(path.is_dir())
    self.assertEqual(path.parts[-1], 'torax')
    self.assertTrue((path / '_src').is_dir())
    self.assertTrue((path / '_src' / 'path_utils.py').exists())

  def test_torax_path_is_absolute(self):
    path = path_utils.torax_path()
    self.assertTrue(path.is_absolute())


if __name__ == '__main__':
  absltest.main()

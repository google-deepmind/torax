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

from absl.testing import absltest
from torax.config import config_loader


class ConfigLoaderTest(absltest.TestCase):

  def test_import_module(self):
    """test the import_module function."""
    module = config_loader.import_module(
        ".tests.test_data.test_iterhybrid_newton",
        config_package="torax",
    )
    assert hasattr(module, "CONFIG")


if __name__ == "__main__":
  absltest.main()

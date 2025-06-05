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
import pathlib
from torax._src.config import config_loader

_SCRIPT_DIR = pathlib.Path(__file__).parent.parent
_COMBINED_TRANSPORT_EXAMPLE_SCRIPT_PATH = (
    _SCRIPT_DIR / "combined_transport_example.py"
)

_combined_transport_example_module = config_loader.import_module(
    _COMBINED_TRANSPORT_EXAMPLE_SCRIPT_PATH
)


class CombinedTransportExampleTest(absltest.TestCase):

  def test_combined_transport_example_runs_without_errors(self):
    _combined_transport_example_module["main"]([])


if __name__ == "__main__":
  absltest.main()

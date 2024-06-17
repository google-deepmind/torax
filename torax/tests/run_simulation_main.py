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

"""Unit tests for torax.run_simulation_main."""

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
from torax import run_simulation_main


class RunSimulationMainTest(parameterized.TestCase):
  """Unit tests for the `torax.run_simulation_main` app."""
  # These tests will make extensive use of access to private members of the
  # run_simulation_main module.
  # pylint: disable=protected-access

  @flagsaver.flagsaver((run_simulation_main._PYTHON_CONFIG_PACKAGE, "torax"))
  def test_import_module(self):
    """docstring."""
    module = run_simulation_main._import_module(
        ".tests.test_data.test_iterhybrid_newton"
    )
    assert hasattr(module, "CONFIG")


if __name__ == "__main__":
  absltest.main()

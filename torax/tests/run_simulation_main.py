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

from unittest import mock
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

  @mock.patch("builtins.input", side_effect=["r"])
  def test_prompt_user_good_input(self, mock_input):
    """Test that prompt_user accepts the 'r' command."""
    del mock_input  # Needed for @patch interface but not used in this test
    config_str = ".tests.test_data.test_implicit"
    # The @patch decorator overrides the `input` function so that when
    # `prompt_user` calls `input`, it will receive an "r". That is a valid
    # command and it should return it.
    user_command = run_simulation_main.prompt_user(config_str)
    self.assertEqual(user_command, run_simulation_main._UserCommand.RUN)

  @mock.patch("builtins.input", side_effect=["invalid", "q"])
  def test_prompt_user_bad_input(self, mock_input):
    """Test that prompt_user rejects invalid input."""
    del mock_input  # Needed for @patch interface but not used in this test
    config_str = ".tests.test_data.test_implicit"
    # The @patch decorator overrides the `input` function so that when
    # `prompt_user` calls `input`, it will receive "invalid". That is not
    # a valid command so it should be rejected. The `prompt_user` function
    # re-prompts forever until a valid input is received so we next send
    # a valid "q" for quit.
    user_command = run_simulation_main.prompt_user(config_str)
    self.assertEqual(user_command, run_simulation_main._UserCommand.QUIT)


if __name__ == "__main__":
  absltest.main()

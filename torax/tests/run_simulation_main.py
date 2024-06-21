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

import io
from unittest import mock
from absl import app
from absl import logging
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
    """test the import_module function."""
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

  @flagsaver.flagsaver(
      (run_simulation_main._PYTHON_CONFIG_PACKAGE, "torax"),
      (
          run_simulation_main._PYTHON_CONFIG_MODULE,
          ".tests.test_data.test_implicit",
      ),
  )
  @mock.patch("builtins.input", side_effect=["q"])
  def test_main_app(self, mock_input):
    """Test that main app runs without errors."""
    del mock_input  # Needed for @patch interface but not used in this test

    # In this test we run the whole app. We send a mocked 'q' input to quit the
    # app after it is done running. The app quits an explicit SystemExit that
    # we need to catch to avoid bringing down the tests. To make sure that the
    # app really ran as expected, we capture stdout and check that it contains
    # an expected message from after the successful execution.

    captured_stdout = io.StringIO()
    handler = logging.PythonHandler(captured_stdout)

    try:
      logging.get_absl_logger().addHandler(handler)
      with self.assertRaises(SystemExit):
        app.run(run_simulation_main.main)
    finally:
      logging.get_absl_logger().removeHandler(handler)
    self.assertIn("Wrote simulation output to", captured_stdout.getvalue())


if __name__ == "__main__":
  absltest.main()

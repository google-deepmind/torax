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
import os
import shutil
import sys
from unittest import mock
from absl import app
from absl import logging
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
from torax import run_simulation_main
from torax import simulation_app
from torax.tests.test_lib import paths


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
  def test_main_app_runs(self, mock_input):
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
      with self.assertRaises(SystemExit) as cm:
        app.run(run_simulation_main.main)
      # Make sure the app ran successfully
      self.assertIsNone(cm.exception.code)
    finally:
      logging.get_absl_logger().removeHandler(handler)
    self.assertIn("Wrote simulation output to", captured_stdout.getvalue())

  @flagsaver.flagsaver(
      (run_simulation_main._PYTHON_CONFIG_PACKAGE, "temp_package"),
      (
          run_simulation_main._PYTHON_CONFIG_MODULE,
          ".test_changing_config",
      ),
  )
  def test_main_app_cc(self):
    """Test that the main app successfully changes the config."""

    # Read only source directory containing the tests
    test_data_dir = paths.test_data_dir()
    # Editable directory we add to Python path to contain our
    # changing config script
    editable_dir = self.create_tempdir("editable_dir")
    editable_dir_path = editable_dir.full_path
    print("editable_dir:", editable_dir_path)
    sys.path.append(editable_dir_path)
    # Create an importable package
    package = os.path.join(editable_dir, "temp_package")
    os.mkdir(package)
    init = os.path.join(package, "__init__.py")
    with open(init, "w") as f:
      f.write("""
""")
    # Path of the "before" config we will run first
    before = os.path.join(test_data_dir, "test_changing_config_before.py")
    # Path that the script reads configs from
    in_use = os.path.join(package, "test_changing_config.py")
    # Path of the "after" config we will run second
    after = os.path.join(test_data_dir, "test_changing_config_after.py")
    # Copy the "before" config to the active location
    shutil.copy(before, in_use)
    os.sync()

    # Redirect stdout to this string buffer
    captured_stdout = io.StringIO()
    handler = logging.PythonHandler(captured_stdout)

    call_count = 0
    filepaths = []

    def mock_input(prompt):
      """Overrides the system `input` function to simulate user input."""
      nonlocal call_count

      logging.error("Calling mock_input")

      if call_count == 0:
        self.assertEqual(prompt, run_simulation_main.CHOICE_PROMPT)
        # The first run on the simulation just completed. Fetch its output.
        filepaths.append(get_latest_filepath(captured_stdout))
        # After the first run of the main program, we copy over the
        # changed config, then send the 'cc' response
        os.remove(in_use)
        shutil.copy(after, in_use)
        os.sync()
        response = "cc"
      elif call_count == 1:
        self.assertEqual(prompt, run_simulation_main.Y_N_PROMPT)
        # The second call to `input` is confirming that we should run with
        # this config.
        response = "y"
      else:
        # The second run on the simulation just completed. Fetch its output.
        filepaths.append(get_latest_filepath(captured_stdout))
        self.assertEqual(prompt, run_simulation_main.CHOICE_PROMPT)
        # After the second run, we quit
        response = "q"

      call_count += 1
      return response

    # Run the app with the modified `input` and stdout
    with mock.patch("builtins.input", side_effect=mock_input):
      try:
        logging.get_absl_logger().addHandler(handler)
        with self.assertRaises(SystemExit) as cm:
          app.run(run_simulation_main.main)
        # Make sure the app ran successfully
        self.assertIsNone(cm.exception.code)
      finally:
        logging.get_absl_logger().removeHandler(handler)


def get_latest_filepath(stream: io.StringIO) -> str:
  """Returns the last filepath written to by the app."""
  value = stream.getvalue()
  assert simulation_app.WRITE_PREFIX in value
  chunks = value.split(simulation_app.WRITE_PREFIX)
  last = chunks[-1]
  suffix = ".nc"
  end = last.index(suffix)
  assert end != -1
  end += len(suffix)
  path = last[:end]
  return path


if __name__ == "__main__":
  absltest.main()

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

r"""Main entrypoint for running transport simulation.

Example command with a configuration defined in Python:
python3 run_simulation_main.py \
 --config='torax.tests.test_data.default_config' \
 --log_progress
"""

from collections.abc import Sequence
import enum
import functools
import time

from absl import app
from absl import flags
from absl import logging
import jax
import torax
from torax import simulation_app
from torax.config import config_loader
from torax.plotting import plotruns_lib
from torax.torax_pydantic import model_config

# Absorb pytest’s “--rootdir” flag so absl doesn’t fatally bail under pytest.
flags.DEFINE_string('rootdir', None, 'Ignored pytest rootdir flag.')

# String used when prompting the user to make a choice of command
CHOICE_PROMPT = 'Your choice: '
# String used when prompting the user to make a yes / no choice
Y_N_PROMPT = 'y/n: '
# String used when printing how long the simulation took
SIMULATION_TIME = 'simulation time'

_PYTHON_CONFIG_MODULE = flags.DEFINE_string(
    'config',
    None,
    'Module from which to import a python-based config. This program expects a '
    '`get_sim()` function to be implemented in this module. Can either be '
    'an absolute or relative path. See importlib.import_module() for more '
    'information on how to use this flag and --config_package.',
)

_PYTHON_CONFIG_PACKAGE = flags.DEFINE_string(
    'config_package',
    None,
    'If provided, it is the base package the --config is imported from. '
    'This is required if --config is a relative path.',
)

_LOG_SIM_PROGRESS = flags.DEFINE_bool(
    'log_progress',
    False,
    'If true, logs the time of each timestep as the simulation runs.',
)

_PLOT_SIM_PROGRESS = flags.DEFINE_bool(
    'plot_progress',
    False,
    'If true, plots the time of each timestep as the simulation runs.'
    ' Note: this is temporarily disabled.',
)

_LOG_SIM_OUTPUT = flags.DEFINE_bool(
    'log_output',
    False,
    'If True, logs extra information to stdout/stderr.',
)

_REFERENCE_RUN = flags.DEFINE_string(
    'reference_run',
    None,
    'If provided, after the simulation is run, we can compare the last run to'
    ' the reference run.',
)

_USE_JAX_PROFILER = flags.DEFINE_bool(
    'use_jax_profiler',
    False,
    'If True, use Jax profiler. Processing will stop and a URL to visualise'
    ' results will be generated. Processing will resume when the URL is opened.'
    'See https://jax.readthedocs.io/en/latest/profiling.html.',
)

_QUIT = flags.DEFINE_bool(
    'quit',
    False,
    'If True, quits after the first operation (no interactive mode).',
)

_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    None,
    'If provided, overrides the default output directory.',
)

_PLOT_CONFIG_MODULE = flags.DEFINE_string(
    'plot_config',
    'torax.plotting.configs.default_plot_config',  # Default
    'Module path to the plot config.',
)

jax.config.parse_flags_with_absl()


@enum.unique
class _UserCommand(enum.Enum):
  """Options to do on every iteration of the script."""

  RUN = ('RUN SIMULATION', 'r', simulation_app.AnsiColors.GREEN)
  MODIFY_CONFIG = ('modify the existing config and reload it', 'mc')
  CHANGE_CONFIG = ('provide a new config file to load', 'cc')
  TOGGLE_LOG_SIM_PROGRESS = ('toggle --log_progress', 'tlp')
  TOGGLE_PLOT_SIM_PROGRESS = ('toggle --plot_progress', 'tpp')
  TOGGLE_LOG_SIM_OUTPUT = ('toggle --log_output', 'tlo')
  PLOT_RUN = ('plot previous run(s) or against reference if provided', 'pr')
  QUIT = ('quit', 'q', simulation_app.AnsiColors.RED)


def _prompt_user(config_module_str: str) -> _UserCommand:
  """Prompts the user for the next thing to do."""
  simulation_app.log_to_stdout('\n')
  simulation_app.log_to_stdout(
      f'Using the config: {config_module_str}',
      color=simulation_app.AnsiColors.YELLOW,
  )
  user_command = None
  simulation_app.log_to_stdout('\n')
  while user_command is None:
    simulation_app.log_to_stdout(
        'What would you like to do next?',
        color=simulation_app.AnsiColors.BLUE,
    )
    for uc in _UserCommand:
      if len(uc.value) == 3:
        color = uc.value[2]
      else:
        color = simulation_app.AnsiColors.BLUE
      simulation_app.log_to_stdout(f'{uc.value[1]}: {uc.value[0]}', color)
    input_text = input(CHOICE_PROMPT)
    text_to_uc = {uc.value[1]: uc for uc in _UserCommand}
    input_text = input_text.lower().strip()
    if input_text in text_to_uc:
      user_command = text_to_uc[input_text]
    else:
      simulation_app.log_to_stdout(
          'Unrecognized input. Try again.',
          color=simulation_app.AnsiColors.YELLOW,
      )
  return user_command


def _maybe_update_config_module(
    config_module_str: str,
) -> str:
  """Returns a possibly-updated config module to import."""
  simulation_app.log_to_stdout(
      f'Existing module: {config_module_str}',
      color=simulation_app.AnsiColors.BLUE,
  )
  simulation_app.log_to_stdout(
      'Would you like to change which file to import?',
      color=simulation_app.AnsiColors.BLUE,
  )
  should_change = _get_yes_or_no()
  if should_change:
    logging.info('Updating the module.')
    return input('Enter the new module to use: ').strip()
  else:
    logging.info('Continuing with %s', config_module_str)
    return config_module_str


def _modify_config(
    config_module_str: str,
) -> tuple[model_config.ToraxConfig, str] | None:
  """Returns a new ToraxConfig from the modified config module."""
  simulation_app.log_to_stdout(
      f'Change {config_module_str} to include new values.',
      color=simulation_app.AnsiColors.BLUE,
  )
  simulation_app.log_to_stdout(
      'Modify the config with new values, then enter "y".',
      color=simulation_app.AnsiColors.BLUE,
  )
  simulation_app.log_to_stdout(
      'Enter "n" to go back to the main menu without loading any changes.',
      color=simulation_app.AnsiColors.BLUE,
  )
  proceed_with_run = _get_yes_or_no()
  if not proceed_with_run:
    return None
  config_module = config_loader.import_module(
      config_module_str, _PYTHON_CONFIG_PACKAGE.value
  )
  if not hasattr(config_module, 'CONFIG'):
    raise ValueError(
        f'Config module {config_module_str} does not have a CONFIG attribute.'
        ' Please use the basic config dict to build Sim.'
    )
  return model_config.ToraxConfig.from_dict(config_module.CONFIG)


def _change_config(
    config_module_str: str,
) -> tuple[model_config.ToraxConfig, str]:
  """Returns a ToraxConfig from the new config module and the config file."""
  config_module_str = _maybe_update_config_module(config_module_str)
  simulation_app.log_to_stdout(
      f'Change {config_module_str} to include new values. Any changes to '
      'CONFIG will be picked up.',
      color=simulation_app.AnsiColors.BLUE,
  )
  input('Press Enter when done changing the module.')
  config_module = config_loader.import_module(
      config_module_str, _PYTHON_CONFIG_PACKAGE.value
  )
  if not hasattr(config_module, 'CONFIG'):
    raise ValueError(
        f'Config module {config_module_str} does not have a CONFIG attribute.'
        ' Please use the basic config dict to build Sim.'
    )
  torax_config = model_config.ToraxConfig.from_dict(config_module.CONFIG)
  return torax_config, config_module_str


def _get_yes_or_no() -> bool:
  """Returns a boolean indicating yes depending on user input."""
  input_text = None
  while input_text is None:
    input_text = input(Y_N_PROMPT)
    input_text = input_text.lower().strip()
    if input_text not in ('y', 'n'):
      simulation_app.log_to_stdout(
          'Unrecognized input. Try again.',
          color=simulation_app.AnsiColors.YELLOW,
      )
      input_text = None
    else:
      return input_text == 'y'


def _toggle_log_progress(log_sim_progress: bool) -> bool:
  """Toggles the --log_progress flag."""
  log_sim_progress = not log_sim_progress
  simulation_app.log_to_stdout(
      f'--log_progress is now {log_sim_progress}.',
      color=simulation_app.AnsiColors.GREEN,
  )
  if log_sim_progress:
    simulation_app.log_to_stdout(
        'Each time step will be logged to stdout.',
        color=simulation_app.AnsiColors.GREEN,
    )
  else:
    simulation_app.log_to_stdout(
        'No time steps will be logged to stdout.',
        color=simulation_app.AnsiColors.GREEN,
    )
  return log_sim_progress


def _toggle_plot_progress(plot_sim_progress: bool) -> bool:
  """Toggles the --plot_progress flag."""
  plot_sim_progress = not plot_sim_progress
  simulation_app.log_to_stdout(
      f'--plot_progress is now {plot_sim_progress}.',
      color=simulation_app.AnsiColors.GREEN,
  )
  if plot_sim_progress:
    simulation_app.log_to_stdout(
        'Each time step will be shown in a plot.',
        color=simulation_app.AnsiColors.GREEN,
    )
  else:
    simulation_app.log_to_stdout(
        'No plots will be shown.',
        color=simulation_app.AnsiColors.GREEN,
    )
  return plot_sim_progress


def _toggle_log_output(log_sim_output: bool) -> bool:
  """Toggles the --log_output flag."""
  log_sim_output = not log_sim_output
  simulation_app.log_to_stdout(
      f'--log_output is now {log_sim_output}.',
      color=simulation_app.AnsiColors.GREEN,
  )
  if log_sim_output:
    simulation_app.log_to_stdout(
        'Extra simulation info will be logged to stdout at the end of the run.',
        color=simulation_app.AnsiColors.GREEN,
    )
  else:
    simulation_app.log_to_stdout(
        'No extra information will be logged to stdout.',
        color=simulation_app.AnsiColors.GREEN,
    )
  return log_sim_output


def _post_run_plotting(
    output_files: Sequence[str],
) -> None:
  """Helper to produce plots after a simulation run."""
  input_text = input(
      'Plot the last run (0), the last two runs (1), the last run against a'
      ' reference run (2): '
  )
  if not output_files:
    simulation_app.log_to_stdout(
        'No output files found, skipping plotting.',
        color=simulation_app.AnsiColors.RED,
    )
    return
  try:
    plot_config = config_loader.import_module(
        _PLOT_CONFIG_MODULE.value
    ).PLOT_CONFIG
  except (ModuleNotFoundError, AttributeError) as e:
    logging.exception(
        'Error loading plot config module %s: %s', _PLOT_CONFIG_MODULE.value, e
    )
    return
  match input_text:
    case '0':
      return plotruns_lib.plot_run(plot_config, output_files[-1])
    case '1':
      if len(output_files) == 1:
        simulation_app.log_to_stdout(
            'Only one output run file found, only plotting the last run.',
            color=simulation_app.AnsiColors.RED,
        )
        return plotruns_lib.plot_run(plot_config, output_files[-1])
      return plotruns_lib.plot_run(
          plot_config, output_files[-1], output_files[-2]
      )
    case '2':
      reference_run = _REFERENCE_RUN.value
      if reference_run is None:
        simulation_app.log_to_stdout(
            'No reference run provided, only plotting the last run.',
            color=simulation_app.AnsiColors.RED,
        )
      return plotruns_lib.plot_run(plot_config, output_files[-1], reference_run)
    case _:
      raise ValueError('Unknown command')


def main(_):
  torax.set_jax_precision()
  config_module_str = _PYTHON_CONFIG_MODULE.value
  if config_module_str is None:
    raise ValueError(f'--{_PYTHON_CONFIG_MODULE.name} must be specified.')
  log_sim_progress = _LOG_SIM_PROGRESS.value
  plot_sim_progress = _PLOT_SIM_PROGRESS.value
  log_sim_output = _LOG_SIM_OUTPUT.value
  torax_config = None
  output_files = []
  output_dir = _OUTPUT_DIR.value
  try:
    start_time = time.time()
    torax_config = config_loader.build_torax_config_from_config_module(
        config_module_str, _PYTHON_CONFIG_PACKAGE.value
    )
    build_time = time.time() - start_time
    start_time = time.time()
    output_file = _call_sim_app_main(
        torax_config,
        output_dir=output_dir,
        log_sim_progress=log_sim_progress,
        plot_sim_progress=plot_sim_progress,
        log_sim_output=log_sim_output,
    )
    simulation_time = time.time() - start_time
    output_files.append(output_file)
    simulation_app.log_to_stdout(
        f'Sim and params build time: {build_time:.2f}s, {SIMULATION_TIME}:'
        f' {simulation_time:.2f}s',
        color=simulation_app.AnsiColors.GREEN,
    )
  except ValueError as ve:
    simulation_app.log_to_stdout(
        f'Error occurred: {ve}',
        color=simulation_app.AnsiColors.RED,
        exc_info=True,
    )
    simulation_app.log_to_stdout(
        'Not running sim. Update config and try again.',
        color=simulation_app.AnsiColors.RED,
    )
  if _QUIT.value:
    return
  user_command = _prompt_user(config_module_str)
  while user_command != _UserCommand.QUIT:
    match user_command:
      case _UserCommand.QUIT:
        # This line shouldn't get hit, but is here for pytype.
        return  # Exit the function.
      case _UserCommand.RUN:
        if torax_config is None:
          simulation_app.log_to_stdout(
              'Need to reload the simulation.',
              color=simulation_app.AnsiColors.RED,
          )
        else:
          start_time = time.time()
          output_file = _call_sim_app_main(
              torax_config,
              output_dir=output_dir,
              log_sim_progress=log_sim_progress,
              plot_sim_progress=plot_sim_progress,
              log_sim_output=log_sim_output,
          )
          output_files.append(output_file)
          simulation_time = time.time() - start_time
          simulation_app.log_to_stdout(
              f'Simulation time: {simulation_time:.2f}s'
          )
      case _UserCommand.MODIFY_CONFIG:
        # See docstring for detailed info on what recompiles.
        if torax_config is None:
          simulation_app.log_to_stdout(
              'Need to reload the simulation.',
              color=simulation_app.AnsiColors.RED,
          )
        else:
          try:
            start_time = time.time()
            torax_config_or_none = _modify_config(config_module_str)
            if torax_config_or_none is not None:
              torax_config = torax_config_or_none
            config_change_time = time.time() - start_time
            simulation_app.log_to_stdout(
                f'Config change time: {config_change_time:.2f}s'
            )
          except ValueError as ve:
            simulation_app.log_to_stdout(
                f'Error occurred: {ve}',
                color=simulation_app.AnsiColors.RED,
            )
            simulation_app.log_to_stdout(
                'Update config and try again.',
                color=simulation_app.AnsiColors.RED,
            )
      case _UserCommand.CHANGE_CONFIG:
        # See docstring for detailed info on what recompiles.
        if torax_config is None:
          simulation_app.log_to_stdout(
              'Need to reload the simulation.',
              color=simulation_app.AnsiColors.RED,
          )
        else:
          try:
            start_time = time.time()
            torax_config_or_none = _change_config(config_module_str)
            if torax_config_or_none is not None:
              torax_config, config_module_str = torax_config_or_none
            config_change_time = time.time() - start_time
            simulation_app.log_to_stdout(
                f'Config change time: {config_change_time:.2f}s'
            )
          except ValueError as ve:
            simulation_app.log_to_stdout(
                f'Error occurred: {ve}',
                color=simulation_app.AnsiColors.RED,
            )
            simulation_app.log_to_stdout(
                'Update config and try again.',
                color=simulation_app.AnsiColors.RED,
            )
      case _UserCommand.TOGGLE_LOG_SIM_PROGRESS:
        log_sim_progress = _toggle_log_progress(log_sim_progress)
      case _UserCommand.TOGGLE_PLOT_SIM_PROGRESS:
        plot_sim_progress = _toggle_plot_progress(plot_sim_progress)
      case _UserCommand.TOGGLE_LOG_SIM_OUTPUT:
        log_sim_output = _toggle_log_output(log_sim_output)
      case _UserCommand.PLOT_RUN:
        _post_run_plotting(output_files)
      case _:
        raise ValueError('Unknown command')
    user_command = _prompt_user(config_module_str)


def use_jax_profiler_if_enabled(f):
  """Decorator that runs `func` with profiling if the flag is enabled."""

  @functools.wraps(f)
  def decorated(*args, **kwargs):
    if _USE_JAX_PROFILER.value:
      with jax.profiler.trace('/tmp/torax-jax-trace'):
        result = f(*args, **kwargs)
        simulation_app.log_to_stdout(
            'Profiling: to display results go to'
            ' https://ui.perfetto.dev/#!/viewer and open the trace file shown'
            ' below.'
        )
        return result
    else:
      return f(*args, **kwargs)

  return decorated


@use_jax_profiler_if_enabled
def _call_sim_app_main(
    torax_config,
    output_dir: str | None,
    log_sim_progress: bool,
    plot_sim_progress: bool,
    log_sim_output: bool,
):
  return simulation_app.main(
      lambda: torax_config,
      output_dir=output_dir,
      log_sim_progress=log_sim_progress,
      plot_sim_progress=plot_sim_progress,
      log_sim_output=log_sim_output,
  )


# Method used by the `run_torax` binary.
def run():
  app.run(main)


if __name__ == '__main__':
  run()

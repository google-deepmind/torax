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
 --python_config='torax.tests.test_data.default_config' \
 --log_progress
"""

import enum
import importlib

from absl import app
from absl import flags
from absl import logging
import jax
import torax
from torax import simulation_app


_PYTHON_CONFIG_MODULE = flags.DEFINE_string(
    'python_config',
    None,
    'Module from which to import a python-based config. This program expects a '
    '`get_sim()` function to be implemented in this module. Can either be '
    'an absolute or relative path. See importlib.import_module() for more '
    'information on how to use this flag and --python_config_package.',
)

_PYTHON_CONFIG_PACKAGE = flags.DEFINE_string(
    'python_config_package',
    None,
    'If provided, it is the base package the --python_config is imported from. '
    'This is required if --python_config is a relative path.',
)

_LOG_SIM_PROGRESS = flags.DEFINE_bool(
    'log_progress',
    False,
    'If true, logs the time of each timestep as the simulation runs.',
)

_PLOT_SIM_PROGRESS = flags.DEFINE_bool(
    'plot_progress',
    False,
    'If true, plots the time of each timestep as the simulation runs.',
)

_LOG_SIM_OUTPUT = flags.DEFINE_bool(
    'log_output',
    False,
    'In True, logs extra information to stdout/stderr.',
)

jax.config.parse_flags_with_absl()


@enum.unique
class _UserCommand(enum.Enum):
  """Options to do on every iteration of the script."""

  QUIT = ('quit', 'q')
  RUN = ('run the simulation', 'r')
  CHANGE_CONFIG = (
      'change config for the same sim object (may recompile)',
      'cc',
  )
  CHANGE_SIM_OBJ = (
      'change config and build new sim object (will recompile)',
      'cs',
  )
  TOGGLE_LOG_SIM_PROGRESS = ('toggle --log_progress', 'tlp')
  TOGGLE_PLOT_SIM_PROGRESS = ('toggle --plot_progress', 'tpp')
  TOGGLE_LOG_SIM_OUTPUT = ('toggle --log_output', 'tlo')


# Tracks all the modules imported so far. Maps the name to the module object.
_ALL_MODULES = {}


def _import_module(module_name: str):
  if module_name in _ALL_MODULES:
    return importlib.reload(_ALL_MODULES[module_name])
  else:
    module = importlib.import_module(module_name, _PYTHON_CONFIG_PACKAGE.value)
    _ALL_MODULES[module_name] = module
    return module


def _get_config_module(
    config_module_str: str | None = None,
):
  config_module_str = config_module_str or _PYTHON_CONFIG_MODULE.value
  return _import_module(config_module_str), config_module_str


def prompt_user(config_module_str: str) -> _UserCommand:
  """Prompts the user for the next thing to do."""
  simulation_app.log_to_stdout(
      f'Running with the config: {config_module_str}',
      color=simulation_app.AnsiColors.YELLOW,
  )
  user_command = None
  while user_command is None:
    simulation_app.log_to_stdout(
        'What would you like to do next?',
        color=simulation_app.AnsiColors.BLUE,
    )
    for uc in _UserCommand:
      simulation_app.log_to_stdout(
          f'{uc.value[1]}: {uc.value[0]}',
          color=simulation_app.AnsiColors.BLUE,
      )
    input_text = input('Your choice: ')
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


def maybe_update_config_module(
    config_module_str: str,
) -> str:
  """Returns a possibly-updated config module to import."""
  simulation_app.log_to_stdout(
      f'Existing module: {config_module_str}',
      color=simulation_app.AnsiColors.BLUE,
  )
  input_text = None
  while input_text is None:
    simulation_app.log_to_stdout(
        'Would you like to change which file to import?',
        color=simulation_app.AnsiColors.BLUE,
    )
    input_text = input('y/n: ')
    input_text = input_text.lower().strip()
    if input_text not in ('y', 'n'):
      simulation_app.log_to_stdout(
          'Unrecognized input. Try again.',
          color=simulation_app.AnsiColors.YELLOW,
      )
      input_text = None
  if input_text == 'y':
    logging.info('Updating the module.')
    return input('Enter the new module to use: ').strip()
  return config_module_str


def change_config(
    sim: torax.Sim,
    config_module_str: str,
) -> tuple[torax.Sim, torax.Config, str]:
  """Returns a new Sim with the updated config but same SimulationStepFn.

  This function gives the user a chance to reuse the SimulationStepFn without
  triggering a recompile. The SimulationStepFn will only recompile if the
  StaticConfigSlice derived from the the new Config changes.

  This function will NOT change the stepper, transport model, or time step
  calculator built into the SimulationStepFn. To change these attributes, the
  user must build a new Sim object.

  Args:
    sim: Sim object used in the previous run.
    config_module_str: Config module used previously. User will have the
      opportunity to update which module to load.

  Returns:
    Tuple with:
     - New Sim object with new config.
     - New Config object with modified configuration attributes
     - Name of the module used to load the config.
  """
  config_module_str = maybe_update_config_module(config_module_str)
  simulation_app.log_to_stdout(
      f'Change {config_module_str} to include new values. Only changes to '
      'get_config() will be picked up.',
      color=simulation_app.AnsiColors.BLUE,
  )
  input('Press Enter when ready.')
  config_module, _ = _get_config_module(config_module_str)
  new_config = config_module.get_config()
  new_geo = config_module.get_geometry(new_config)
  new_transport_model = config_module.get_transport_model()
  source_models = config_module.get_sources()
  new_source_params = {
      name: source.runtime_params
      for name, source in source_models.sources.items()
  }
  # Make sure the transport model has not changed.
  # TODO(b/330172917): Improve the check for updated configs.
  if not isinstance(new_transport_model, type(sim.transport_model)):
    raise ValueError(
        f'New transport model type {type(new_transport_model)} does not match'
        f' the existing transport model {type(sim.transport_model)}. When using'
        ' this option, you cannot change the transport model.'
    )
  sim = simulation_app.update_sim(
      sim=sim,
      config=new_config,
      geo=new_geo,
      transport_runtime_params=new_transport_model.runtime_params,
      source_runtime_params=new_source_params,
  )
  return sim, new_config, config_module_str


def change_sim_obj(
    config_module_str: str,
) -> tuple[torax.Sim, torax.Config, str]:
  """Builds a new Sim from the config module.

  Unlike change_config(), this function builds a brand new Sim object with a
  new transport model, stepper, time step calculator, and so on. It will always
  recompile (unless requested not to).

  Args:
    config_module_str: Config module used previously. User will have the
      opportunity to update which module to load.

  Returns:
    Tuple with:
     - New Sim object with a new config.
     - New Config object with modified config attributes
     - Name of the module used to load the config.
  """
  config_module_str = maybe_update_config_module(config_module_str)
  simulation_app.log_to_stdout(
      f'Change {config_module_str} to include new values. Any changes to '
      'get_sim() will be picked up.',
      color=simulation_app.AnsiColors.BLUE,
  )
  input('Press Enter when done changing the module.')
  config_module, _ = _get_config_module(config_module_str)
  new_config = config_module.get_config()
  sim = config_module.get_sim()
  return sim, new_config, config_module_str


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


def main(_):
  config_module, config_module_str = _get_config_module()
  new_config = config_module.get_config()
  sim = config_module.get_sim()
  log_sim_progress = _LOG_SIM_PROGRESS.value
  plot_sim_progress = _PLOT_SIM_PROGRESS.value
  log_sim_output = _LOG_SIM_OUTPUT.value
  simulation_app.main(
      lambda: sim,
      output_dir=new_config.output_dir,
      log_sim_progress=log_sim_progress,
      plot_sim_progress=plot_sim_progress,
      log_sim_output=log_sim_output,
  )
  user_command = prompt_user(config_module_str)
  while user_command != _UserCommand.QUIT:
    match user_command:
      case _UserCommand.QUIT:
        # This line shouldn't get hit, but is here for pytype.
        return  # Exit the function.
      case _UserCommand.RUN:
        simulation_app.main(
            lambda: sim,
            output_dir=new_config.output_dir,
            log_sim_progress=log_sim_progress,
            plot_sim_progress=plot_sim_progress,
            log_sim_output=log_sim_output,
        )
      case _UserCommand.CHANGE_CONFIG:
        # See docstring for detailed info on what recompiles.
        sim, new_config, config_module_str = change_config(
            sim, config_module_str
        )
      case _UserCommand.CHANGE_SIM_OBJ:
        # This always builds a new object and requires recompilation.
        sim, new_config, config_module_str = change_sim_obj(config_module_str)
      case _UserCommand.TOGGLE_LOG_SIM_PROGRESS:
        log_sim_progress = _toggle_log_progress(log_sim_progress)
      case _UserCommand.TOGGLE_PLOT_SIM_PROGRESS:
        plot_sim_progress = _toggle_plot_progress(plot_sim_progress)
      case _UserCommand.TOGGLE_LOG_SIM_OUTPUT:
        log_sim_output = _toggle_log_output(log_sim_output)
      case _:
        raise ValueError('Unknown command')
    user_command = prompt_user(config_module_str)


if __name__ == '__main__':
  app.run(main)

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

import enum
import importlib

from absl import app
from absl import flags
from absl import logging
import jax
import torax
from torax import simulation_app
from torax.config import build_sim


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

  RUN = ('RUN SIMULATION', 'r', simulation_app.AnsiColors.GREEN)
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
  QUIT = ('quit', 'q', simulation_app.AnsiColors.RED)


# Tracks all the modules imported so far. Maps the name to the module object.
_ALL_MODULES = {}


def _import_module(module_name: str):
  """Imports a module."""
  try:
    if module_name in _ALL_MODULES:
      return importlib.reload(_ALL_MODULES[module_name])
    else:
      module = importlib.import_module(
          module_name, _PYTHON_CONFIG_PACKAGE.value
      )
      _ALL_MODULES[module_name] = module
      return module
  except Exception as e:
    simulation_app.log_to_stdout(f'Exception raised: {e}')
    raise ValueError('Exception while importing.') from e


def prompt_user(config_module_str: str) -> _UserCommand:
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


def change_config(
    sim: torax.Sim,
    config_module_str: str,
) -> tuple[torax.Sim, torax.GeneralRuntimeParams] | None:
  """Returns a new Sim with the updated config but same SimulationStepFn.

  This function gives the user a chance to reuse the SimulationStepFn without
  triggering a recompile. The SimulationStepFn will only recompile if the
  StaticConfigSlice derived from the the new Config changes.

  This function will NOT change the stepper, transport model, or time step
  calculator built into the SimulationStepFn. To change these attributes, the
  user must build a new Sim object.

  Args:
    sim: Sim object used in the previous run.
    config_module_str: Config module being used.

  Returns:
    Tuple with:
     - New Sim object with new config.
     - New Config object with modified configuration attributes
  """
  simulation_app.log_to_stdout(
      f'Change {config_module_str} to include new values.',
      color=simulation_app.AnsiColors.BLUE,
  )
  yellow = simulation_app.AnsiColors.YELLOW
  simulation_app.log_to_stdout('You cannot change the following:', color=yellow)
  simulation_app.log_to_stdout('  - stepper type', color=yellow)
  simulation_app.log_to_stdout('  - transport model type', color=yellow)
  simulation_app.log_to_stdout('  - source types', color=yellow)
  simulation_app.log_to_stdout('  - time step calculator', color=yellow)
  simulation_app.log_to_stdout(
      'To change these parameters, select "cs" from the main menu.',
      color=yellow,
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
  config_module = _import_module(config_module_str)
  if hasattr(config_module, 'CONFIG'):
    # Assume that the config module uses the basic config dict to build Sim.
    sim_config = config_module.CONFIG
    new_runtime_params = build_sim.build_runtime_params_from_config(
        sim_config['runtime_params']
    )
    new_geo = build_sim.build_geometry_from_config(
        sim_config['geometry'], new_runtime_params
    )
    new_transport_model = build_sim.build_transport_model_from_config(
        sim_config['transport']
    )
    source_models = build_sim.build_sources_from_config(sim_config['sources'])
    new_stepper_builder = build_sim.build_stepper_builder_from_config(
        sim_config['stepper']
    )
  else:
    # Assume the config module has several methods to define the individual Sim
    # attributes (the "advanced", more Python-forward configuration method).
    new_runtime_params = config_module.get_runtime_params()
    new_geo = config_module.get_geometry(new_runtime_params)
    new_transport_model = config_module.get_transport_model()
    source_models = config_module.get_sources()
    new_stepper_builder = config_module.get_stepper_builder()
  new_source_params = {
      name: source.runtime_params
      for name, source in source_models.sources.items()
  }
  # We pass the getter so that any changes to the objects runtime params after
  # the Sim object is created are picked up.
  stepper_params_getter = lambda: new_stepper_builder.runtime_params
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
      runtime_params=new_runtime_params,
      geo=new_geo,
      transport_runtime_params=new_transport_model.runtime_params,
      source_runtime_params=new_source_params,
      stepper_runtime_params_getter=stepper_params_getter,
  )
  return sim, new_runtime_params


def change_sim_obj(
    config_module_str: str,
) -> tuple[torax.Sim, torax.GeneralRuntimeParams, str]:
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
      'CONFIG or get_sim() will be picked up.',
      color=simulation_app.AnsiColors.BLUE,
  )
  input('Press Enter when done changing the module.')
  sim, new_runtime_params = _build_sim_and_runtime_params_from_config_module(
      config_module_str
  )
  return sim, new_runtime_params, config_module_str


def _build_sim_and_runtime_params_from_config_module(
    config_module_str: str,
) -> tuple[torax.Sim, torax.GeneralRuntimeParams]:
  """Returns a Sim and RuntimeParams from the config module."""
  config_module = _import_module(config_module_str)
  if hasattr(config_module, 'CONFIG'):
    # The module likely uses the "basic" config setup which has a single CONFIG
    # dictionary defining the full simulation.
    config = config_module.CONFIG
    new_runtime_params = build_sim.build_runtime_params_from_config(
        config['runtime_params']
    )
    sim = build_sim.build_sim_from_config(config)
  elif hasattr(config_module, 'get_runtime_params') and hasattr(
      config_module, 'get_sim'
  ):
    # The module is likely using the "advances", more Python-forward
    # configuration setup.
    new_runtime_params = config_module.get_runtime_params()
    sim = config_module.get_sim()
  else:
    raise ValueError(
        f'Config module {config_module_str} must either define a get_sim() '
        'method or a CONFIG dictionary.'
    )
  return sim, new_runtime_params


def _get_yes_or_no() -> bool:
  """Returns a boolean indicating yes depending on user input."""
  input_text = None
  while input_text is None:
    input_text = input('y/n: ')
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


def main(_):
  config_module_str = _PYTHON_CONFIG_MODULE.value
  if config_module_str is None:
    raise ValueError(f'--{_PYTHON_CONFIG_MODULE.name} must be specified.')
  log_sim_progress = _LOG_SIM_PROGRESS.value
  plot_sim_progress = _PLOT_SIM_PROGRESS.value
  log_sim_output = _LOG_SIM_OUTPUT.value
  sim = None
  new_runtime_params = None
  try:
    sim, new_runtime_params = _build_sim_and_runtime_params_from_config_module(
        config_module_str
    )
    simulation_app.main(
        lambda: sim,
        output_dir=new_runtime_params.output_dir,
        log_sim_progress=log_sim_progress,
        plot_sim_progress=plot_sim_progress,
        log_sim_output=log_sim_output,
    )
  except ValueError as ve:
    simulation_app.log_to_stdout(
        f'Error ocurred: {ve}',
        color=simulation_app.AnsiColors.RED,
    )
    simulation_app.log_to_stdout(
        'Not running sim. Update config and try again.',
        color=simulation_app.AnsiColors.RED,
    )
  user_command = prompt_user(config_module_str)
  while user_command != _UserCommand.QUIT:
    match user_command:
      case _UserCommand.QUIT:
        # This line shouldn't get hit, but is here for pytype.
        return  # Exit the function.
      case _UserCommand.RUN:
        if sim is None or new_runtime_params is None:
          simulation_app.log_to_stdout(
              'Need to reload the simulation.',
              color=simulation_app.AnsiColors.RED,
          )
          simulation_app.log_to_stdout(
              'Try changing the config and running with'
              f' {_UserCommand.CHANGE_SIM_OBJ.value[1]} from the main menu.',
              color=simulation_app.AnsiColors.RED,
          )
        else:
          simulation_app.main(
              lambda: sim,
              output_dir=new_runtime_params.output_dir,
              log_sim_progress=log_sim_progress,
              plot_sim_progress=plot_sim_progress,
              log_sim_output=log_sim_output,
          )
      case _UserCommand.CHANGE_CONFIG:
        # See docstring for detailed info on what recompiles.
        if sim is None or new_runtime_params is None:
          simulation_app.log_to_stdout(
              'Need to reload the simulation.',
              color=simulation_app.AnsiColors.RED,
          )
          simulation_app.log_to_stdout(
              'Try changing the config and running with'
              f' {_UserCommand.CHANGE_SIM_OBJ.value[1]} from the main menu.',
              color=simulation_app.AnsiColors.RED,
          )
        else:
          try:
            sim_and_runtime_params_or_none = change_config(
                sim, config_module_str
            )
            if sim_and_runtime_params_or_none is not None:
              sim, new_runtime_params = sim_and_runtime_params_or_none
          except ValueError as ve:
            simulation_app.log_to_stdout(
                f'Error ocurred: {ve}',
                color=simulation_app.AnsiColors.RED,
            )
            simulation_app.log_to_stdout(
                'Update config and try again.',
                color=simulation_app.AnsiColors.RED,
            )
      case _UserCommand.CHANGE_SIM_OBJ:
        # This always builds a new object and requires recompilation.
        try:
          sim, new_runtime_params, config_module_str = change_sim_obj(
              config_module_str
          )
        except ValueError as ve:
          simulation_app.log_to_stdout(
              f'Error ocurred: {ve}',
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
      case _:
        raise ValueError('Unknown command')
    user_command = prompt_user(config_module_str)


if __name__ == '__main__':
  app.run(main)

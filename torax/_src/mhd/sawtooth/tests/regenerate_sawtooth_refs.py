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

r"""Script to regenerate reference values for sawtooth model tests.

This script recalculates the sawtooth crash reference values and saves them
to a local JSON file: `sawtooth_references.json` in this test directory.

Usage Examples:

# To regenerate references and print a summary:
python -m torax._src.mhd.sawtooth.tests.regenerate_sawtooth_refs

# To regenerate and save to JSON file:
python -m torax._src.mhd.sawtooth.tests.regenerate_sawtooth_refs --write_to_file
"""

from collections.abc import Sequence
import json
import logging
import pathlib
import pprint
from typing import Any

from absl import app
from absl import flags
import numpy as np

from torax._src.config import build_runtime_params
from torax._src.orchestration import initial_state as initial_state_lib
from torax._src.orchestration import step_function
from torax._src.torax_pydantic import model_config


FLAGS = flags.FLAGS

_WRITE_TO_FILE = flags.DEFINE_bool(
    'write_to_file',
    False,
    'If True, saves the new reference values to sawtooth_references.json.',
)
_PRINT_SUMMARY = flags.DEFINE_bool(
    'print_summary',
    True,
    'If True, prints the arrays to the console.',
)

# Test configuration constants
NRHO = 10
CRASH_STEP_DURATION = 1e-3
FIXED_DT = 0.1

# Path to the local references file
REFERENCES_FILE = 'sawtooth_references.json'


class NumpyEncoder(json.JSONEncoder):
  """Custom JSON encoder for NumPy types."""

  def default(self, o):
    if isinstance(o, np.ndarray):
      return o.tolist()
    if isinstance(o, np.integer):
      return int(o)
    if isinstance(o, np.floating):
      return float(o)
    return super().default(o)


def get_sawtooth_test_config() -> dict[str, Any]:
  """Returns the test configuration dictionary for sawtooth tests.
  
  This configuration is shared between the test and reference generation.
  """
  return {
      'numerics': {
          'evolve_current': True,
          'evolve_density': True,
          'evolve_ion_heat': True,
          'evolve_electron_heat': True,
          'fixed_dt': FIXED_DT,
      },
      # Default initial current will lead to a sawtooth being triggered.
      'profile_conditions': {
          'Ip': 13e6,
          'initial_j_is_total_current': True,
          'initial_psi_from_j': True,
          'current_profile_nu': 3,
          'n_e_nbar_is_fGW': True,
          'normalize_n_e_to_nbar': True,
          'nbar': 0.85,
          'n_e': {0: {0.0: 1.5, 1.0: 1.0}},
      },
      'plasma_composition': {},
      'geometry': {'geometry_type': 'circular', 'n_rho': NRHO},
      'pedestal': {},
      'sources': {'ohmic': {}},
      'solver': {
          'solver_type': 'linear',
          'use_pereverzev': False,
      },
      'time_step_calculator': {'calculator_type': 'fixed'},
      'transport': {'model_name': 'constant'},
      'mhd': {
          'sawtooth': {
              'trigger_model': {
                  'model_name': 'simple',
                  'minimum_radius': 0.2,
                  's_critical': 0.2,
              },
              'redistribution_model': {
                  'model_name': 'simple',
                  'flattening_factor': 1.01,
                  'mixing_radius_multiplier': 1.5,
              },
              'crash_step_duration': CRASH_STEP_DURATION,
          }
      },
  }


def calculate_sawtooth_crash_references() -> dict[str, Any]:
  """Calculates sawtooth crash reference values by running a simulation step.

  This function:
  1. Builds a test configuration that triggers a sawtooth crash
  2. Runs one simulation step
  3. Verifies the crash occurred
  4. Returns post-crash profile values

  Returns:
    Dictionary containing post-crash reference values.

  Raises:
    ValueError: If sawtooth crash did not occur.
  """
  test_config_dict = get_sawtooth_test_config()
  torax_config = model_config.ToraxConfig.from_dict(test_config_dict)

  # Build solver and step function
  solver = torax_config.solver.build_solver(
      physics_models=torax_config.build_physics_models(),
  )
  geometry_provider = torax_config.geometry.build_provider
  runtime_params_provider = (
      build_runtime_params.RuntimeParamsProvider.from_config(torax_config)
  )

  step_fn = step_function.SimulationStepFn(
      solver=solver,
      time_step_calculator=torax_config.time_step_calculator.time_step_calculator,
      geometry_provider=geometry_provider,
      runtime_params_provider=runtime_params_provider,
  )

  # Get initial state (using new API that takes step_fn as main argument)
  initial_state, initial_post_processed_outputs = (
      initial_state_lib.get_initial_state_and_post_processed_outputs(
          step_fn=step_fn,
      )
  )

  # Run one step - this should trigger sawtooth crash
  output_state, _ = step_fn(
      input_state=initial_state,
      previous_post_processed_outputs=initial_post_processed_outputs,
  )

  # Verify sawtooth crash occurred
  if not output_state.solver_numeric_outputs.sawtooth_crash:
    raise ValueError(
        'Sawtooth crash did not occur! Check sawtooth model configuration.'
    )

  # Extract post-crash profiles
  return {
      'post_crash_temperature': np.asarray(output_state.core_profiles.T_e.value),
      'post_crash_n': np.asarray(output_state.core_profiles.n_e.value),
      'post_crash_psi': np.asarray(output_state.core_profiles.psi.value),
  }


def _print_full_summary(new_values: dict[str, np.ndarray]):
  """Prints the full regenerated reference values for inspection."""
  pretty_printer = pprint.PrettyPrinter(indent=4, width=100)
  logging.info('Sawtooth crash reference values:')
  for name, value in new_values.items():
    logging.info('  %s:', name)
    pretty_printer.pprint(value)
  print('-' * 20)


def get_references_path() -> pathlib.Path:
  """Returns the path to the local references file."""
  return pathlib.Path(__file__).parent / REFERENCES_FILE


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  output_path = get_references_path()

  np.set_printoptions(
      precision=12, suppress=True, threshold=np.inf, linewidth=100
  )

  logging.info('Regenerating sawtooth crash references...')
  new_values = calculate_sawtooth_crash_references()

  if _PRINT_SUMMARY.value:
    _print_full_summary(new_values)

  if _WRITE_TO_FILE.value:
    logging.info('Writing regenerated data to %s...', output_path)
    with open(output_path, 'w') as f:
      json.dump(new_values, f, indent=2, cls=NumpyEncoder)
    logging.info('Done.')
  else:
    logging.info(
        'Finished dry run. Not writing to file. Use --write_to_file to save.'
    )


if __name__ == '__main__':
  app.run(main)

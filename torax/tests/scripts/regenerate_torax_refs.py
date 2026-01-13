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

r"""Script to regenerate reference values for unit tests.

This script recalculates the reference values and saves them to a single JSON
file: `torax/_src/test_utils/references.json`. The `torax_refs.py`
module then loads this file to build the reference objects for tests.

Usage Examples:

# To regenerate all reference cases and print full summaries to the console
python -m torax.tests.scripts.regenerate_refs

# To regenerate and print a summary of a specific case:
python -m torax.tests.scripts.regenerate_refs
--case=circular_references

# To regenerate and overwrite the references.json data file, with no summaries:
python -m torax.tests.scripts.regenerate_refs --write_to_file --no_summary

# By default the script writes to the in-tree references.json file. To write to
a custom location, use the --output_dir flag:
python -m torax.tests.scripts.regenerate_refs --write_to_file --no_summary
--output_dir=/path/to/my/custom/directory
"""
from collections.abc import Callable, Sequence
import json
import logging
import os
import pathlib
import pprint
from typing import Any
from absl import app
from absl import flags
import numpy as np
from torax._src import constants
from torax._src import path_utils
from torax._src.core_profiles import initialization
from torax._src.fvm import cell_variable
from torax._src.geometry import standard_geometry
from torax._src.physics import psi_calculations
from torax._src.sources import source_profile_builders
from torax._src.test_utils import torax_refs

FLAGS = flags.FLAGS

_CASE = flags.DEFINE_multi_string(
    'case',
    None,
    'Which reference case(s) to regenerate. If not specified, all cases are'
    ' regenerated. Options are: '
    + ', '.join(torax_refs.REFERENCES_REGISTRY.keys()),
)
_WRITE_TO_FILE = flags.DEFINE_bool(
    'write_to_file',
    False,
    'If True, saves the new reference values to references.json, overwriting'
    ' the old file.',
)
_PRINT_SUMMARY = flags.DEFINE_bool(
    'print_summary',
    False,
    'If True, prints the arrays to the console.',
)
# Needed to test-time name collision with the flag in run_simulation_main.py.
if 'output_dir' not in FLAGS:
  _OUTPUT_DIR = flags.DEFINE_string(
      'output_dir',
      None,
      'Custom directory to write the references.json file to. If not specified,'
      ' the default location in the TORAX source tree will be used.',
  )
else:
  _OUTPUT_DIR = None


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


def _calculate_new_references(
    config_generator_func: Callable[[], torax_refs.References],
) -> dict[str, Any]:
  """Calculates a fresh set of reference values using TORAX's core logic."""
  reference = config_generator_func()
  torax_config = reference.config
  source_models = torax_config.sources.build_models()
  neoclassical_models = torax_config.neoclassical.build_models()
  runtime_params, geo = reference.get_runtime_params_and_geo()
  initial_core_profiles = initialization.initial_core_profiles(
      runtime_params,
      geo,
      source_models=source_models,
      neoclassical_models=neoclassical_models,
  )
  source_profiles = source_profile_builders.build_all_zero_profiles(geo)
  source_profile_builders.build_standard_source_profiles(
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=initial_core_profiles,
      source_models=source_models,
      psi_only=True,
      calculate_anyway=True,
      calculated_source_profiles=source_profiles,
  )
  if not isinstance(geo, standard_geometry.StandardGeometry):
    external_current = sum(source_profiles.psi.values())
    j_total_hires = (
        initialization.get_j_toroidal_total_hires_with_external_sources(
            runtime_params=runtime_params,
            geo=geo,
            bootstrap_current=source_profiles.bootstrap_current,
            j_toroidal_external=psi_calculations.j_parallel_to_j_toroidal(
                sum(source_profiles.psi.values()), geo
            ),
        )
    )
    psi = initialization.update_psi_from_j(
        runtime_params.profile_conditions.Ip,
        geo,
        j_total_hires,
    )
  elif isinstance(geo, standard_geometry.StandardGeometry):
    external_current = sum(source_profiles.psi.values())
    psi_value = geo.psi_from_Ip
    psi_constraint = (
        runtime_params.profile_conditions.Ip
        * (16 * np.pi**3 * constants.CONSTANTS.mu_0 * geo.Phi_b)
        / (geo.g2g3_over_rhon_face[-1] * geo.F_face[-1])
    )
    psi = cell_variable.CellVariable(
        value=psi_value,
        face_centers=geo.rho_face_norm,
        right_face_grad_constraint=psi_constraint,
    )
  else:
    raise ValueError(f'Unsupported geometry type: {type(geo)}')

  j_total, _, _ = psi_calculations.calc_j_total(
      geo,
      psi,
  )

  q_face = psi_calculations.calc_q_face(geo, psi)

  s_face = psi_calculations.calc_s_face(geo, psi)

  conductivity = neoclassical_models.conductivity.calculate_conductivity(
      geometry=geo,
      core_profiles=initial_core_profiles,
  )
  psidot = psi_calculations.calculate_psidot_from_psi_sources(
      psi_sources=external_current,
      sigma=conductivity.sigma,
      resistivity_multiplier=runtime_params.numerics.resistivity_multiplier,
      psi=psi,
      geo=geo,
  )

  return {
      'psi': np.asarray(psi.value),
      'psi_face_grad': np.asarray(psi.face_grad()),
      'psidot': np.asarray(psidot),
      'j_total': np.asarray(j_total),
      'q': np.asarray(q_face),
      's': np.asarray(s_face),
  }


def _print_full_summary(case_name: str, new_values: dict[str, np.ndarray]):
  """Prints the full regenerated reference values for inspection."""
  pretty_printer = pprint.PrettyPrinter(indent=4, width=100)
  logging.info('  Full data for %s:', case_name)
  for name, value in new_values.items():
    logging.info('    %s.%s:', case_name, name)
    pretty_printer.pprint(value)
  print('-' * 20)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  cases_to_run = _CASE.value or torax_refs.REFERENCES_REGISTRY.keys()
  if _OUTPUT_DIR is not None and _OUTPUT_DIR.value is not None:
    output_dir = pathlib.Path(_OUTPUT_DIR.value).expanduser()
    output_path = output_dir / torax_refs.JSON_FILENAME
  else:
    # Use the default path inside the torax source tree.
    output_path = (
        path_utils.torax_path()
        / '_src'
        / 'test_utils'
        / torax_refs.JSON_FILENAME
    )

  # Read existing data first if it exists, to preserve un-regenerated cases.
  if _WRITE_TO_FILE.value and os.path.exists(output_path):
    logging.info('Loading existing data from %s to update...', output_path)
    with open(output_path, 'r') as f:
      all_data = json.load(f)
  else:
    all_data = {}

  np.set_printoptions(
      precision=12, suppress=True, threshold=np.inf, linewidth=100
  )

  # Regenerate data for the selected cases and update the dictionary.
  for case_name in cases_to_run:
    if case_name not in torax_refs.REFERENCES_REGISTRY:
      raise ValueError(
          f"Case '{case_name}' not found. Available cases:"
          f" {', '.join(torax_refs.REFERENCES_REGISTRY.keys())}"
      )

    logging.info('Regenerating references for: %s...', case_name)
    config_generator_func = torax_refs.REFERENCES_REGISTRY[case_name]
    new_values = _calculate_new_references(config_generator_func)
    all_data[case_name] = new_values

    if _PRINT_SUMMARY.value:
      _print_full_summary(case_name, new_values)

  if _WRITE_TO_FILE.value:
    logging.info('Writing all regenerated data to %s...', output_path)
    with open(output_path, 'w') as f:
      json.dump(all_data, f, indent=2, cls=NumpyEncoder)
    logging.info('Done.')
  else:
    logging.info(
        'Finished dry run. Not writing to file. Use --write_to_file to save.'
    )


if __name__ == '__main__':
  app.run(main)

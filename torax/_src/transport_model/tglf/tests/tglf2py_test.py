# Copyright 2026 DeepMind Technologies Limited
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

"""Unit tests for tglf2py against committed GACODE regression precision files."""

import os
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax._src.transport_model.tglf import tglf2py

# Internal import.


def _get_tglf_case_and_binary_paths(test_case: str) -> tuple[str, str, str]:
  """Returns (case_dir, tglf_binary_path, tglf_parse_path) for GACODE regression testing."""
  tglf_dir = os.path.join(os.environ['GACODE_ROOT'], 'tglf')
  case_dir = os.path.join(tglf_dir, 'tools', 'input', test_case)
  tglf_binary_path = os.path.join(tglf_dir, 'tglf_binary')
  tglf_parse_path = os.path.join(tglf_dir, 'tglf_parse')
  return case_dir, tglf_binary_path, tglf_parse_path


def _parse_input_tglf(input_path: str) -> dict[str, Any]:
  """Standalone parser for input.tglf files, casting raw strings to clean Python types."""
  user_dict = {}
  with open(input_path, 'r') as f:
    for line in f:
      line = line.strip()
      if line and not line.startswith('#'):
        parts = line.split('=')
        if len(parts) == 2:
          key = parts[0].strip()
          val_str = parts[1].split('#')[0].strip()

          # Cast val_str to appropriate Python type
          val_low = val_str.lower()
          if val_low in [
              '.true.',
              '.false.',
              'true',
              'false',
              't',
              'f',
              'yes',
              'no',
          ]:
            val = val_low in ['.true.', 'true', '1', 't', 'yes', 'y']
          else:
            try:
              val = int(val_str)
            except ValueError:
              try:
                val = float(val_str)
              except ValueError:
                val = val_str

          user_dict[key] = val
  return user_dict


def _calculate_precision_sum(pe, pi, qe, qi) -> float:
  """Calculates GACODE's exact lumped regression precision sum across all harvested fluxes."""
  ns = len(pi) + 1
  qe_low = float(tglf2py.tglf_interface.tglf_elec_eflux_low_out)
  qi_low = np.array(tglf2py.tglf_interface.tglf_ion_eflux_low_out)[: ns - 1]
  me = float(tglf2py.tglf_interface.tglf_elec_mflux_out)
  mi = np.array(tglf2py.tglf_interface.tglf_ion_mflux_out)[: ns - 1]

  prec = (
      abs(pe)
      + abs(qe)
      + abs(qe_low)
      + abs(me)
      + np.sum(np.abs(pi))
      + np.sum(np.abs(qi))
      + np.sum(np.abs(qi_low))
      + np.sum(np.abs(mi))
  )
  return float(prec)


@absltest.skipIf(
    tglf2py.tglf2py_lib is None,
    "TGLF extension module 'tglf2py_lib' is not compiled/installed.",
)
class Tglf2pyTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Common settings for the GACODE regression test cases.
    self.gacode_regression_settings = {
        'RLTS_1': 3.0,
        'RLTS_2': 3.0,
        'RLNS_1': 1.0,
        'RLNS_2': 1.0,
        'AS_1': 1.0,
        'AS_2': 1.0,
        'TAUS_1': 1.0,
        'TAUS_2': 1.0,
        'VPAR_1': 0.0,
        'VPAR_2': 0.0,
    }

  def test_run_tglf_lowercase_raises(self):
    """Tests that only capitalized GACODE parameters are accepted."""
    with self.assertRaisesRegex(ValueError, 'Expected capitalized'):
      tglf2py.run_tglf(tglf_use_bper_in=True)

  def test_registers_reset_between_runs(self):
    """Tests that Fortran module registers are reset to defaults between runs."""
    # 1. Run first simulation passing custom overrides.
    tglf2py.run_tglf(
        **self.gacode_regression_settings,
        USE_BPER=True,  # bool
        ZS_2=50.0,  # float, species vector
        ETG_FACTOR=2.5,  # float, scalar
    )

    # 2. Check that overrides were applied to the interface registers.
    self.assertEqual(int(tglf2py.tglf_interface.tglf_use_bper_in), 1)
    self.assertAlmostEqual(float(tglf2py.tglf_interface.tglf_zs_in[1]), 50.0)
    self.assertAlmostEqual(
        float(tglf2py.tglf_interface.tglf_etg_factor_in), 2.5
    )

    # 3. Run second simulation with only base settings.
    tglf2py.run_tglf(**self.gacode_regression_settings)

    # 4. Ensure all registers were reset back to their default values.
    self.assertEqual(int(tglf2py.tglf_interface.tglf_use_bper_in), 0)
    self.assertAlmostEqual(float(tglf2py.tglf_interface.tglf_zs_in[1]), 1.0)
    self.assertAlmostEqual(
        float(tglf2py.tglf_interface.tglf_etg_factor_in), 1.25
    )

  @parameterized.parameters(
      ('tglf01',),
      ('tglf02',),
      ('tglf03',),
      ('tglf04',),
      ('tglf05',),
      ('tglf06',),
      ('tglf07',),
      ('tglf08',),
      ('tglf09',),
  )
  def test_numerical_parity_against_gacode_regress_files(self, test_case):
    case_dir, _, _ = _get_tglf_case_and_binary_paths(test_case)

    # Load expected output precision sum.
    prec_path = os.path.join(case_dir, 'out.tglf.prec')
    with open(prec_path, 'r') as f:
      ref_prec = float(f.read().strip())

    # Load per-case input overrides.
    input_path = os.path.join(case_dir, 'input.tglf')
    settings_dict = _parse_input_tglf(input_path)

    # Run pytglf wrapper passing base profiles and parsed overrides.
    pe, pi, qe, qi = tglf2py.run_tglf(
        **(self.gacode_regression_settings | settings_dict)
    )
    py_prec = _calculate_precision_sum(pe, pi, qe, qi)

    np.testing.assert_allclose(py_prec, ref_prec, rtol=1e-8, atol=1e-10)


if __name__ == '__main__':
  absltest.main()

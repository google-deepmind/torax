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
import shutil
import subprocess
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax._src.transport_model.tglf import tglf2py

# Internal import.


def _get_tglf_case_and_binary_paths(test_case: str) -> tuple[str, str]:
  """Returns (case_dir, tglf_binary_path) for GACODE regression testing."""
  tglf_dir = os.path.join(os.environ['GACODE_ROOT'], 'tglf')
  case_dir = os.path.join(tglf_dir, 'tools', 'input', test_case)
  tglf_binary_path = os.path.join(tglf_dir, 'tglf_binary')
  return case_dir, tglf_binary_path


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


class Tglf2pyTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # GA standard case base profiles in raw TGLF notation
    self.rlts = np.array([3.0, 3.0])
    self.rlns = np.array([1.0, 1.0])
    self.as_vec = np.array([1.0, 1.0])
    self.taus = np.array([1.0, 1.0])
    self.vpar = np.array([0.0, 0.0])

  def test_run_tglf_defaults(self):
    """Tests calling run_tglf with default profile vectors using raw TGLF kwarg notation."""
    pe, pi, qe, qi = tglf2py.run_tglf(
        RLTS=self.rlts,
        RLNS=self.rlns,
        AS=self.as_vec,
        TAUS=self.taus,
        VPAR=self.vpar,
    )
    prec = _calculate_precision_sum(pe, pi, qe, qi)
    self.assertTrue(np.isfinite(pe))
    self.assertTrue(np.all(np.isfinite(pi)))
    self.assertTrue(np.isfinite(qe))
    self.assertTrue(np.all(np.isfinite(qi)))
    self.assertTrue(np.isfinite(prec))
    self.assertEqual(pi.shape, (1,))
    self.assertEqual(qi.shape, (1,))

  def test_run_tglf_capitalized_indexed_kwargs(self):
    """Tests calling run_tglf using capitalized indexed kwarg notation."""
    pe, pi, qe, qi = tglf2py.run_tglf(
        RLTS=self.rlts,
        RLNS=self.rlns,
        AS_1=1.0,
        AS_2=1.0,
        TAUS=self.taus,
        VPAR=self.vpar,
        USE_BPER=True,
    )
    self.assertTrue(np.isfinite(pe))
    self.assertTrue(np.all(np.isfinite(pi)))
    self.assertTrue(np.isfinite(qe))
    self.assertTrue(np.all(np.isfinite(qi)))

  def test_run_tglf_lowercase_raises(self):
    with self.assertRaisesRegex(ValueError, 'Expected capitalized'):
      tglf2py.run_tglf(tglf_use_bper_in=True)

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
    """Dynamically loads expected inputs and outputs from committed GACODE files."""
    case_dir, _ = _get_tglf_case_and_binary_paths(test_case)
    run_dir = self.create_tempdir(name=test_case).full_path

    # Load expected output precision sum.
    prec_path = os.path.join(case_dir, 'out.tglf.prec')
    with open(prec_path, 'r') as f:
      ref_prec = float(f.read().strip())

    # Load expected input overrides from writable temp copy.
    input_src = os.path.join(case_dir, 'input.tglf')
    input_path = os.path.join(run_dir, 'input.tglf')
    shutil.copy2(input_src, input_path)
    user_dict = _parse_input_tglf(input_path)

    # Run pytglf wrapper passing base profiles and parsed overrides.
    pe, pi, qe, qi = tglf2py.run_tglf(
        RLTS=self.rlts,
        RLNS=self.rlns,
        AS=self.as_vec,
        TAUS=self.taus,
        VPAR=self.vpar,
        **user_dict,
    )
    py_prec = _calculate_precision_sum(pe, pi, qe, qi)

    np.testing.assert_allclose(py_prec, ref_prec, rtol=1e-8, atol=1e-10)

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
  def test_vanilla_tglf_subprocess_comparison(self, test_case):
    """Runs vanilla TGLF in a subprocess and compares outputs against pytglf."""
    case_dir, tglf_binary_path = _get_tglf_case_and_binary_paths(test_case)
    run_dir = self.create_tempdir(name=f'vanilla_{test_case}').full_path

    # Copy input.tglf and input.tglf.gen to run_dir
    input_src = os.path.join(case_dir, 'input.tglf')
    input_path = os.path.join(run_dir, 'input.tglf')
    shutil.copy2(input_src, input_path)
    gen_src = os.path.join(case_dir, 'input.tglf.gen')
    gen_path = os.path.join(run_dir, 'input.tglf.gen')
    shutil.copy2(gen_src, gen_path)
    user_dict = _parse_input_tglf(input_path)

    # Run vanilla TGLF binary in subprocess
    subprocess.run([tglf_binary_path], cwd=run_dir, check=True)

    # Parse vanilla TGLF outputs from out.tglf.gbflux
    with open(os.path.join(run_dir, 'out.tglf.gbflux'), 'r') as f:
      vals = [float(x) for x in f.read().split()]

    # Run pytglf
    pe, pi, qe, qi = tglf2py.run_tglf(
        RLTS=self.rlts,
        RLNS=self.rlns,
        AS=self.as_vec,
        TAUS=self.taus,
        VPAR=self.vpar,
        **user_dict,
    )

    n = len(pi)
    vanilla_pe = vals[0]
    vanilla_pi = np.array(vals[1 : 1 + n])
    vanilla_qe = vals[1 + n]
    vanilla_qi = np.array(vals[2 + n : 2 + 2 * n])

    np.testing.assert_allclose(pe, vanilla_pe, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(pi, vanilla_pi, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(qe, vanilla_qe, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(qi, vanilla_qi, rtol=1e-6, atol=1e-8)


if __name__ == '__main__':
  absltest.main()

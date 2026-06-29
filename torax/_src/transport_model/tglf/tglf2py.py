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

"""TGLF Python wrapper via f2py."""

from __future__ import annotations

import numpy as np


try:
  # If type checking, this imports the stub file.
  # At runtime, this imports the compiled module.
  from torax._src.transport_model.tglf import tglf2py_lib

  tglf_interface: tglf2py_lib.TGLFInterface | None = tglf2py_lib.tglf_interface
except (ImportError, ModuleNotFoundError, AttributeError):
  # At runtime, if the extension fails to import, then set to None.
  # Type checkers will still use the stub file.
  tglf_interface = None
  tglf2py_lib = None


def _check_tglf_installed():
  if tglf2py_lib is None or tglf_interface is None:
    raise RuntimeError(
        "TGLF extension module 'tglf2py_lib' is not installed or failed to"
        ' import. Please compile the TGLF wrapper before running TGLF'
        ' simulations.'
    )


def _assign_setting(key: str, value: str | float | int | bool | np.ndarray):
  """Resolves config key and assigns setting by applying type transformations."""
  _check_tglf_installed()
  assert tglf_interface is not None
  if not key.isupper() or key.startswith('TGLF_') or key.endswith('_IN'):
    raise ValueError(
        "Expected capitalized GACODE parameter name (e.g. 'USE_BPER'), got"
        f" '{key}'."
    )

  key_lower = key.lower()

  k = f'tglf_{key_lower}_in'
  interface_key = None
  idx = None

  # Find matching interface attribute
  if hasattr(tglf_interface, k):
    # 1. Direct attribute match (e.g., tglf_use_bper_in)
    interface_key = k
  else:
    # 2. Indexed species match (e.g., tglf_mass_1_in or tglf_as_2_in)
    parts = k.split('_')
    if len(parts) > 3 and parts[-2].isdigit() and parts[-1] == 'in':
      # Convert 1-based index from label to 0-based for Python array indexing.
      candidate_idx = int(parts[-2]) - 1
      base_attr = '_'.join(parts[:-2] + ['in'])
      if hasattr(tglf_interface, base_attr) and isinstance(
          getattr(tglf_interface, base_attr), np.ndarray
      ):
        interface_key = base_attr
        idx = candidate_idx
  if interface_key is None:
    raise ValueError(f"Unknown TGLF configuration parameter: '{key}'")

  previous_value = getattr(tglf_interface, interface_key)

  # Assign value to interface attribute.
  # Species vector assignment.
  if isinstance(previous_value, np.ndarray) and previous_value.ndim > 0:
    if idx is not None:
      previous_value[idx] = float(value)
    else:
      vals = [float(x) for x in np.atleast_1d(value)]
      previous_value[: len(vals)] = vals
    return

  # Scalar assignment.
  setattr(tglf_interface, interface_key, value)


def run_tglf(**kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Runs TGLF via tglf_interface module memory.

  Args:
    **kwargs: See https://gacode.io/tglf/tglf_table.html. Values must be passed
      as capitalized GACODE parameters (e.g. USE_BPER=True). Values must be
      Python-native types (e.g. bool, str, int, float, list, or numpy.ndarray).

  Returns:
    Tuple of (tglf_elec_pflux_out, tglf_ion_pflux_out, tglf_elec_eflux_out,
    tglf_ion_eflux_out).
  """
  _check_tglf_installed()
  assert tglf2py_lib is not None
  assert tglf_interface is not None
  # Reset input arrays/vectors to match GACODE tglf_defaults.py.
  # Required to ensure hermeticity, as these registers are persistent across
  # function calls.
  # Default values for species index > 2 (species 3+) is 0.0.
  # Species 1 and 2 are set to their standard GACODE baseline defaults.
  tglf_interface.tglf_rlns_in[:] = 0.0
  tglf_interface.tglf_rlns_in[:2] = 1.0
  tglf_interface.tglf_rlts_in[:] = 0.0
  tglf_interface.tglf_rlts_in[:2] = 3.0
  tglf_interface.tglf_vpar_shear_in[:] = 0.0
  tglf_interface.tglf_vns_shear_in[:] = 0.0
  tglf_interface.tglf_vts_shear_in[:] = 0.0
  tglf_interface.tglf_taus_in[:] = 0.0
  tglf_interface.tglf_taus_in[:2] = 1.0
  tglf_interface.tglf_as_in[:] = 0.0
  tglf_interface.tglf_as_in[:2] = 1.0
  tglf_interface.tglf_vpar_in[:] = 0.0
  tglf_interface.tglf_mass_in[:] = 0.0
  tglf_interface.tglf_mass_in[0] = 2.723e-4
  tglf_interface.tglf_mass_in[1] = 1.0
  tglf_interface.tglf_zs_in[:] = 0.0
  tglf_interface.tglf_zs_in[0] = -1.0
  tglf_interface.tglf_zs_in[1] = 1.0

  # Assign all settings from kwargs to interface.
  for key, val in kwargs.items():
    _assign_setting(key, val)

  # Run TGLF.
  tglf2py_lib.tglf_run()

  # Extract flux outputs from interface.
  pe = np.array(tglf_interface.tglf_elec_pflux_out)
  qe = np.array(tglf_interface.tglf_elec_eflux_out)
  # Strip empty elements of ion buffer.
  ns = tglf_interface.tglf_ns_in
  pi = np.array(tglf_interface.tglf_ion_pflux_out)[: ns - 1]
  qi = np.array(tglf_interface.tglf_ion_eflux_out)[: ns - 1]

  return pe, pi, qe, qi

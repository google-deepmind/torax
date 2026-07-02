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

import types
from typing import Any
import numpy as np
from torax._src.transport_model.tglf import defaults as tglf_defaults

# pylint: disable=invalid-name

try:
  # If type checking, this imports the stub file.
  # At runtime, this imports the compiled module, or raises an ImportError if
  # the extension fails to import.
  from torax._src.transport_model.tglf import tglf2py_lib  # pylint: disable=g-import-not-at-top

  tglf_interface = tglf2py_lib.tglf_interface
  _TGLFInterfaceType = tglf2py_lib.TGLFInterface
  _TGLF2pyLibType = types.ModuleType
except (ImportError, ModuleNotFoundError, AttributeError):
  # At runtime, if the extension fails to import, then set it to None.
  tglf_interface = None
  tglf2py_lib = None
  _TGLFInterfaceType = Any
  _TGLF2pyLibType = Any


def _get_tglf_lib_and_interface() -> tuple[_TGLF2pyLibType, _TGLFInterfaceType]:  # pytype: disable=invalid-annotation
  """Returns the compiled tglf2py_lib and tglf_interface, or raises RuntimeError."""
  if tglf2py_lib is None or tglf_interface is None:
    raise RuntimeError(
        "TGLF extension module 'tglf2py_lib' is not installed or failed to"
        ' import. Please compile the TGLF wrapper before running TORAX with'
        ' TGLF (see'
        ' https://torax.readthedocs.io/en/latest/installation.html#optional-install-tglf).'
    )
  return tglf2py_lib, tglf_interface


def _assign_setting(key: str, value: str | bool | int | float | np.ndarray):
  """Resolves config key and assigns setting by applying type transformations."""
  _, interface = _get_tglf_lib_and_interface()
  key_lower = key.lower()
  if (
      not key.isupper()
      or key_lower.startswith('tglf_')
      or key_lower.endswith('_in')
  ):
    raise ValueError(
        "Expected capitalized GACODE parameter name (e.g. 'USE_BPER'), got"
        f" '{key}'."
    )

  parts = key_lower.split('_')
  if parts[-1].isdigit():
    # Case 1: Species vector setting (e.g. ZS_1, MASS_2). We need to set the
    # appropriate element of the corresponding array in the interface.
    # TGLF settings use 1-based indexing, convert to 0-based index for Python.
    idx = int(parts[-1]) - 1
    key_lower = '_'.join(parts[:-1])

    # Get previous value.
    interface_key = f'tglf_{key_lower}_in'
    previous_value = getattr(interface, interface_key)

    # Update the array element at idx and write to the interface.
    new_value = np.array(previous_value)
    new_value[idx] = float(value)
    setattr(interface, interface_key, new_value)
  else:
    # Case 2: Scalar setting (e.g. USE_BPER, TAUE, etc.).
    # Get previous value.
    interface_key = f'tglf_{key_lower}_in'
    previous_value = getattr(interface, interface_key)

    # Parse based on type of previous value.
    # In TGLF settings, bools can be passed as strings or as integers, so they
    # need special handling.
    # Note: bool is a subclass of int in Python, so we must check for bool
    # before checking for int.
    match previous_value.item():
      case bool() | int() | np.integer():
        val_low = str(value).strip().lower()
        if val_low in ('.true.', 'true', 't', 'yes', 'y'):
          new_value = 1
        elif val_low in ('.false.', 'false', 'f', 'no', 'n'):
          new_value = 0
        else:
          new_value = int(value)
      case float() | np.floating():
        new_value = float(value)
      case str() | bytes() | np.str_() | np.bytes_():
        new_value = str(value)
      case _:
        raise ValueError(
            f"Unsupported type '{type(previous_value.item())}' for setting"
            f" '{key}'."
        )

    # Write to the interface.
    setattr(interface, interface_key, new_value)


def run_tglf(**kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Runs TGLF via tglf_interface module memory.

  Args:
    **kwargs: See https://gacode.io/tglf/tglf_table.html. Values must be passed
      as capitalized GACODE parameters (e.g. USE_BPER=True). Values must be
      Python-native types (e.g. bool, str, int, float, list, or numpy.ndarray).

  Returns:
    Tuple of (electron_particle_flux, ion_particle_flux, electron_heat_flux,
    ion_heat_flux).
  """
  lib, interface = _get_tglf_lib_and_interface()

  # Reset all registers to GACODE defaults merged with user inputs.
  merged = tglf_defaults.TGLF_DEFAULTS | kwargs
  for key, val in merged.items():
    _assign_setting(key, val)

  # Run TGLF.
  lib.tglf_run()

  # Extract flux outputs from interface.
  electron_particle_flux = np.array(interface.tglf_elec_pflux_out)
  electron_heat_flux = np.array(interface.tglf_elec_eflux_out)
  # Strip empty elements of ion buffer.
  ns = interface.tglf_ns_in
  ion_particle_flux = np.array(interface.tglf_ion_pflux_out)[: ns - 1]
  ion_heat_flux = np.array(interface.tglf_ion_eflux_out)[: ns - 1]

  return (
      electron_particle_flux,
      ion_particle_flux,
      electron_heat_flux,
      ion_heat_flux,
  )

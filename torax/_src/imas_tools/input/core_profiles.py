# Copyright 2025 DeepMind Technologies Limited
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
"""Useful functions to load IMAS core_profiles or plasma_profiles IDSs."""
from collections.abc import Collection, Mapping
import logging
from typing import Any, Mapping, Final, Literal

from imas import ids_toplevel, ids_structure, ids_struct_array
import numpy as np
from torax._src import constants

_PROFILE_CONDITIONS_VALIDATION_FIELDS: Final[dict[str, object]]= {
    "global_quantities": ["ip", "v_loop"],
    "profiles_1d": {"required":[
        "grid.rho_tor_norm",],
        "optional": [
        "time", # Not necessary when building from a single time slice.
        "grid.psi",
        "electrons.temperature",
        "t_i_average",
        "electrons.density",
    ],},
}

_PLASMA_COMPOSITION_VALIDATION_FIELDS: Final[dict[str, object]]= {
    "global_quantities": [],
    "profiles_1d": {"required":[
        "grid.rho_tor_norm",
        "electrons.density",
        "ion.density",
        ],
        "optional": [
        "time", # Not necessary when building from a single time slice.
        "zeff",
    ],},
}
_VALIDATED_FUNCTIONS = Literal["profile_conditions", "plasma_composition"]

# pylint: disable=invalid-name
def profile_conditions_from_IMAS(
    ids: ids_toplevel.IDSToplevel,
    t_initial: float | None = None,
) -> Mapping[str, Any]:
  """Converts core_profiles IDS to a profile_conditions dict for TORAX config.

  Args:
    ids: A core_profiles IDS object. The IDS can contain multiple time slices.
    t_initial: Initial time used to map the profiles in the dicts. If None the
      initial time will be the time of the first time slice of the ids. Else all
      time slices will be shifted such that the first time slice has time =
      t_initial.

  Returns:
    The updated fields read from the IDS that can be used to completely or
    partially fill the `profile_conditions` section of a TORAX `CONFIG`.
  """
  # Set to false as the IDS as no global_quantities.v_loop
  _validate_core_profiles_ids(ids, strict=False, target="profiles_conditions")
  profiles_1d, rhon_array, time_array = _get_time_and_radial_arrays(
      ids, t_initial
  )

  # profile_conditions
  psi = (np.array(rhon_array[0]), np.array(profiles_1d[0].grid.psi))
  # TODO(b/335204606): Clean this up once we finalize our COCOS convention.
  # Ip sign is switched due to the difference between input COCOS conventions
  # and TORAX ones
  Ip = (
      time_array,
      -1 * ids.global_quantities.ip,
  )

  # It is assumed the temperatures and density profiles are defined until
  # rhon=1. Validator will raise an error if rhon[-1]!= 1.
  # Temperatures are converted from eV (IMAS standard unit) to keV.
  T_e = (
      time_array,
      rhon_array,
      [profile.electrons.temperature / 1e3 for profile in profiles_1d],
  )

  if profiles_1d[0].t_i_average:
    T_i = (
        time_array,
        rhon_array,
        [profile.t_i_average / 1e3 for profile in profiles_1d],
    )
  else:
    t_i_average = [
        np.average(
            [ion.temperature / 1e3 for ion in profile.ion],
            axis=0,
            weights=[ion.density for ion in profile.ion],
        )
        for profile in profiles_1d
    ]
    T_i = (
        time_array,
        rhon_array,
        t_i_average,
    )

  n_e = (
      time_array,
      rhon_array,
      [profile.electrons.density for profile in profiles_1d],
  )

  # Map v_loop_lcfs in case it is used as bc for psi equation.
  if ids.global_quantities.v_loop:
    v_loop_lcfs = (time_array, ids.global_quantities.v_loop)
  else:
    v_loop_lcfs = 0.0

  return {
      "Ip": Ip,
      "psi": psi,
      "T_i": T_i,
      "T_i_right_bc": None,
      "T_e": T_e,
      "T_e_right_bc": None,
      "n_e_right_bc_is_fGW": False,
      "n_e_right_bc": None,
      "n_e_nbar_is_fGW": False,
      "nbar": None,
      "n_e": n_e,
      "normalize_n_e_to_nbar": False,
      "v_loop_lcfs": v_loop_lcfs,
  }


def plasma_composition_from_IMAS(
    ids: ids_toplevel.IDSToplevel,
    t_initial: float | None = None,
    expected_impurities: Collection[str] | None = None,
    main_ions_symbols: Collection[str] = constants.HYDROGENIC_IONS,
) -> Mapping[str, Any]:
  """Returns dict with args for plasma_composition config from a given ids.

  plasma_composition dict obtained from IMAS can be used with two impurity
  modes (contained within the `impurity` key):
    - `n_e_ratios` - default returned from this function (note Z_eff will be
    ignored in this case).
    - `n_e_ratios_Z_eff` - In this case one impurity ratio should be set to
    None and the `impurity_mode` string overwritten.
    - `fractions` impurity mode is unsupported.
  Note that if the input ids does not contain info for the different ion
  species, the function will not raise an error but return an empty dict for
  `main_ion` and `species` plasma_composition keys unless the expected species
  are specified using the expected_impurities and main_ion_symbols args.

  Args:
    ids: A core_profiles IDS object. The IDS can contain multiple time slices.
    t_initial: Initial time used to map the profiles in the dicts. If None the
      initial time will be the time of the first time slice of the ids. Else all
      time slices will be shifted such that the first time slice has time =
      t_initial.
    expected_impurities: Optional arg to check that the input IDS contains the
      wanted impurity species and raise and error if not, or if its density is
      empty.
    main_ions_symbols: collection of ions to be used to define the main_ion
      mixture. If value is not the default one, will check that the given ions
      exist in the IDS and their density is filled. Default are hydrogenic ions
      H, D, T.

  Returns:
    The updated fields read from the IDS that can be used to completely or
    partially fill the `plasma_composition` section of a TORAX `CONFIG`.
  """
  profiles_1d, rhon_array, time_array = _get_time_and_radial_arrays(
      ids, t_initial
  )
  _validate_core_profiles_ids(ids, strict=True, target="plasma_composition")
  # Check that the expected ions are present in the IDS
  ids_ions = [ion.name for ion in profiles_1d[0].ion if ion.density]
  if expected_impurities:
    _check_expected_ions_presence(ids_ions, expected_impurities)
  if main_ions_symbols is not constants.HYDROGENIC_IONS:
    _check_expected_ions_presence(ids_ions, main_ions_symbols)

  Z_eff = (
      time_array,
      rhon_array,
      [profile.zeff for profile in profiles_1d],
  )
  impurity_species = {}
  main_ion_density = {}
  for ion in range(len(profiles_1d[0].ion)):
    try:
      symbol = str(profiles_1d[0].ion[ion].name)
    except AttributeError:
      # TODO(b/459479939): i/539) - Indicate supported dd_versions and switch on
      # that instead of using a try-except.
      # Case ids is plasma_profiles in early DDv4 releases.
      symbol = str(profiles_1d[0].ion[ion].label)
    if symbol in constants.ION_PROPERTIES_DICT.keys():
      # Fill main ions
      if symbol in main_ions_symbols:
        main_ion_density[symbol] = [
            profile.ion[ion].density[0] for profile in profiles_1d
        ]
        # TODO(b/459479939): i/539 - Take the ratios of volume integrated
        # densities instead of the central density values.
      # Fill impurities
      else:
        n_e_ratio = (
            time_array,
            rhon_array,
            [
                profile.ion[ion].density / profile.electrons.density
                for profile in profiles_1d
            ],
        )
        impurity_species[symbol] = n_e_ratio
  total_main_ion_density = np.sum(
      [ratio for ratio in main_ion_density.values()], axis=0
  )
  main_ion = {}
  for symbol, ratio in main_ion_density.items():
    main_ion[symbol] = (time_array, ratio / total_main_ion_density)
  return {
      "main_ion": main_ion,
      "Z_eff": Z_eff,
      "impurity": {
          "impurity_mode": "n_e_ratios",
          "species": impurity_species,
      },
  }


def _get_time_and_radial_arrays(
    ids: ids_toplevel.IDSToplevel,
    t_initial: float | None = None,
) -> tuple[ids_toplevel.IDSStructure, list[list[float]], list[float]]:
  profiles_1d = ids.profiles_1d
  time_array = [profile.time for profile in profiles_1d]
  if t_initial:
    time_array = [t - time_array[0] + t_initial for t in time_array]
  rhon_array = [profile.grid.rho_tor_norm for profile in profiles_1d]
  return profiles_1d, rhon_array, time_array


def _check_expected_ions_presence(
    ids_ions: list[str],
    expected_ions: Collection[str],
) -> None:
  """Checks that the expected_ions symbols are in the ids_ions."""
  for ion in expected_ions:
    if ion not in constants.ION_PROPERTIES_DICT.keys():
      raise (KeyError(f"{ion} is not a valid symbol of a TORAX valid ion."))
    if ion not in ids_ions:
      raise (
          ValueError(
              f"The expected ion {ion} cannot be found in the input"
              " IDS or has no valid data. \n Please check that the IDS is"
              " properly filled"
          )
      )


def _validate_core_profiles_ids(
    ids: ids_toplevel.IDSToplevel,
    strict: bool, 
    target: str,
):
  """Checks that all expected attributes exist in the IDS."""
  profiles_1d = ids.profiles_1d
  global_quantities = ids.global_quantities
  logged_fields = set()
  if target == "profiles_conditions":
    validation_fields = _PROFILE_CONDITIONS_VALIDATION_FIELDS
  if target == "plasma_composition":
    validation_fields = _PLASMA_COMPOSITION_VALIDATION_FIELDS
  # Validate global_quantities
  for field in validation_fields["global_quantities"]:
    leaf = getattr(global_quantities,field)
    if not leaf.has_value:
      if strict: 
        raise ValueError(f"The IDS is missing global_quantities.{field} required to build"
            " profile_conditions. \n Please Check that your IDS is properly"
            " filled.")
      else:
        logging.warning(
            f"The IDS is missing global_quantities.{field} which may cause undesired behavior when building profile_conditions. \n Please Check that your IDS is properly"
            " filled."
        )
  # Validate profiles_1d 
  for profile in profiles_1d:
    for field in validation_fields["profiles_1d"]["required"]:
      _check_profiles_1d_attribute(field, profile, logged_fields, strict=True)
    for field in validation_fields["profiles_1d"]["optional"]:
      _check_profiles_1d_attribute(field, profile, logged_fields, strict)

def _check_profiles_1d_attribute(field: str, profile: ids_structure.IDSStructure, logged_fields:set, strict: bool):
  leaves = [profile]
  # Explore the IDS tree to recover the attribute to validate.
  for node in field.split("."):
    next_leaves = []
    for leaf in leaves:
      # Case when the attribute exists on the current object.
      if hasattr(leaf, node):
        next_leaves.append(getattr(leaf, node))
      # AoS case: the current object is an IDS array of structure. e.g. profiles_1d[i].ion
      elif isinstance(leaf, ids_struct_array.IDSStructArray):
        for sub_leaf in leaf:
          next_leaves.append(getattr(sub_leaf, node))
    leaves = next_leaves
  # Validate that all retrieved attributes have a value.  
  for leaf in leaves:
    if not leaf.has_value:
      if field not in logged_fields:
        if strict:
          raise ValueError(f"The IDS is missing profiles_1d.{field} to build"
          " profile_conditions. \n Please Check that your IDS is properly"
          " filled.")
        else:
          logging.warning(
              f"The IDS is missing profiles_1d.{field} which may cause undesired behavior in building profile_conditions. \n Please Check that your IDS is properly"
              " filled."
          )
          logged_fields.add(field)

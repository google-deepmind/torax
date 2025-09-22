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

"""Useful functions to load IMAS core_profiles or plasma_profiles IDSs and
converts them into TORAX objects.
"""
from typing import Any, Mapping

from imas.ids_toplevel import IDSToplevel
import numpy as np
import torax._src.constants as constants


def update_dict(old_dict: dict, updates: dict) -> dict:
  """Recursively modify the fields from the original dict old_dict using the
     values contained in updates dict.

  Used to update config dict fields more easily. Use case is to update
  config dict with output from core_profiles.core_profiles_from_IMAS().
  The function will read the keys of the old dict and replace the keys
  existing in updates. To handle nested dicts, if the value of a key is a
  dict it will either replace the whole dict if the keys of the dict are
  floats (profiles type dicts) or recursively call the function with the
  value dict as old_dict arg if the key is a string.

  Args:
     old_dict: The current dict that needs to be updated.
     updates: Dict containing the values of the keys that need to be updated in
     old_dict.

  Returns:
      New updated copy of the dict.
  """
  new_dict = old_dict.copy()
  for key, value in updates.items():
    if (
        isinstance(value, dict)
        and key in new_dict
        and isinstance(new_dict[key], dict)
    ):
      if all(isinstance(k, float) for k in value.keys()):
        # Replace completely if keys of the dict are numeric
        new_dict[key] = value
      else:
        new_dict[key] = update_dict(new_dict[key], value)
    else:
      new_dict[key] = value
  return new_dict


def core_profiles_from_IMAS(
    ids: IDSToplevel,
    t_initial: float | None = None,
) -> Mapping[str, Mapping[str, Any]]:
  """Converts core_profiles IDS to a dict with the input profiles for the config.

  Args:
     ids: IDS object. Can be either core_profiles or plasma_profiles. The IDS can
        contain multiple time slices.
     t_initial: Initial time used to map the profiles in the dicts. If None the
        initial time will be the time of the first time slice of the ids. Else
        all time slices will be shifted such that the first time slice has
        time = t_initial.

  Returns:
     Dict containing the updated fields read from the IDS that need to be replaced
     in the input config using ToraxConfig.update_fields method.
  """
  profiles_1d = ids.profiles_1d
  time_array = [float(profiles_1d[i].time) for i in range(len(profiles_1d))]
  if t_initial:
    time_array = [ti - time_array[0] + t_initial for ti in time_array]
  else:
    t_initial = float(profiles_1d[0].time)
  rhon_array = [
      profiles_1d[i].grid.rho_tor_norm for i in range(len(profiles_1d))
  ]

  # profile_conditions
  psi = (np.array(rhon_array[0]), np.array(profiles_1d[0].grid.psi))
  Ip = (
      time_array,
      -1 * ids.global_quantities.ip,
  )
  # It is assumed the temperatures and density profiles are defined until rhon=1.
  # Validator will raise an error if rhon[-1]!= 1.
  T_e = (
      time_array,
      rhon_array,
      [
          profiles_1d[ti].electrons.temperature / 1e3
          for ti in range(len(time_array))
      ],
  )

  if len(profiles_1d[0].t_i_average) > 0:
    T_i = (
        time_array,
        rhon_array,
        [profiles_1d[ti].t_i_average / 1e3 for ti in range(len(time_array))],
    )
  else:
    t_i_average = [
        np.mean(
            [
                profiles_1d[ti].ion[iion].temperature
                for iion in range(len(profiles_1d[ti].ion))
            ],
            axis=0,
        )
        for ti in range(len(time_array))
    ]
    T_i = (
        time_array,
        rhon_array,
        t_i_average,
    )

  n_e = (
      time_array,
      rhon_array,
      [profiles_1d[ti].electrons.density for ti in range(len(time_array))],
  )

  # Map v_loop_lcfs in case it is used as bc for psi equation.
  if len(ids.global_quantities.v_loop) > 0:
    v_loop_lcfs = (
        time_array,
        ids.global_quantities.v_loop,
    )  # TODO: Check the sign for v_loop when it will be used.
  else:
    v_loop_lcfs = [0.0]

  # Plasma composition
  plasma_composition_dict = _get_plasma_composition_info(
      ids, time_array, rhon_array
  )

  return {
      "profile_conditions": {
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
      },
      "plasma_composition": {
          **plasma_composition_dict,
      },
  }


def _get_plasma_composition_info(
    ids, time_array, rhon_array
) -> Mapping[str, Any]:
  """Returns dict with args for plasma_composition config from a given ids.

  Loading IMAS data for plasma composition should only be used with n_e_ratios
  and n_e_ratios_Zeff impurity modes, not with fractions mode. The impurity
  mode needs to be specified explicitly after loading the IMAS data. In case
  n_e_ratios_Zeff is specified, one impurity ratio must be set to None
  explicitly. In case n_e_ratios is used, Z_eff will simply be ignored.
  Note that if the ids indivual ions properties are not filled, it will not
  raise an error and just return an empty dict as main_ion and species.
  """
  profiles_1d = ids.profiles_1d
  Z_eff = (
      time_array,
      rhon_array,
      [profiles_1d[ti].zeff for ti in range(len(time_array))],
  )
  species = {}  # Impurity mapping {symbol: n_e_ratio,}.
  ratios = {}
  for iion in range(len(profiles_1d[0].ion)):
    try:
      symbol = str(profiles_1d[0].ion[iion].name)
    except (
        AttributeError
    ):  # Case ids is plasma_profiles in early DDv4 releases.
      symbol = str(profiles_1d[0].ion[iion].label)
    if symbol in constants.ION_PROPERTIES_DICT.keys():
      # Fill impurities
      if symbol not in ("D", "T", "H"):
        n_e_ratio = (
            time_array,
            rhon_array,
            [
                profiles_1d[ti].ion[iion].density
                / profiles_1d[ti].electrons.density
                for ti in range(len(time_array))
            ],
        )
        species[symbol] = n_e_ratio
      # Fill main ions
      else:
        ratios[symbol] = [
            profiles_1d[ti].ion[iion].density[0]
            for ti in range(len(time_array))
        ]
        # Currently take ratios of central density value, would it be more
        # accurate to take ratios of volume integrated densities ?
  total_main_ion_density = np.sum([ratio for ratio in ratios.values()], axis=0)
  main_ion = {}
  for symbol, ratio in ratios.items():
    main_ion[symbol] = (time_array, ratio / total_main_ion_density)
  return {
      "main_ion": main_ion,
      "Z_eff": Z_eff,
      "impurity": {
          "species": species,
      },
  }

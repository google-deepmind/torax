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

"""Useful functions to load IMAS core_profiles or plasma_profiles IDSs and
converts them into TORAX objects.
"""
import os
from typing import Any, Mapping

import imas
from imas import ids_toplevel
from imas.ids_toplevel import IDSToplevel
import numpy as np
import torax


def update_dict(old_dict: dict, updates: dict) -> dict:
  """Recursively modify the fields from the original dict old_dict using the values contained in updates dict.
  Used to update config dict fields more easily. Use case is to update config dict with output from core_profiles.core_profiles_from_IMAS().
  Args:
    old_dict: The current dict that needs to be updated.
    updates: Dict containing the values of the keys that need to be updated in old_dict.
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
        new_dict[key] = (
            value  # Needed to replace completely the time slices profiles, instead of keeping the initial ones.
        )
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
  read_psi_from_geo: Decides either to read psi from the geometry or from the
      input core/plasma_profiles IDS. Default value is True meaning that psi is
      taken from the geometry.
  t_initial: Initial time used to map the profiles in the dicts. If None the
      initial time will be the time of the first time slice of the ids. Else
      all time slices will be shifted by t_initial.

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
  psi = {
      t_initial: {
          rhon_array[0][rj]: profiles_1d[0].grid.psi[rj]
          for rj in range(len(rhon_array[0]))
      }
  }
  # Will be overwritten if Ip_from_parameters = False, when Ip is given by the equilibrium.
  Ip = {
      time_array[ti]: -1 * ids.global_quantities.ip[ti]
      for ti in range(len(time_array))
  }
  # It is assumed the temperatures and density profiles are defined until rhon=1.
  # Validator will raise an error if rhon[-1]!= 1.
  T_e = {
      time_array[ti]: {
          rhon_array[ti][rj]: profiles_1d[ti].electrons.temperature[rj] / 1e3
          for rj in range(len(rhon_array[ti]))
      }
      for ti in range(len(time_array))
  }

  if len(profiles_1d[0].t_i_average) > 0:
    T_i = {
        time_array[ti]: {
            rhon_array[ti][rj]: profiles_1d[ti].t_i_average[rj] / 1e3
            for rj in range(len(rhon_array[ti]))
        }
        for ti in range(len(time_array))
    }
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
    T_i = {
        time_array[ti]: {
            rhon_array[ti][rj]: t_i_average[rj] / 1e3
            for rj in range(len(rhon_array[ti]))
        }
        for ti in range(len(time_array))
    }

  n_e = {
      time_array[ti]: {
          rhon_array[ti][ri]: profiles_1d[ti].electrons.density[ri]
          for ri in range(len(rhon_array[ti]))
      }
      for ti in range(len(time_array))
  }

  if len(
      ids.global_quantities.v_loop > 0
  ):  # Map v_loop_lcfs in case it is used as bc for psi equation.
    v_loop_lcfs = {
        time_array[ti]: ids.global_quantities.v_loop[ti]
        for ti in range(len(time_array))
    }  # TODO: Check the sign for v_loop when it will be used.
  else:
    v_loop_lcfs = [0.0]

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
  }


def load_core_profiles_data(
    uri: str,
    ids_name: str,
    directory: str | None = None,
) -> ids_toplevel.IDSToplevel:
  """Loads a full IDS for a given uri or path_name and a given ids_name which
  should be either core_profiles or plasma_profiles. It can load either an
  IMAS netCDF file with filename as uri and given directory or from an IMASdb
  by giving the full uri of the IDS and the directory arg will be ignored.
  Note that loading from an IMASdb requires IMAS-core.
  The loaded IDS can then be used as input to core_profiles_from_IMAS().
  """
  # Differentiate between netCDF and IMASdb uris. For IMASdb files the full
  # filepath is already provided in the uri.
  if uri[-3:] == ".nc":
    if directory is None:
      directory = os.path.join(torax.__path__[0], "data/third_party/imas_data")
    uri = os.path.join(directory, uri)
  with imas.DBEntry(uri=uri, mode="r") as db:
    ids = db.get(ids_name=ids_name)
  return ids

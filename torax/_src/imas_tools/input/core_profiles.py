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
from typing import Any

import imas
from imas import ids_toplevel
from imas.ids_toplevel import IDSToplevel
import numpy as np
import torax


def core_profiles_from_IMAS(
    ids: IDSToplevel,
    read_psi_from_geo: bool = True,
    t_initial: float | None = None,
) -> dict:
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
  if not read_psi_from_geo:
    psi = {
        t_initial: {
            rhon_array[0][rj]: profiles_1d[0].grid.psi[rj]
            for rj in range(len(rhon_array[0]))
        }
    }
  else:
    psi = None
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
      'profile_conditions.Ip': Ip,
      'profile_conditions.psi': psi,
      'profile_conditions.T_i': T_i,
      'profile_conditions.T_i_right_bc': None,
      'profile_conditions.T_e': T_e,
      'profile_conditions.T_e_right_bc': None,
      'profile_conditions.n_e_right_bc_is_fGW': False,
      'profile_conditions.n_e_right_bc': None,
      'profile_conditions.n_e_nbar_is_fGW': False,
      'profile_conditions.nbar': None,
      'profile_conditions.n_e': n_e,
      'profile_conditions.normalize_n_e_to_nbar': False,
      'profile_conditions.v_loop_lcfs': v_loop_lcfs,
  }


def load_core_profiles_data(
    uri: str,
    ids_name: str,
    directory: str | None = None,
) -> ids_toplevel.IDSToplevel:
  """Loads a full IDS for a given uri or path_name and a given ids_name."""
  if directory is None:
    directory = os.path.join(torax.__path__[0], 'data/third_party/imas_data')
  uri = os.path.join(directory, uri)
  with imas.DBEntry(uri=uri, mode='r') as db:
    ids = db.get(ids_name=ids_name)
  return ids

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
from typing import Any

import numpy as np

try:
  from imas.ids_toplevel import IDSToplevel
except ImportError:
  IDSToplevel = Any


def core_profiles_from_IMAS(
    ids: IDSToplevel,
    read_psi_from_geo: bool = True,
) -> dict:
  """Converts core_profiles IDS to a dict with the input profiles for the config.
  Args:
  ids: IDS object. Can be either core_profiles or plasma_profiles. The IDS can contain several time slices.
  read_psi_from_geo: Decides either to read psi from the geometry or from the input core/plasma_profiles IDS. Default value is True meaning that psi is taken from the geometry.

  Returns:
  Dict containing the updated fields read from the IDS that need to be replaced in the input config.
  """
  profiles_1d = ids.profiles_1d
  time_array = [float(profiles_1d[i].time) for i in range(len(profiles_1d))]
  rhon_array = [
      profiles_1d[i].grid.rho_tor_norm for i in range(len(profiles_1d))
  ]
  # numerics
  t_initial = float(profiles_1d[0].time)

  # plasma_composition
  # Zeff taken from here or set into config before ?
  Z_eff = {
      time_array[ti]: {
          rhon_array[ti][rj]: profiles_1d[ti].zeff[rj]
          for rj in range(len(rhon_array[ti]))
      }
      for ti in range(len(time_array))
  }

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
  # It is assumed the temperatures and density profiles are defined until rhon=1. Validator will raise an error if rhon[-1]!= 1.
  T_e = {
      time_array[ti]: {
          rhon_array[ti][rj]: profiles_1d[ti].electrons.temperature[rj] / 1e3
          for rj in range(len(rhon_array[ti]))
      }
      for ti in range(len(time_array))
  }

  if len(profiles_1d[0].t_i_average > 0):
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
      'plasma_composition.Z_eff': Z_eff,
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
      'numerics.t_initial': t_initial,
      'numerics.t_final': (
          t_initial + 80.0
      ),  # How to define it ? Somewhere else ?
  }

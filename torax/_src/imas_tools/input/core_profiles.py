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

from imas import ids_toplevel
import numpy as np


def profile_conditions_from_IMAS(
    ids: ids_toplevel.IDSToplevel,
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
     Dict containing the updated fields read from the IDS that can be used to
     completely or partially to fill profile_conditions dict from a CONFIG dict
     via standard dict manipulations.
  """
  profiles_1d = ids.profiles_1d
  time_array = [profile.time for profile in profiles_1d]
  if t_initial:
    time_array = [ti - time_array[0] + t_initial for ti in time_array]
  else:
    t_initial = float(profiles_1d[0].time)
  rhon_array = [profile.grid.rho_tor_norm for profile in profiles_1d]

  # profile_conditions
  psi = (np.array(rhon_array[0]), np.array(profiles_1d[0].grid.psi))
  # Ip sign is switched due to the difference between input COCOS conventions
  # and TORAX ones
  Ip = (
      time_array,
      -1 * ids.global_quantities.ip,
  )

  # It is assumed the temperatures and density profiles are defined until rhon=1.
  # Validator will raise an error if rhon[-1]!= 1.
  # Temperatures are converted from eV (IMAS standard unit) to keV (TORAX units).
  T_e = (
      time_array,
      rhon_array,
      [profile.electrons.temperature / 1e3 for profile in profiles_1d],
  )

  if len(profiles_1d[0].t_i_average) > 0:
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
  if len(ids.global_quantities.v_loop) > 0:
    v_loop_lcfs = (
        time_array,
        ids.global_quantities.v_loop,
    )  # TODO: Check the sign for v_loop when it will be used.
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

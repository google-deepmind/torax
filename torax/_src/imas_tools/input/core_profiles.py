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
import logging
from typing import Any

from imas import ids_toplevel
import numpy as np
import torax._src.constants as constants

_PROFILE_CONDITIONS_REQUIRED_FIELDS = {
    "gloabl_quantities": ["ip", "v_loop"],
    "profiles_1d": [
        "time",
        "grid.rho_tor_norm",
        "grid.psi",
        "electrons.temperature",
        "t_i_average",
        "electrons.density",
    ],
}


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
  profiles_1d = ids.profiles_1d
  time_array = [profile.time for profile in profiles_1d]
  if t_initial:
    time_array = [ti - time_array[0] + t_initial for ti in time_array]
  rhon_array = [profile.grid.rho_tor_norm for profile in profiles_1d]

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

def _validate_core_profiles_ids_for_profile_conditions(
    ids: ids_toplevel.IDSToplevel,
):
  """Checks that all expected attributes exist in the IDS."""
  profiles_1d = ids.profiles_1d
  global_quantities = ids.global_quantities
  logged_fields = []
  for field in _PROFILE_CONDITIONS_REQUIRED_FIELDS["gloabl_quantities"]:
    if not getattr(global_quantities, field):
      # Warning or critical ?
      logging.critical(
          f"The IDS is missing global_quantities.{field} to build"
          " profile_conditions. \n Please Check that your IDS is properly"
          " filled."
      )
  for profile in profiles_1d:
    for field in _PROFILE_CONDITIONS_REQUIRED_FIELDS["profiles_1d"]:
      leaf = profile
      for node in field.split("."):
        leaf = getattr(leaf, node)
      if not leaf:
        if field not in logged_fields:
          # Warning or critical ?
          logging.critical(
              f"The IDS is missing profiles_1d.{field} to build"
              " profile_conditions. \n Please Check that your IDS is properly"
              " filled."
          )
          logged_fields.append(field)

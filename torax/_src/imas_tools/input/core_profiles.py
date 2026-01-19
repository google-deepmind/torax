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
from typing import Any

from imas import ids_toplevel
import numpy as np
from torax._src import constants
from torax._src.imas_tools.input import loader
from torax._src.imas_tools.input import validation


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
  validation.validate_core_profiles_ids(ids)
  profiles_1d, rhon_array, time_array = loader.get_time_and_radial_arrays(
      ids, t_initial
  )

  # profile_conditions
  psi = (time_array, rhon_array, [profile.grid.psi for profile in profiles_1d])
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
    excluded_impurities: Collection[str] | None = None,
    main_ions_symbols: Collection[str] | None = None,
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
    excluded_impurities: Optional arg to specify which impurities from the IDS
      should not be parsed.
    main_ions_symbols: collection of ions to be used to define the main_ion
      mixture. If value is not None, will check that the given ions
      exist in the IDS and their density is filled. If not explicitly provided,
      will parse H, D, T as main ions.

  Returns:
    The updated fields read from the IDS that can be used to completely or
    partially fill the `plasma_composition` section of a TORAX `CONFIG`.
  """
  profiles_1d, rhon_array, time_array = loader.get_time_and_radial_arrays(
      ids, t_initial
  )
  # Parse only ions with non empty density and not in excluded_impurities.
  if excluded_impurities:
    parsed_ions = [
        ion.name
        for ion in profiles_1d[0].ion
        if ion.density and ion.name not in excluded_impurities
    ]
  else:
    parsed_ions = [ion.name for ion in profiles_1d[0].ion if ion.density]
  # main_ions_symbols not explicitly provided: no validation of main ions and
  # value set to hydrogenic ions.
  if main_ions_symbols is None:
    main_ions_symbols = constants.HYDROGENIC_IONS
  # main_ions_symbols explicitly provided: validate ions presence in IDS.
  else:
    validation.validate_main_ions_presence(parsed_ions, main_ions_symbols)
  validation.validate_core_profiles_ions(parsed_ions)

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
    if symbol in parsed_ions:
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

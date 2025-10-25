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
"""Helpers for calculating main ion species outputs per main ion species."""

import dataclasses

import jax
import numpy as np
from torax._src import array_typing
from torax._src.config import runtime_params_slice
from torax._src.orchestration import sim_state as sim_state_lib
from torax._src.physics import charge_states
import xarray as xr

DENSITY_OUTPUT_NAME = 'n_main_ion_species'
Z_OUTPUT_NAME = 'Z_main_ion_species'
FRACTION_OUTPUT_NAME = 'main_ion_fractions'
MAIN_ION_DIM = 'main_ion_symbol'


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MainIonSpeciesOutput:
  """Main ion species output for a single species.

  Attributes:
    n_main_ion: The main ion density [m^-3].
    z_main_ion: The main ion average charge state.
    fraction: The main ion fractional abundance.
  """

  n_main_ion: array_typing.FloatVectorCell
  z_main_ion: array_typing.FloatVectorCell
  fraction: array_typing.FloatVectorCell


def calculate_main_ion_species_output(
    sim_state: sim_state_lib.ToraxSimState,
    runtime_params: runtime_params_slice.RuntimeParams,
) -> dict[str, MainIonSpeciesOutput]:
  """Calculates the output for each main ion species.

  Args:
    sim_state: The current state of the simulation.
    runtime_params: The runtime parameters for the current simulation slice.

  Returns:
    A dictionary mapping main ion symbols to their MainIonSpeciesOutput.
  """
  main_ion_species_output: dict[str, MainIonSpeciesOutput] = {}

  main_ion_names = runtime_params.plasma_composition.main_ion_names
  main_ion_fractions = runtime_params.plasma_composition.main_ion.fractions

  charge_state_info = charge_states.get_average_charge_state(
      T_e=sim_state.core_profiles.T_e.value,
      fractions=main_ion_fractions,
      Z_override=runtime_params.plasma_composition.main_ion.Z_override,
  )

  for symbol in main_ion_names:
    core_profiles = sim_state.core_profiles

    n_main_ion = main_ion_fractions[symbol] * core_profiles.n_i.value
    z_main_ion = charge_state_info.Z_per_species[symbol]
    fraction = main_ion_fractions[symbol]

    main_ion_species_output[symbol] = MainIonSpeciesOutput(
        n_main_ion=n_main_ion,
        z_main_ion=z_main_ion,
        fraction=fraction,
    )

  return main_ion_species_output


def construct_xarray_for_main_ion_output(
    main_ion_species_outputs: dict[str, MainIonSpeciesOutput],
    times: jax.Array,
    rho_cell_norm: jax.Array,
    time_coord: str,
    rho_cell_norm_coord: str,
) -> dict[str, xr.DataArray]:
  """Constructs an xarray dictionary for main ion species output.

  Args:
    main_ion_species_outputs: Dictionary mapping main ion symbols to their
      MainIonSpeciesOutput data.
    times: Array of time values for the simulation.
    rho_cell_norm: Array of normalized radial coordinates.
    time_coord: Name of the time coordinate dimension.
    rho_cell_norm_coord: Name of the radial coordinate dimension.

  Returns:
    Dictionary mapping output variable names to xarray DataArrays containing
    the main ion species data with proper dimensions and coordinates.
  """
  # Handle empty main ion species case
  if not main_ion_species_outputs:
    # Return empty DataArrays with correct structure
    xr_dict = {
        DENSITY_OUTPUT_NAME: xr.DataArray(
            np.empty((0, len(times), len(rho_cell_norm))),
            dims=[MAIN_ION_DIM, time_coord, rho_cell_norm_coord],
            coords={
                MAIN_ION_DIM: [],
                time_coord: times,
                rho_cell_norm_coord: rho_cell_norm,
            },
            name=DENSITY_OUTPUT_NAME,
        ),
        Z_OUTPUT_NAME: xr.DataArray(
            np.empty((0, len(times), len(rho_cell_norm))),
            dims=[MAIN_ION_DIM, time_coord, rho_cell_norm_coord],
            coords={
                MAIN_ION_DIM: [],
                time_coord: times,
                rho_cell_norm_coord: rho_cell_norm,
            },
            name=Z_OUTPUT_NAME,
        ),
        FRACTION_OUTPUT_NAME: xr.DataArray(
            np.empty((0, len(times))),
            dims=[MAIN_ION_DIM, time_coord],
            coords={
                MAIN_ION_DIM: [],
                time_coord: times,
            },
            name=FRACTION_OUTPUT_NAME,
        ),
    }
    return xr_dict

  density_data = []
  z_data = []
  fraction_data = []
  main_ion_symbols = []

  for main_ion_symbol in main_ion_species_outputs:
    output = main_ion_species_outputs[main_ion_symbol]
    density_data.append(output.n_main_ion)
    z_data.append(output.z_main_ion)
    # Fraction is spatially constant, so take first rho point
    # It has shape (time,) or (time, rho) - we want (time,)
    if output.fraction.ndim == 2:
      fraction_data.append(output.fraction[:, 0])
    else:
      fraction_data.append(output.fraction)
    main_ion_symbols.append(main_ion_symbol)

  density_data = np.stack(density_data, axis=0)
  z_data = np.stack(z_data, axis=0)
  fraction_data = np.stack(fraction_data, axis=0)

  xr_dict = {
      DENSITY_OUTPUT_NAME: xr.DataArray(
          density_data,
          dims=[MAIN_ION_DIM, time_coord, rho_cell_norm_coord],
          coords={
              MAIN_ION_DIM: main_ion_symbols,
              time_coord: times,
              rho_cell_norm_coord: rho_cell_norm,
          },
          name=DENSITY_OUTPUT_NAME,
      ),
      Z_OUTPUT_NAME: xr.DataArray(
          z_data,
          dims=[MAIN_ION_DIM, time_coord, rho_cell_norm_coord],
          coords={
              MAIN_ION_DIM: main_ion_symbols,
              time_coord: times,
              rho_cell_norm_coord: rho_cell_norm,
          },
          name=Z_OUTPUT_NAME,
      ),
      FRACTION_OUTPUT_NAME: xr.DataArray(
          fraction_data,
          dims=[MAIN_ION_DIM, time_coord],
          coords={
              MAIN_ION_DIM: main_ion_symbols,
              time_coord: times,
          },
          name=FRACTION_OUTPUT_NAME,
      ),
  }

  return xr_dict
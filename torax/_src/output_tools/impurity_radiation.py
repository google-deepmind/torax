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
"""Helpers for calculating impurity radiation outputs per impurity species."""

import dataclasses

import jax
from jax import numpy as jnp
import numpy as np
from torax._src import array_typing
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.orchestration import sim_state as sim_state_lib
from torax._src.physics import charge_states
from torax._src.sources import runtime_params as source_runtime_params_lib
from torax._src.sources.impurity_radiation_heat_sink import impurity_radiation_heat_sink
from torax._src.sources.impurity_radiation_heat_sink import impurity_radiation_mavrin_fit
import xarray as xr


RADIATION_OUTPUT_NAME = "radiation_impurity_species"
DENSITY_OUTPUT_NAME = "n_impurity_species"
Z_OUTPUT_NAME = "Z_impurity_species"
IMPURITY_DIM = "impurity_symbol"


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ImpuritySpeciesOutput:
  """Impurity radiation output for a single species.

  Attributes:
    radiation: The impurity radiation for the species [Wm^-3].
    n_impurity: The impurity density [m^-3].
    Z_impurity: The impurity average charge state.
  """

  radiation: array_typing.FloatVectorCell
  n_impurity: array_typing.FloatVectorCell
  Z_impurity: array_typing.FloatVectorCell


def calculate_impurity_species_output(
    sim_state: sim_state_lib.SimState,
    runtime_params: runtime_params_lib.RuntimeParams,
) -> dict[str, ImpuritySpeciesOutput]:
  """Calculates the output for each impurity species.

  Args:
    sim_state: The current state of the simulation.
    runtime_params: The runtime parameters for the current simulation slice.

  Returns:
    A dictionary mapping impurity symbols to their `ImpuritySpeciesOutput`.
    If the mavrin impurity radiation heat sink is not enabled then we output
    zeros for the radiation output.
  """
  impurity_species_output: dict[str, ImpuritySpeciesOutput] = {}
  mavrin_active = True
  # If the impurity radiation heat sink is not enabled, return empty dictionary.
  if (
      impurity_radiation_heat_sink.ImpurityRadiationHeatSink.SOURCE_NAME
      not in runtime_params.sources
  ):
    mavrin_active = False
  else:
    runtime_params_impurity = runtime_params.sources[
        impurity_radiation_heat_sink.ImpurityRadiationHeatSink.SOURCE_NAME
    ]
    # If the impurity radiation heat sink is not the mavrin model and in model
    # based mode, return empty dictionary.
    if not (
        isinstance(
            runtime_params_impurity, impurity_radiation_mavrin_fit.RuntimeParams
        )
        and runtime_params_impurity.mode
        == source_runtime_params_lib.Mode.MODEL_BASED
    ):
      mavrin_active = False

  impurity_fractions = sim_state.core_profiles.impurity_fractions
  impurity_names = runtime_params.plasma_composition.impurity_names
  charge_state_info = charge_states.get_average_charge_state(
      T_e=sim_state.core_profiles.T_e.value,
      fractions=impurity_fractions,
      Z_override=runtime_params.plasma_composition.impurity.Z_override,
  )

  for symbol in impurity_names:
    core_profiles = sim_state.core_profiles
    impurity_density_scaling = (
        core_profiles.Z_impurity / charge_state_info.Z_avg
    )
    n_imp = (
        impurity_fractions[symbol]
        * core_profiles.n_impurity.value
        * impurity_density_scaling
    )
    Z_imp = charge_state_info.Z_per_species[symbol]
    if mavrin_active:
      lz = impurity_radiation_mavrin_fit.calculate_impurity_radiation_single_species(
          core_profiles.T_e.value, symbol
      )
      radiation = n_imp * core_profiles.n_e.value * lz
    else:
      radiation = jnp.zeros_like(n_imp)
    impurity_species_output[symbol] = ImpuritySpeciesOutput(
        radiation=radiation, n_impurity=n_imp, Z_impurity=Z_imp
    )

  return impurity_species_output


def construct_xarray_for_radiation_output(
    impurity_radiation_outputs: dict[str, ImpuritySpeciesOutput],
    times: jax.Array,
    rho_cell_norm: jax.Array,
    time_coord: str,
    rho_cell_norm_coord: str,
) -> dict[str, xr.DataArray]:
  """Constructs an xarray dictionary for impurity radiation output."""

  radiation_data = []
  n_impurity_data = []
  Z_impurity_data = []
  impurity_symbols = []
  xr_dict = {}

  for impurity_symbol in impurity_radiation_outputs:
    radiation_data.append(impurity_radiation_outputs[impurity_symbol].radiation)
    n_impurity_data.append(
        impurity_radiation_outputs[impurity_symbol].n_impurity
    )
    Z_impurity_data.append(
        impurity_radiation_outputs[impurity_symbol].Z_impurity
    )
    impurity_symbols.append(impurity_symbol)
  radiation_data = np.stack(radiation_data, axis=0)
  n_impurity_data = np.stack(n_impurity_data, axis=0)
  Z_impurity_data = np.stack(Z_impurity_data, axis=0)
  xr_dict[RADIATION_OUTPUT_NAME] = xr.DataArray(
      radiation_data,
      dims=[IMPURITY_DIM, time_coord, rho_cell_norm_coord],
      coords={
          IMPURITY_DIM: impurity_symbols,
          time_coord: times,
          rho_cell_norm_coord: rho_cell_norm,
      },
      name=RADIATION_OUTPUT_NAME,
  )
  xr_dict[DENSITY_OUTPUT_NAME] = xr.DataArray(
      n_impurity_data,
      dims=[IMPURITY_DIM, time_coord, rho_cell_norm_coord],
      coords={
          IMPURITY_DIM: impurity_symbols,
          time_coord: times,
          rho_cell_norm_coord: rho_cell_norm,
      },
      name=DENSITY_OUTPUT_NAME,
  )
  xr_dict[Z_OUTPUT_NAME] = xr.DataArray(
      Z_impurity_data,
      dims=[IMPURITY_DIM, time_coord, rho_cell_norm_coord],
      coords={
          IMPURITY_DIM: impurity_symbols,
          time_coord: times,
          rho_cell_norm_coord: rho_cell_norm,
      },
      name=Z_OUTPUT_NAME,
  )
  return xr_dict

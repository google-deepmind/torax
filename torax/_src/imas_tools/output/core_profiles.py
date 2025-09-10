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

"""Useful functions to save IMAS core_profiles or plasma_profiles IDSs from
TORAX objects."""
import datetime
from typing import Any

import jax.numpy as jnp
import numpy as np

try:
  import imas
  from imas.ids_toplevel import IDSToplevel
except ImportError:
  IDSToplevel = Any
from torax._src import array_typing
from torax._src import constants
from torax._src import state
from torax._src.config import plasma_composition
from torax._src.config import runtime_params_slice
from torax._src.geometry.geometry import face_to_cell
from torax._src.orchestration.sim_state import ToraxSimState
from torax._src.output_tools import post_processing
from torax._src.physics import charge_states
from torax._src.torax_pydantic import model_config


# TODO: Add option to save entire state history in one core_profiles output.
def core_profiles_to_IMAS(
    config: model_config.ToraxConfig,
    runtime_params_slice: runtime_params_slice.RuntimeParams,
    post_processed_outputs: post_processing.PostProcessedOutputs,
    state: ToraxSimState,
    ids: IDSToplevel = imas.IDSFactory().core_profiles(),
) -> IDSToplevel:
  """Converts torax core_profiles to IMAS IDS.
  Takes the cell grid as a basis and converts values on face grid to cell.
  Args:
  config: TORAX config used for the simulation to get the names of the ions.
  dynamic_runtime_params_slice: Used to get the ions fractions for the time slice used.
  post_processed_outputs: TORAX post_processed_outputs with many useful data to output to the IDS.
  state: A ToraxSimState object.
  ids: Optional IDS object to be filled. Can be either core_profiles or plasma_profiles. Default is an empty core_profiles IDS. Note that both exists currently from Data Dictionary version 4, with plasma_profiles being the union of core_profiles and edge_profiles.

  Returns:
    Filled core_profiles or plas_profiles IDS object."""
  t = state.t
  cp_state = state.core_profiles
  cs_state = state.core_sources
  geometry = state.geometry
  ids.ids_properties.comment = (
      'IDS built from TORAX sim output. Grid based on torax cell grid +'
      ' boundaries.'
  )
  ids.ids_properties.homogeneous_time = 1
  ids.ids_properties.creation_date = datetime.date.today().isoformat()
  ids.time = [t]
  ids.code.name = 'TORAX'
  ids.code.description = (
      'TORAX is a differentiable tokamak core transport simulator aimed for'
      ' fast and accurate forward modelling, pulse-design, trajectory'
      ' optimization, and controller design workflows.'
  )
  ids.code.repository = 'https://github.com/google-deepmind/torax'
  ids.vacuum_toroidal_field.b0.resize(1)
  ids.global_quantities.current_bootstrap.resize(1)
  ids.global_quantities.ip.resize(1)
  ids.global_quantities.v_loop.resize(1)
  ids.global_quantities.li_3.resize(1)
  ids.global_quantities.beta_pol.resize(1)
  ids.global_quantities.beta_tor.resize(1)
  ids.global_quantities.beta_tor_norm.resize(1)
  ids.global_quantities.t_e_volume_average.resize(1)
  ids.global_quantities.n_e_volume_average.resize(1)

  ids.profiles_1d.resize(1)
  ids.profiles_1d[0].time = t

  ids.vacuum_toroidal_field.r0 = geometry.R_major
  ids.vacuum_toroidal_field.b0[0] = (
      geometry.B_0
  )  # TODO: Check sign(s) once TORAX COCOS will be defined.

  ids.global_quantities.ip[0] = -1 * cp_state.Ip_profile_face[-1]
  ids.global_quantities.current_bootstrap[0] = (
      -1 * post_processed_outputs.I_bootstrap
  )
  ids.global_quantities.v_loop[0] = cp_state.v_loop_lcfs
  ids.global_quantities.li_3[0] = post_processed_outputs.li3
  ids.global_quantities.beta_pol[0] = post_processed_outputs.beta_pol
  ids.global_quantities.beta_tor[0] = post_processed_outputs.beta_tor
  ids.global_quantities.beta_tor_norm[0] = post_processed_outputs.beta_N
  ids.global_quantities.t_e_volume_average[0] = (
      post_processed_outputs.T_e_volume_avg * 1e3
  )
  ids.global_quantities.n_e_volume_average[0] = (
      post_processed_outputs.n_e_volume_avg
  )
  ids.global_quantities.ion_time_slice = t

  ids.profiles_1d[0].grid.rho_tor_norm = np.concatenate(
      [[0.0], geometry.rho_norm, [1.0]]
  )
  ids.profiles_1d[0].grid.rho_tor = np.concatenate(
      [[0.0], geometry.rho, [geometry.rho_b]]
  )
  ids.profiles_1d[0].grid.psi = cp_state.psi.cell_plus_boundaries()
  ids.profiles_1d[0].grid.psi_magnetic_axis = cp_state.psi._left_face_value()[0]
  ids.profiles_1d[0].grid.psi_boundary = cp_state.psi._right_face_value()[0]
  ids.profiles_1d[0].grid.rho_pol_norm = np.sqrt(
      (cp_state.psi.cell_plus_boundaries() - cp_state.psi._left_face_value()[0])
      / (
          cp_state.psi._right_face_value()[0]
          - cp_state.psi._left_face_value()[0]
      )
  )
  volume = np.concatenate(
      [[geometry.volume_face[0]], geometry.volume, [geometry.volume_face[-1]]]
  )
  area = np.concatenate(
      [[geometry.area_face[0]], geometry.area, [geometry.area_face[-1]]]
  )
  ids.profiles_1d[0].grid.volume = volume
  ids.profiles_1d[0].grid.area = area

  ids.profiles_1d[0].electrons.temperature = (
      cp_state.T_e.cell_plus_boundaries() * 1e3
  )
  ids.profiles_1d[0].electrons.density = cp_state.n_e.cell_plus_boundaries()
  ids.profiles_1d[0].electrons.density_thermal = (
      cp_state.n_e.cell_plus_boundaries()
  )
  ids.profiles_1d[0].electrons.density_fast = np.zeros(
      len(ids.profiles_1d[0].grid.rho_tor_norm)
  )
  ids.profiles_1d[0].electrons.pressure_thermal = (
      post_processed_outputs.pressure_thermal_e.cell_plus_boundaries()
  )
  ids.profiles_1d[0].pressure_ion_total = (
      post_processed_outputs.pressure_thermal_i.cell_plus_boundaries()
  )
  ids.profiles_1d[0].pressure_thermal = (
      post_processed_outputs.pressure_thermal_total.cell_plus_boundaries()
  )
  ids.profiles_1d[0].t_i_average = cp_state.T_i.cell_plus_boundaries() * 1e3
  ids.profiles_1d[0].n_i_total_over_n_e = (
      cp_state.n_i.cell_plus_boundaries()
      + cp_state.n_impurity.cell_plus_boundaries()
  ) / cp_state.n_e.cell_plus_boundaries()
  Z_eff = np.concatenate(
      [[cp_state.Z_eff_face[0]], cp_state.Z_eff, [cp_state.Z_eff_face[-1]]]
  )
  ids.profiles_1d[0].zeff = Z_eff

  main_ion = list(
      zip(
          config.plasma_composition.get_main_ion_names(),
          runtime_params_slice.plasma_composition.main_ion.fractions,
      )
  )
  impurity_symbols = runtime_params_slice.plasma_composition.impurity_names
  impurity_fractions_arr = jnp.stack(
      [cp_state.impurity_fractions[symbol] for symbol in impurity_symbols]
  )
  impurities = list(zip(impurity_symbols, impurity_fractions_arr))
  impurity_density_scaling, Z_avg_per_species = (
      calculate_impurity_density_scaling_and_charge_states(
          cp_state, runtime_params_slice
      )
  )
  num_of_main_ions = len(main_ion)
  num_ions = num_of_main_ions + len(impurities)
  ids.profiles_1d[0].ion.resize(num_ions)
  ids.global_quantities.ion.resize(num_ions)
  # Fill main ions quantities
  for iion in range(len(main_ion)):
    symbol, frac = main_ion[iion]
    ion_properties = constants.ION_PROPERTIES_DICT[symbol]
    if (
        ids.metadata.name == 'core_profiles'
    ):  # Temporary if, will be removed in future versions where the path should be the same for both core and plasma_profiles.
      ids.profiles_1d[0].ion[iion].name = symbol
    else:
      ids.profiles_1d[0].ion[iion].label = symbol
    ids.profiles_1d[0].ion[iion].temperature = (
        cp_state.T_i.cell_plus_boundaries() * 1e3
    )
    ids.profiles_1d[0].ion[iion].density = (
        cp_state.n_i.cell_plus_boundaries() * frac
    )
    ids.profiles_1d[0].ion[iion].density_thermal = (
        cp_state.n_i.cell_plus_boundaries() * frac
    )
    ids.profiles_1d[0].ion[iion].density_fast = np.zeros(
        len(ids.profiles_1d[0].grid.rho_tor_norm)
    )
    total_ions_mixture_fraction = (
        frac
        * cp_state.n_i.cell_plus_boundaries()
        / (
            cp_state.n_i.cell_plus_boundaries()
            + cp_state.n_impurity.cell_plus_boundaries()
            * impurity_density_scaling  # Access true total impurity density
        )
    )  # Proportion of this ion among total ions species for pressure ratio computation.
    ids.profiles_1d[0].ion[iion].pressure = (
        total_ions_mixture_fraction
        * post_processed_outputs.pressure_thermal_i.cell_plus_boundaries()
    )
    ids.profiles_1d[0].ion[iion].pressure_thermal = (
        total_ions_mixture_fraction
        * post_processed_outputs.pressure_thermal_i.cell_plus_boundaries()
    )
    ids.profiles_1d[0].ion[iion].element.resize(1)
    ids.profiles_1d[0].ion[iion].element[0].a = ion_properties.A
    ids.profiles_1d[0].ion[iion].element[0].z_n = ion_properties.Z

    ids.global_quantities.ion[iion].t_i_volume_average.resize(1)
    ids.global_quantities.ion[iion].n_i_volume_average.resize(1)
    ids.global_quantities.ion[iion].t_i_volume_average[0] = (
        post_processed_outputs.T_i_volume_avg * 1e3
    )
    ids.global_quantities.ion[iion].n_i_volume_average[0] = (
        post_processed_outputs.n_i_volume_avg * frac
    )  # Valid to do like this ? Volume average ni only available for main ion.

  # Fill impurity quantities. Helper function is called when impurities array
  # is defined to access "true" impurity density and compute average charge
  # states for each impurity.
  for iion in range(len(impurities)):
    symbol, individual_frac = impurities[iion]
    frac = (
        np.concatenate(
            [[individual_frac[0]], individual_frac, [individual_frac[-1]]]
        )
        * impurity_density_scaling
    )  # Extend to cell_plus_boundaries_grid by copying neighbouring values.
    # Is there a better way to have it on cell_plus_boundaries grid ?
    ion_properties = constants.ION_PROPERTIES_DICT[symbol]
    # ids.profiles_1d[0].ion[num_of_main_ions+iion].z_ion = np.mean(cp_state.Z_impurity_face) # Change to make it correspond to volume average over plasma radius
    ids.profiles_1d[0].ion[num_of_main_ions + iion].z_ion_1d = (
        Z_avg_per_species[iion]
    )
    if (
        ids.metadata.name == 'core_profiles'
    ):  # Temporary if, will be removed in future versions where the path will
      # be name for both core and plasma_profiles. Should be fixed for DD versions >4.1
      ids.profiles_1d[0].ion[num_of_main_ions + iion].name = symbol
    else:
      ids.profiles_1d[0].ion[num_of_main_ions + iion].label = symbol
    ids.profiles_1d[0].ion[
        num_of_main_ions + iion
    ].temperature = cp_state.T_i.cell_plus_boundaries()
    ids.profiles_1d[0].ion[num_of_main_ions + iion].density = (
        cp_state.n_impurity.cell_plus_boundaries() * frac
    )
    ids.profiles_1d[0].ion[num_of_main_ions + iion].density_thermal = (
        cp_state.n_impurity.cell_plus_boundaries() * frac
    )
    ids.profiles_1d[0].ion[num_of_main_ions + iion].density_fast = np.zeros(
        len(ids.profiles_1d[0].grid.rho_tor_norm)
    )
    total_ions_mixture_fraction = (
        frac
        * cp_state.n_impurity.cell_plus_boundaries()
        / (
            cp_state.n_i.cell_plus_boundaries()
            + cp_state.n_impurity.cell_plus_boundaries()
            * impurity_density_scaling  # Access true total impurity density
        )
    )  # Proportion of this ion among total ions species for pressure ratio computation.
    ids.profiles_1d[0].ion[num_of_main_ions + iion].pressure = (
        total_ions_mixture_fraction
        * post_processed_outputs.pressure_thermal_i.cell_plus_boundaries()
    )
    ids.profiles_1d[0].ion[num_of_main_ions + iion].pressure_thermal = (
        total_ions_mixture_fraction
        * post_processed_outputs.pressure_thermal_i.cell_plus_boundaries()
    )
    ids.profiles_1d[0].ion[num_of_main_ions + iion].element.resize(1)
    ids.profiles_1d[0].ion[num_of_main_ions + iion].element[
        0
    ].a = ion_properties.A
    ids.profiles_1d[0].ion[num_of_main_ions + iion].element[
        0
    ].z_n = ion_properties.Z

    ids.global_quantities.ion[
        num_of_main_ions + iion
    ].t_i_volume_average.resize(1)
    ids.global_quantities.ion[num_of_main_ions + iion].t_i_volume_average[0] = (
        post_processed_outputs.T_i_volume_avg * 1e3
    )  # Volume average Ti and ni only available for main ion.

  q_cell = face_to_cell(cp_state.q_face)
  s_cell = face_to_cell(cp_state.s_face)
  ids.profiles_1d[0].q = np.concatenate(
      [[cp_state.q_face[0]], q_cell, [cp_state.q_face[-1]]]
  )
  ids.profiles_1d[0].magnetic_shear = np.concatenate(
      [[cp_state.s_face[0]], s_cell, [cp_state.s_face[-1]]]
  )
  j_total = np.concatenate([
      [cp_state.j_total_face[0]],
      cp_state.j_total,
      [cp_state.j_total_face[-1]],
  ])
  j_bootstrap = np.concatenate([
      [cs_state.bootstrap_current.j_bootstrap_face[0]],
      cs_state.bootstrap_current.j_bootstrap,
      [cs_state.bootstrap_current.j_bootstrap_face[-1]],
  ])
  ids.profiles_1d[0].j_total = -1 * j_total
  # TODO: Add consistent boundary values. They are not available for the moment for j_ohmic and jni so copied the neighbouring points values.
  j_ohmic = -1 * post_processed_outputs.j_ohmic
  ids.profiles_1d[0].j_ohmic = np.concatenate(
      [[j_ohmic[0]], j_ohmic, [j_ohmic[-1]]]
  )
  j_non_inductive = (
      -1 * sum(cs_state.psi.values()) + cs_state.bootstrap_current.j_bootstrap
  )
  ids.profiles_1d[0].j_non_inductive = -(
      np.concatenate(
          [[j_non_inductive[0]], j_non_inductive, [j_non_inductive[1]]]
      )
  )
  ids.profiles_1d[0].j_bootstrap = -1 * j_bootstrap
  sigma = np.concatenate(
      [[cp_state.sigma_face[0]], cp_state.sigma, [cp_state.sigma_face[-1]]]
  )
  ids.profiles_1d[0].conductivity_parallel = sigma
  return ids


def calculate_impurity_density_scaling_and_charge_states(
    core_profiles: state.CoreProfiles,
    runtime_params_slice: runtime_params_slice.RuntimeParams,
) -> tuple[array_typing.FloatVector, array_typing.FloatVector]:
  """Computes the impurity_density_scaling factor to compute "True" impurity density.
  Reproduces what is done in impurity_radiation_mavrin_fit in
  sources/impurity_radiation_heat_sink/impurity_radiation_mavrin_fit.py
  Also outputs Z_per_species to avoid calculating them again.
  Returns:
      FloatVector of impurity density scaling Z_imp_eff / <Z> on and
      FloatVector of avg Z_per_specie for all impurities on
      cell_plus_boundaries grid."""
  ion_symbols = runtime_params_slice.plasma_composition.impurity_names
  impurity_fractions_arr = jnp.stack([
      np.concatenate([
          [core_profiles.impurity_fractions[symbol][0]],
          core_profiles.impurity_fractions[symbol],
          [core_profiles.impurity_fractions[symbol][-1]],
      ])
      for symbol in ion_symbols
  ])  # Extend fractions to cell_plus_boundaries grid to compute everything on this grid directly
  ion_mixture = plasma_composition.DynamicIonMixture(
      fractions=impurity_fractions_arr,
      A_avg=core_profiles.A_impurity,
      Z_override=runtime_params_slice.plasma_composition.impurity.Z_override,
  )
  charge_state_info = charge_states.get_average_charge_state(
      ion_symbols=runtime_params_slice.plasma_composition.impurity_names,
      ion_mixture=ion_mixture,
      T_e=core_profiles.T_e.cell_plus_boundaries(),
  )
  Z_avg = charge_state_info.Z_avg
  Z_impurity = np.concatenate([
      [core_profiles.Z_impurity_face[0]],
      core_profiles.Z_impurity,
      [core_profiles.Z_impurity_face[-1]],
  ])
  impurity_density_scaling = Z_impurity / Z_avg
  return impurity_density_scaling, charge_state_info.Z_per_species

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

"""Functions to save IMAS core_profiles or plasma_profiles IDSs from TORAX."""

from collections.abc import Sequence
import datetime

import imas
from imas import ids_toplevel
import jax
import jaxtyping as jt
import numpy as np
from torax._src import array_typing
from torax._src import constants
from torax._src import math_utils
from torax._src import state
from torax._src.geometry import geometry as geometry_lib
from torax._src.output_tools import output
from torax._src.output_tools import post_processing
from torax._src.sources import source_profiles
from torax._src.torax_pydantic import model_config


# pylint: disable=invalid-name
def core_profiles_to_IMAS(
    torax_config: model_config.ToraxConfig,
    post_processed_outputs: Sequence[post_processing.PostProcessedOutputs],
    core_profiles: Sequence[state.CoreProfiles],
    core_sources: Sequence[source_profiles.SourceProfiles],
    geometry: Sequence[geometry_lib.Geometry],
    times: array_typing.FloatVector,
    ids: ids_toplevel.IDSToplevel | None = None,
) -> ids_toplevel.IDSToplevel:
  """Save TORAX profiles into an IMAS IDS.

  The output grid for all 1D quantities is the "cell_plus_boundaries".
  Quantities not available on boundaries are extrapolated by copying the values
  of the neighbouring points.
  The function can be used to save an entire trajectory or a single time slice.
  If you want to use this function programatically and save a single time
  slice, please make sure the inputs are `Sequence`s of length 1.


  Args:
    torax_config: ToraxConfig object to get number of main ions.
    post_processed_outputs: Sequence of TORAX PostProcessedOutputs objects.
    core_profiles: Sequence of TORAX CoreProfiles objects.
    core_sources: Sequence of TORAX SourceProfiles objects.
    geometry: Sequence of TORAX Geometry objects.
    times: Time array of the slices to save.
    ids: Optional IDS object to be filled. Can be either core_profiles or
      plasma_profiles. If not provided a core_profiles IDS will be created and
      output. Note that both exists currently from Data Dictionary version 4,
      with plasma_profiles being the merge of core_profiles and edge_profiles.

  Returns:
    Filled core_profiles or plasma_profiles IDS object.
    Signs used for currents and magnetic quantities might not be consistent
    with IMAS COCOS convention. This should be checked once TORAX COCOS will be
    defined.
  """
  if ids is None:
    ids = imas.IDSFactory().core_profiles()
  elif ids.metadata.name not in ['core_profiles', 'plasma_profiles']:
    raise TypeError(
        'Expected core_profiles or plasma_profiles IDS, got'
        f' {ids.metadata.name} IDS.'
    )

  _fill_metadata(ids)
  _fill_global_quantities(
      ids,
      torax_config,
      post_processed_outputs,
      core_profiles,
      geometry,
      times,
  )
  _fill_profiles_1d(
      ids,
      post_processed_outputs,
      core_profiles,
      core_sources,
      geometry,
      times,
  )
  return ids


def _fill_metadata(ids: ids_toplevel.IDSToplevel):
  """Fills metadata in-place for the IDS."""
  ids.ids_properties.comment = (
      'IDS built from TORAX simulation output. Grid based on torax cell grid +'
      ' boundaries.'
  )
  ids.ids_properties.homogeneous_time = 1
  ids.ids_properties.creation_date = datetime.date.today().isoformat()
  ids.code.name = 'TORAX'
  ids.code.description = (
      'TORAX is a differentiable tokamak core transport simulator aimed for'
      ' fast and accurate forward modelling, pulse-design, trajectory'
      ' optimization, and controller design workflows.'
  )
  ids.code.repository = 'https://github.com/google-deepmind/torax'


def _fill_global_quantities(
    ids: ids_toplevel.IDSToplevel,
    torax_config: model_config.ToraxConfig,
    post_processed_outputs: Sequence[post_processing.PostProcessedOutputs],
    core_profiles: Sequence[state.CoreProfiles],
    geometry: Sequence[geometry_lib.Geometry],
    times: array_typing.FloatVector,
) -> None:
  """Fills `global_quantities` in-place for the IDS."""
  ids.time = times
  ids.vacuum_toroidal_field.r0 = geometry[0].R_major
  # TODO(b/335204606): Check signs of B_0 and currents once TORAX COCOS defined.
  # Currents sign are flipped due to the difference between TORAX COCOS
  # convention and IMAS one.
  ids.vacuum_toroidal_field.b0 = [
      geometry_slice.B_0 for geometry_slice in geometry
  ]
  Ip = []
  v_loop = []
  I_bootstrap = []
  li_3 = []
  beta_pol = []
  beta_tor = []
  beta_tor_norm = []
  t_e_volume_average = []
  n_e_volume_average = []
  for cp_slice in core_profiles:
    Ip.append(-1 * cp_slice.Ip_profile_face[-1])
    v_loop.append(cp_slice.v_loop_lcfs)
  for ppo_slice in post_processed_outputs:
    I_bootstrap.append(-1 * ppo_slice.I_bootstrap)
    li_3.append(ppo_slice.li3)
    beta_pol.append(ppo_slice.beta_pol)
    beta_tor.append(ppo_slice.beta_tor)
    beta_tor_norm.append(ppo_slice.beta_N)
    # Temperatures are converted from keV to eV(IMAS standard unit).
    t_e_volume_average.append(ppo_slice.T_e_volume_avg * 1e3)
    n_e_volume_average.append(ppo_slice.n_e_volume_avg)
  ids.global_quantities.ip = Ip
  ids.global_quantities.current_bootstrap = I_bootstrap
  ids.global_quantities.v_loop = v_loop
  ids.global_quantities.li_3 = li_3
  ids.global_quantities.beta_pol = beta_pol
  ids.global_quantities.beta_tor = beta_tor
  ids.global_quantities.beta_tor_norm = beta_tor_norm
  ids.global_quantities.t_e_volume_average = t_e_volume_average
  ids.global_quantities.n_e_volume_average = n_e_volume_average
  ids.global_quantities.ion_time_slice = times[0]
  # TODO(b/459479939): i/1660) - Add main_ion information to outputs
  main_ion = torax_config.plasma_composition.main_ion.keys()
  impurities = post_processed_outputs[0].impurity_species.keys()
  num_ions = len(main_ion) + len(impurities)
  # Resize global_quantities.ion to create its substructure for every ion.
  ids.global_quantities.ion.resize(num_ions)
  for ion in range(num_ions):
    ids.global_quantities.ion[ion].t_i_volume_average.resize(len(times))
    ids.global_quantities.ion[ion].n_i_volume_average.resize(len(times))


def _fill_profiles_1d(
    ids: ids_toplevel.IDSToplevel,
    post_processed_outputs: Sequence[post_processing.PostProcessedOutputs],
    core_profiles: Sequence[state.CoreProfiles],
    core_sources: Sequence[source_profiles.SourceProfiles],
    geometry: Sequence[geometry_lib.Geometry],
    times: array_typing.FloatVector,
) -> None:
  """Fills `profiles_1d` section in-place for the IDS."""
  # Resize profiles_1d to create its substructure for the different slices.
  ids.profiles_1d.resize(len(times))
  for i in range(len(times)):
    t = times[i]
    geometry_slice = geometry[i]
    cp_state = core_profiles[i]
    cs_state = core_sources[i]
    post_processed_outputs_slice = post_processed_outputs[i]

    ids.profiles_1d[i].time = t
    _fill_profiles_1d_grid(ids, i, geometry_slice, cp_state)
    _fill_profiles_1d_currents(
        ids, i, post_processed_outputs_slice, cp_state, cs_state
    )
    # Temperatures are converted from keV to eV(IMAS standard unit).
    T_i = cp_state.T_i.cell_plus_boundaries() * 1e3
    n_e = cp_state.n_e.cell_plus_boundaries()
    n_i = cp_state.n_i.cell_plus_boundaries()
    n_impurity = cp_state.n_impurity.cell_plus_boundaries()
    pressure_thermal_i = cp_state.pressure_thermal_i.cell_plus_boundaries()
    ids.profiles_1d[i].t_i_average = T_i
    ids.profiles_1d[i].electrons.temperature = (
        cp_state.T_e.cell_plus_boundaries() * 1e3
    )
    ids.profiles_1d[i].electrons.density = n_e
    ids.profiles_1d[i].electrons.density_thermal = n_e
    ids.profiles_1d[i].electrons.density_fast = np.zeros(
        len(ids.profiles_1d[i].grid.rho_tor_norm)
    )
    ids.profiles_1d[i].electrons.pressure_thermal = (
        cp_state.pressure_thermal_e.cell_plus_boundaries()
    )
    ids.profiles_1d[i].pressure_ion_total = pressure_thermal_i
    ids.profiles_1d[i].pressure_thermal = (
        cp_state.pressure_thermal_total.cell_plus_boundaries()
    )
    _fill_profiles_1d_ions(
        ids,
        i,
        post_processed_outputs_slice,
        geometry_slice,
        cp_state,
        T_i,
        n_i,
        n_e,
        n_impurity,
        pressure_thermal_i,
    )
    Z_eff = output.extend_cell_grid_to_boundaries(
        [cp_state.Z_eff], np.array([cp_state.Z_eff_face])
    )[0]
    ids.profiles_1d[i].zeff = Z_eff


# TODO: Map impurity quantities in PostProcessedOutputs on cell_plus_boundaries
# grid to get rid of this function.
def _calculate_impurity_density_scaling_and_charge_states(
    core_profiles: state.CoreProfiles,
) -> tuple[array_typing.FloatVector, array_typing.FloatVector]:
  """Computes the impurity_density_scaling factor to compute "True" impurity density.

  Output quantities are on the cell_plus_boundaries grid.
  Reproduces what is done in impurity_radiation_mavrin_fit in
  sources/impurity_radiation_heat_sink/impurity_radiation_mavrin_fit.py
  Also outputs Z_per_species to avoid calculating them again.

  Returns:
      FloatVector of impurity density scaling Z_imp_eff / <Z> and
      FloatVector of avg Z_per_species for all impurities on
      cell_plus_boundaries grid.
  """
  charge_state_info = core_profiles.charge_state_info
  charge_state_info_face = core_profiles.charge_state_info_face
  Z_avg_per_species = {
      species: np.concatenate([
          [charge_state_info_face.Z_per_species[species][0]],
          Z_species_cell,
          [charge_state_info_face.Z_per_species[species][-1]],
      ])
      for species, Z_species_cell in charge_state_info.Z_per_species.items()
  }
  Z_avg = np.concatenate([
      [charge_state_info_face.Z_avg[0]],
      charge_state_info.Z_avg,
      [charge_state_info_face.Z_avg[-1]],
  ])

  Z_impurity = np.concatenate([
      [charge_state_info_face.Z_mixture[0]],
      charge_state_info.Z_mixture,
      [charge_state_info_face.Z_mixture[-1]],
  ])

  impurity_density_scaling = Z_impurity / Z_avg
  return impurity_density_scaling, Z_avg_per_species


def _fill_profiles_1d_grid(
    ids: ids_toplevel.IDSToplevel,
    i: int,
    geometry_slice: geometry_lib.Geometry,
    cp_state: state.CoreProfiles,
) -> None:
  """Fills `grid` section of `profiles_1d` in-place for the IDS."""
  ids.profiles_1d[i].grid.rho_tor_norm = np.concatenate(
      [[0.0], geometry_slice.rho_norm, [1.0]]
  )
  ids.profiles_1d[i].grid.rho_tor = np.concatenate(
      [[0.0], geometry_slice.rho, [geometry_slice.rho_b]]
  )
  ids.profiles_1d[i].grid.psi = cp_state.psi.cell_plus_boundaries()
  ids.profiles_1d[i].grid.psi_magnetic_axis = cp_state.psi.left_face_value()[0]
  ids.profiles_1d[i].grid.psi_boundary = cp_state.psi.right_face_value[0]
  ids.profiles_1d[i].grid.rho_pol_norm = np.sqrt(
      (cp_state.psi.cell_plus_boundaries() - cp_state.psi.left_face_value()[0])
      / (cp_state.psi.right_face_value[0] - cp_state.psi.left_face_value()[0])
  )
  ids.profiles_1d[i].grid.volume = output.extend_cell_grid_to_boundaries(
      [geometry_slice.volume], np.array([geometry_slice.volume_face])
  )[0]
  ids.profiles_1d[i].grid.area = output.extend_cell_grid_to_boundaries(
      [geometry_slice.area], np.array([geometry_slice.area_face])
  )[0]


def _fill_profiles_1d_currents(
    ids: ids_toplevel.IDSToplevel,
    i: int,
    post_processed_outputs_slice: post_processing.PostProcessedOutputs,
    cp_state: state.CoreProfiles,
    cs_state: source_profiles.SourceProfiles,
) -> None:
  q_cell = geometry_lib.face_to_cell(cp_state.q_face)
  s_cell = geometry_lib.face_to_cell(cp_state.s_face)
  ids.profiles_1d[i].q = output.extend_cell_grid_to_boundaries(
      [q_cell], np.array([cp_state.q_face])
  )[0]
  ids.profiles_1d[i].magnetic_shear = output.extend_cell_grid_to_boundaries(
      [s_cell], np.array([cp_state.s_face])
  )[0]
  # TODO(b/335204606): Clean this up once we finalize our COCOS convention.
  # Currents sign flipped due to the difference between TORAX COCOS convention
  # and IMAS one.
  # Total toroidal current density
  j_phi = output.extend_cell_grid_to_boundaries(
      [cp_state.j_total], np.array([cp_state.j_total_face])
  )[0]
  ids.profiles_1d[i].j_phi = -1 * j_phi
  # Parallel current densities
  j_bootstrap = output.extend_cell_grid_to_boundaries(
      [cs_state.bootstrap_current.j_parallel_bootstrap],
      np.array([cs_state.bootstrap_current.j_parallel_bootstrap_face]),
  )[0]
  ids.profiles_1d[i].j_bootstrap = -1 * j_bootstrap
  # TODO: Add consistent boundary values. They are not available for the moment
  # for j_ohmic, jni and parallel j_total so copied the neighbouring points
  # values.
  # Decision to make if we want to change it to spline or linear extrapolation,
  # or do it "properly" by calculating more strictly the face values of these
  # currents.
  j_parallel_total = -1 * post_processed_outputs_slice.j_parallel_total
  ids.profiles_1d[i].j_total = np.pad(j_parallel_total, (1, 1), mode='edge')

  j_ohmic = -1 * post_processed_outputs_slice.j_parallel_ohmic
  ids.profiles_1d[i].j_ohmic = np.pad(j_ohmic, (1, 1), mode='edge')

  j_non_inductive = -1 * post_processed_outputs_slice.j_parallel_non_inductive
  ids.profiles_1d[i].j_non_inductive = np.pad(
      j_non_inductive, (1, 1), mode='edge'
  )
  ids.profiles_1d[i].conductivity_parallel = (
      output.extend_cell_grid_to_boundaries(
          [cp_state.sigma], np.array([cp_state.sigma_face])
      )[0]
  )


def _fill_profiles_1d_ions(
    ids: ids_toplevel.IDSToplevel,
    i: int,
    post_processed_outputs_slice: post_processing.PostProcessedOutputs,
    geometry_slice: geometry_lib.Geometry,
    cp_state: state.CoreProfiles,
    T_i: jt.Float[jax.Array, 't* cell+2'],
    n_i: jt.Float[jax.Array, 't* cell+2'],
    n_e: jt.Float[jax.Array, 't* cell+2'],
    n_impurity: jt.Float[jax.Array, 't* cell+2'],
    pressure_thermal_i: jt.Float[jax.Array, 't* cell+2'],
) -> None:
  """Fills `ion` section of `profiles_1d` in-place for the IDS."""
  main_ion_fractions = cp_state.main_ion_fractions
  impurity_symbols = list(cp_state.impurity_fractions.keys())
  # Extend to cell_plus_boundaries grid by copying neighboring values.
  # TODO: As for currents, decide on exterpolation mode:
  # spline or linear, or do it " properly"
  impurity_fractions_arr = [
      np.pad(impurity_fraction, (1, 1), mode='edge')
      for impurity_fraction in cp_state.impurity_fractions.values()
  ]
  impurities = list(zip(impurity_symbols, impurity_fractions_arr, strict=True))
  # Computes impurity quantities on cell_plus_boundaries grid
  impurity_fractions_sum = np.sum(impurity_fractions_arr, axis=0)
  impurity_density_scaling, Z_avg_per_species = (
      _calculate_impurity_density_scaling_and_charge_states(cp_state)
  )
  n_impurity_true = (
      n_impurity * impurity_density_scaling * impurity_fractions_sum
  )
  # Index variables
  num_of_impurities = len(impurity_symbols)
  num_of_main_ions = len(main_ion_fractions)
  num_ions = num_of_main_ions + num_of_impurities
  ids.profiles_1d[i].ion.resize(num_ions)
  # Use the true n_impurity to get the real total ion density.
  ids.profiles_1d[i].n_i_total_over_n_e = (n_i + n_impurity_true) / n_e
  for ion, symbol in enumerate(cp_state.main_ion_fractions.keys()):
    frac = cp_state.main_ion_fractions[symbol]
    _fill_main_ions(
        ids,
        i,
        ion,
        symbol,
        frac,
        T_i,
        n_i,
        n_impurity_true,
        pressure_thermal_i,
        post_processed_outputs_slice,
    )

  # Helper function is called when impurities array is defined to access
  # "true" impurity density and compute impurities average charge states.
  for ion in range(num_of_impurities):
    _fill_impurities(
        ids,
        i,
        ion,
        T_i,
        n_i,
        n_impurity,
        n_impurity_true,
        pressure_thermal_i,
        Z_avg_per_species,
        impurity_density_scaling,
        num_of_main_ions,
        post_processed_outputs_slice,
        geometry_slice,
        impurities,
    )


def _fill_main_ions(
    ids: ids_toplevel.IDSToplevel,
    i: int,
    ion: int,
    symbol: str,
    frac: float,
    T_i: jt.Float[jax.Array, 't* cell+2'],
    n_i: jt.Float[jax.Array, 't* cell+2'],
    n_impurity_true: jt.Float[jax.Array, 't* cell+2'],
    pressure_thermal_i: jt.Float[jax.Array, 't* cell+2'],
    post_processed_outputs_slice: post_processing.PostProcessedOutputs,
) -> None:
  """Fills main ion quantities for the IDS."""
  ion_properties = constants.ION_PROPERTIES_DICT[symbol]
  # TODO(b/459479939): i/539) - Indicate supported dd_versions and switch on
  # that instead of using a try-except.
  try:
    ids.profiles_1d[i].ion[ion].name = symbol
  except AttributeError:
    # Case ids is plasma_profiles in early DDv4 releases.
    ids.profiles_1d[i].ion[ion].label = symbol
  ids.profiles_1d[i].ion[ion].temperature = T_i
  n_species = n_i * frac  # Individual density
  ids.profiles_1d[i].ion[ion].density = n_species
  ids.profiles_1d[i].ion[ion].density_thermal = n_species
  ids.profiles_1d[i].ion[ion].density_fast = np.zeros(
      len(ids.profiles_1d[i].grid.rho_tor_norm)
  )
  # Proportion of this ion for pressure ratio computation.
  total_ions_mixture_fraction = n_species / (n_i + n_impurity_true)
  ion_pressure = total_ions_mixture_fraction * pressure_thermal_i
  ids.profiles_1d[i].ion[ion].pressure = ion_pressure
  ids.profiles_1d[i].ion[ion].pressure_thermal = ion_pressure
  ids.profiles_1d[i].ion[ion].element.resize(1)
  ids.profiles_1d[i].ion[ion].element[0].a = ion_properties.A
  ids.profiles_1d[i].ion[ion].element[0].z_n = ion_properties.Z
  # Temperatures are converted from keV to eV(IMAS standard unit).
  ids.global_quantities.ion[ion].t_i_volume_average[i] = (
      post_processed_outputs_slice.T_i_volume_avg * 1e3
  )
  # frac is scalar for main ions so can just take fraction of volume_avg.
  ids.global_quantities.ion[ion].n_i_volume_average[i] = (
      post_processed_outputs_slice.n_i_volume_avg * frac
  )


def _fill_impurities(
    ids: ids_toplevel.IDSToplevel,
    i: int,
    ion: int,
    T_i: jt.Float[jax.Array, 't* cell+2'],
    n_i: jt.Float[jax.Array, 't* cell+2'],
    n_impurity_eff: jt.Float[jax.Array, 't* cell+2'],
    n_impurity_true: jt.Float[jax.Array, 't* cell+2'],
    pressure_thermal_i: jt.Float[jax.Array, 't* cell+2'],
    Z_avg_per_species: dict[str, jt.Float[jax.Array, 't* cell+2']],
    impurity_density_scaling: jt.Float[jax.Array, 't* cell+2'],
    num_of_main_ions: int,
    post_processed_outputs_slice: post_processing.PostProcessedOutputs,
    geometry_slice: geometry_lib.Geometry,
    impurities: Sequence[tuple[str, jt.Float[jax.Array, 't* cell+2']]],
) -> None:
  """Fills impurity quantities for the IDS."""
  # TODO(b/459479939): i/1660) - We can directly use the impurity densities
  # from post_processed_outputs.impurity_species.n_impurity here, if they get
  # extended to cell_plus_boundaries grid.
  symbol, individual_frac = impurities[ion]
  index = num_of_main_ions + ion
  scaled_frac = individual_frac * impurity_density_scaling
  ion_properties = constants.ION_PROPERTIES_DICT[symbol]
  ids.profiles_1d[i].ion[index].z_ion_1d = Z_avg_per_species[symbol]
  # TODO(b/459479939): i/539) - Indicate supported dd_versions and switch on
  # that instead of using a try-except.
  try:
    ids.profiles_1d[i].ion[index].name = symbol
  except AttributeError:
    # Case ids is plasma_profiles in early DDv4 releases.
    ids.profiles_1d[i].ion[index].label = symbol
  ids.profiles_1d[i].ion[index].temperature = T_i
  n_impurity_species = n_impurity_eff * scaled_frac
  ids.profiles_1d[i].ion[index].density = n_impurity_species
  ids.profiles_1d[i].ion[index].density_thermal = n_impurity_species
  ids.profiles_1d[i].ion[index].density_fast = np.zeros(
      len(ids.profiles_1d[i].grid.rho_tor_norm)
  )
  # Proportion of this ion for pressure ratio computation.
  total_ions_mixture_fraction = n_impurity_species / (n_i + n_impurity_true)
  impurity_pressure = total_ions_mixture_fraction * pressure_thermal_i
  ids.profiles_1d[i].ion[index].pressure = impurity_pressure
  ids.profiles_1d[i].ion[index].pressure_thermal = impurity_pressure
  ids.profiles_1d[i].ion[index].element.resize(1)
  ids.profiles_1d[i].ion[index].element[0].a = ion_properties.A
  ids.profiles_1d[i].ion[index].element[0].z_n = ion_properties.Z

  ids.global_quantities.ion[index].t_i_volume_average[i] = (
      post_processed_outputs_slice.T_i_volume_avg * 1e3
  )
  n_impurity_cell = post_processed_outputs_slice.impurity_species[
      symbol
  ].n_impurity
  ids.global_quantities.ion[index].n_i_volume_average[i] = (
      math_utils.volume_average(n_impurity_cell, geometry_slice)
  )

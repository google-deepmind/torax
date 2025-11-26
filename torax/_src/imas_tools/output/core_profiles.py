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

"""Useful functions to save IMAS core_profiles or plasma_profiles IDSs from
TORAX objects."""
from collections.abc import Iterable
import datetime

import imas
from imas import ids_toplevel
import numpy as np
from torax._src import array_typing
from torax._src import constants
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.config import runtime_params
from torax._src.geometry import geometry
from torax._src.geometry import geometry as geometry_module
from torax._src.output_tools import post_processing
from torax._src.physics import charge_states
from torax._src.sources import source_profiles


def core_profiles_to_IMAS(
    runtime_params_provider: build_runtime_params.RuntimeParamsProvider,
    post_processed_outputs: (
        post_processing.PostProcessedOutputs
        | list[post_processing.PostProcessedOutputs]
    ),
    core_profiles: state.CoreProfiles | list[state.CoreProfiles],
    core_sources: (
        source_profiles.SourceProfiles | list[source_profiles.SourceProfiles]
    ),
    geometry: geometry.Geometry | list[geometry.Geometry],
    times: float | array_typing.FloatVector,
    ids: ids_toplevel.IDSToplevel = None,
) -> ids_toplevel.IDSToplevel:
  """Save TORAX profiles into an IMAS core_profiles IDS.

  The output grid for all 1D quantities is the cell_plus_boundaries one. Values
  not available on boundaries are copied from neghbouring points. The function
  can be used to save an entire StateHistory or a single time slice.

  Args:
    runtime_params_provider: TORAX RuntimeParamsProvider to get the names of
      the ions and main_ion fractions.
    post_processed_outputs: Single occurence or list of TORAX
      PostProcessedOutputs object.
    core_profiles: Single occurence or list of TORAX CoreProfiles object.
    core_sources: Single occurence or list of TORAX SourceProfiles object.
    geometry: Single occurence or list of TORAX Geometry object.
    times: Time or list of times of the slices to save.
    ids: Optional IDS object to be filled. Can be either core_profiles or
      plasma_profiles. Default is an empty core_profiles IDS. Note that both
      exists currently from Data Dictionary version 4, with plasma_profiles
      being the merge of core_profiles and edge_profiles.

  Returns:
    Filled core_profiles or plasma_profiles IDS object.
  """

  def _ensure_list(arg):
    # Handle case receiving np.array with a single value
    if isinstance(arg, np.ndarray):
      arg = arg.tolist()
    return arg if isinstance(arg, Iterable) else [arg]

  if ids is None:
    ids = imas.IDSFactory().core_profiles()

  times = times.tolist()
  times = _ensure_list(times)
  core_profiles = _ensure_list(core_profiles)
  core_sources = _ensure_list(core_sources)
  geometry = _ensure_list(geometry)
  post_processed_outputs = _ensure_list(post_processed_outputs)

  _fill_metadata(ids)
  _fill_global_quantities(
      ids,
      runtime_params_provider,
      post_processed_outputs,
      core_profiles,
      geometry,
      times,
  )
  _fill_profiles_1d(
      ids,
      runtime_params_provider,
      post_processed_outputs,
      core_profiles,
      core_sources,
      geometry,
      times,
  )
  return ids


def _fill_metadata(ids: ids_toplevel.IDSToplevel) -> None:
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
    runtime_params_provider: build_runtime_params.RuntimeParamsProvider,
    post_processed_outputs: list[post_processing.PostProcessedOutputs],
    core_profiles: list[state.CoreProfiles],
    geometry: list[geometry.Geometry],
    times: array_typing.FloatVector,
) -> None:
  ids.time = times
  ids.vacuum_toroidal_field.r0 = geometry[0].R_major
  ids.vacuum_toroidal_field.b0 = [
      geometry_slice.B_0 for geometry_slice in geometry
  ]  # TODO: Check sign(s) once TORAX COCOS will be defined.
  ip = []
  v_loop = []
  I_bs = []
  li_3 = []
  beta_pol = []
  beta_tor = []
  beta_tor_norm = []
  t_e_volume_average = []
  n_e_volume_average = []
  for cp_slice in core_profiles:
    ip.append(-1 * cp_slice.Ip_profile_face[-1])
    v_loop.append(cp_slice.v_loop_lcfs)
  for ppo_slice in post_processed_outputs:
    I_bs.append(-1 * ppo_slice.I_bootstrap)
    li_3.append(ppo_slice.li3)
    beta_pol.append(ppo_slice.beta_pol)
    beta_tor.append(ppo_slice.beta_tor)
    beta_tor_norm.append(ppo_slice.beta_N)
    t_e_volume_average.append(ppo_slice.T_e_volume_avg * 1e3)
    n_e_volume_average.append(ppo_slice.n_e_volume_avg)
  ids.global_quantities.ip = ip
  ids.global_quantities.current_bootstrap = I_bs
  ids.global_quantities.v_loop = v_loop
  ids.global_quantities.li_3 = li_3
  ids.global_quantities.beta_pol = beta_pol
  ids.global_quantities.beta_tor = beta_tor
  ids.global_quantities.beta_tor_norm = beta_tor_norm
  # Temperatures are converted from keV to eV(IMAS standard unit).
  ids.global_quantities.t_e_volume_average = t_e_volume_average
  ids.global_quantities.n_e_volume_average = n_e_volume_average
  ids.global_quantities.ion_time_slice = times[0]
  main_ion = runtime_params_provider(
      times[0]
  ).plasma_composition.main_ion.fractions
  impurities = post_processed_outputs[0].impurity_species.keys()
  num_ions = len(main_ion) + len(impurities)
  ids.global_quantities.ion.resize(num_ions)
  for ion in range(num_ions):
    ids.global_quantities.ion[ion].t_i_volume_average.resize(len(times))
    ids.global_quantities.ion[ion].n_i_volume_average.resize(len(times))


def _fill_profiles_1d(
    ids: ids_toplevel.IDSToplevel,
    runtime_params_provider: build_runtime_params.RuntimeParamsProvider,
    post_processed_outputs: list[post_processing.PostProcessedOutputs],
    core_profiles: list[state.CoreProfiles],
    core_sources: list[source_profiles.SourceProfiles],
    geometry: list[geometry.Geometry],
    times: array_typing.FloatVector,
) -> None:
  ids.profiles_1d.resize(len(times))
  for i in range(len(times)):
    t = times[i]
    geometry_slice = geometry[i]
    cp_state = core_profiles[i]
    cs_state = core_sources[i]
    post_processed_outputs_slice = post_processed_outputs[i]
    runtime_params = runtime_params_provider(t)

    ids.profiles_1d[i].time = t
    _fill_grid(ids, i, geometry_slice, cp_state)
    _fill_ions(ids, i, runtime_params, post_processed_outputs_slice, cp_state)
    _fill_currents(ids, i, post_processed_outputs_slice, cp_state, cs_state)
    ids.profiles_1d[i].electrons.temperature = (
        cp_state.T_e.cell_plus_boundaries() * 1e3
    )
    ids.profiles_1d[i].electrons.density = cp_state.n_e.cell_plus_boundaries()
    ids.profiles_1d[i].electrons.density_thermal = (
        cp_state.n_e.cell_plus_boundaries()
    )
    ids.profiles_1d[i].electrons.density_fast = np.zeros(
        len(ids.profiles_1d[i].grid.rho_tor_norm)
    )
    ids.profiles_1d[i].electrons.pressure_thermal = (
        cp_state.pressure_thermal_e.cell_plus_boundaries()
    )
    ids.profiles_1d[i].pressure_ion_total = (
        cp_state.pressure_thermal_i.cell_plus_boundaries()
    )
    ids.profiles_1d[i].pressure_thermal = (
        cp_state.pressure_thermal_total.cell_plus_boundaries()
    )
    ids.profiles_1d[i].t_i_average = cp_state.T_i.cell_plus_boundaries() * 1e3
    ids.profiles_1d[i].n_i_total_over_n_e = (
        cp_state.n_i.cell_plus_boundaries()
        + cp_state.n_impurity.cell_plus_boundaries()
    ) / cp_state.n_e.cell_plus_boundaries()
    Z_eff = np.concatenate(
        [[cp_state.Z_eff_face[0]], cp_state.Z_eff, [cp_state.Z_eff_face[-1]]]
    )
    ids.profiles_1d[i].zeff = Z_eff


def _calculate_impurity_density_scaling_and_charge_states(
    core_profiles: state.CoreProfiles,
    runtime_params: runtime_params.RuntimeParams,
) -> tuple[array_typing.FloatVector, array_typing.FloatVector]:
  """Computes the impurity_density_scaling factor to compute "True" impurity density.

  Reproduces what is done in impurity_radiation_mavrin_fit in
  sources/impurity_radiation_heat_sink/impurity_radiation_mavrin_fit.py
  Also outputs Z_per_species to avoid calculating them again.

  Returns:
      FloatVector of impurity density scaling Z_imp_eff / <Z> on and
      FloatVector of avg Z_per_specie for all impurities on
      cell_plus_boundaries grid.
  """
  ion_symbols = runtime_params.plasma_composition.impurity_names
  # Extend fractions to cell_plus_boundaries grid to compute everything on this
  # grid directly.
  impurity_fractions = {
      symbol: np.concatenate([
          [core_profiles.impurity_fractions[symbol][0]],
          core_profiles.impurity_fractions[symbol],
          [core_profiles.impurity_fractions[symbol][-1]],
      ])
      for symbol in ion_symbols
  }

  charge_state_info = charge_states.get_average_charge_state(
      T_e=core_profiles.T_e.cell_plus_boundaries(),
      fractions=impurity_fractions,
      Z_override=runtime_params.plasma_composition.impurity.Z_override,
  )
  Z_avg = charge_state_info.Z_avg
  Z_impurity = np.concatenate([
      [core_profiles.Z_impurity_face[0]],
      core_profiles.Z_impurity,
      [core_profiles.Z_impurity_face[-1]],
  ])
  impurity_density_scaling = Z_impurity / Z_avg
  return impurity_density_scaling, charge_state_info.Z_per_species


def _fill_ions(
    ids: ids_toplevel.IDSToplevel,
    i: int,
    runtime_params: runtime_params.RuntimeParams,
    post_processed_outputs_slice: post_processing.PostProcessedOutputs,
    cp_state: state.CoreProfiles,
) -> None:
  def _fill_main_ion_quantities() -> None:
    ion_properties = constants.ION_PROPERTIES_DICT[symbol]
    # Temporary if, will be removed in future versions where the path
    # should be the same for both core and plasma_profiles.
    try:
      ids.profiles_1d[i].ion[ion].name = symbol
    except AttributeError:
      # TODO(b/459479939): i/539) - Indicate supported dd_versions and switch on
      # that instead of using a try-except.
      # Case ids is plasma_profiles in early DDv4 releases.
      ids.profiles_1d[i].ion[ion].label = symbol
    ids.profiles_1d[i].ion[ion].temperature = (
        cp_state.T_i.cell_plus_boundaries() * 1e3
    )
    ids.profiles_1d[i].ion[ion].density = (
        cp_state.n_i.cell_plus_boundaries() * frac
    )
    ids.profiles_1d[i].ion[ion].density_thermal = (
        cp_state.n_i.cell_plus_boundaries() * frac
    )
    ids.profiles_1d[i].ion[ion].density_fast = np.zeros(
        len(ids.profiles_1d[i].grid.rho_tor_norm)
    )
    # Proportion of this ion for pressure ratio computation.
    total_ions_mixture_fraction = (
        frac
        * cp_state.n_i.cell_plus_boundaries()
        / (
            cp_state.n_i.cell_plus_boundaries()
            + cp_state.n_impurity.cell_plus_boundaries()
            * impurity_density_scaling  # Access true total impurity density
        )
    )
    ids.profiles_1d[i].ion[ion].pressure = (
        total_ions_mixture_fraction
        * cp_state.pressure_thermal_i.cell_plus_boundaries()
    )
    ids.profiles_1d[i].ion[ion].pressure_thermal = (
        total_ions_mixture_fraction
        * cp_state.pressure_thermal_i.cell_plus_boundaries()
    )
    ids.profiles_1d[i].ion[ion].element.resize(1)
    ids.profiles_1d[i].ion[ion].element[0].a = ion_properties.A
    ids.profiles_1d[i].ion[ion].element[0].z_n = ion_properties.Z

    ids.global_quantities.ion[ion].t_i_volume_average[i] = (
        post_processed_outputs_slice.T_i_volume_avg * 1e3
    )
    ids.global_quantities.ion[ion].n_i_volume_average[i] = (
        post_processed_outputs_slice.n_i_volume_avg * frac
    )  # Valid to do like this ? Volume average ni only available for main ion.

  def _fill_impurity_quantities() -> None:
    symbol, individual_frac = impurities[ion]
    index = num_of_main_ions + ion
    frac = (
        np.concatenate(
            [[individual_frac[0]], individual_frac, [individual_frac[-1]]]
        )
        * impurity_density_scaling
    )  # Extend to cell_plus_boundaries_grid by copying neighbouring values.
    # Is there a better way to have it on cell_plus_boundaries grid ?
    ion_properties = constants.ION_PROPERTIES_DICT[symbol]
    ids.profiles_1d[i].ion[index].z_ion_1d = Z_avg_per_species[symbol]
    try:
      ids.profiles_1d[i].ion[index].name = symbol
    except AttributeError:
      # TODO(b/459479939): i/539) - Indicate supported dd_versions and switch on
      # that instead of using a try-except.
      # Case ids is plasma_profiles in early DDv4 releases.
      ids.profiles_1d[i].ion[index].label = symbol
    ids.profiles_1d[i].ion[
        index
    ].temperature = cp_state.T_i.cell_plus_boundaries()
    ids.profiles_1d[i].ion[index].density = (
        cp_state.n_impurity.cell_plus_boundaries() * frac
    )
    ids.profiles_1d[i].ion[index].density_thermal = (
        cp_state.n_impurity.cell_plus_boundaries() * frac
    )
    ids.profiles_1d[i].ion[index].density_fast = np.zeros(
        len(ids.profiles_1d[i].grid.rho_tor_norm)
    )
    # Proportion of this ion for pressure ratio computation.
    total_ions_mixture_fraction = (
        frac
        * cp_state.n_impurity.cell_plus_boundaries()
        / (
            cp_state.n_i.cell_plus_boundaries()
            + cp_state.n_impurity.cell_plus_boundaries()
            * impurity_density_scaling  # Access true total impurity density
        )
    )
    ids.profiles_1d[i].ion[index].pressure = (
        total_ions_mixture_fraction
        * cp_state.pressure_thermal_i.cell_plus_boundaries()
    )
    ids.profiles_1d[i].ion[index].pressure_thermal = (
        total_ions_mixture_fraction
        * cp_state.pressure_thermal_i.cell_plus_boundaries()
    )
    ids.profiles_1d[i].ion[index].element.resize(1)
    ids.profiles_1d[i].ion[index].element[0].a = ion_properties.A
    ids.profiles_1d[i].ion[index].element[0].z_n = ion_properties.Z

    ids.global_quantities.ion[index].t_i_volume_average[i] = (
        post_processed_outputs_slice.T_i_volume_avg * 1e3
    )  # Volume average Ti and ni only available for main ion.

  main_ion = runtime_params.plasma_composition.main_ion.fractions
  impurity_symbols = runtime_params.plasma_composition.impurity_names
  impurity_fractions_arr = np.stack(
      [cp_state.impurity_fractions[symbol] for symbol in impurity_symbols]
  )
  impurities = list(zip(impurity_symbols, impurity_fractions_arr))
  impurity_density_scaling, Z_avg_per_species = (
      _calculate_impurity_density_scaling_and_charge_states(
          cp_state, runtime_params
      )
  )
  num_of_main_ions = len(main_ion)
  num_ions = num_of_main_ions + len(impurities)
  ids.profiles_1d[i].ion.resize(num_ions)
  # Fill main ions quantities
  for ion, (symbol, frac) in enumerate(main_ion.items()):
    _fill_main_ion_quantities()

  # Fill impurity quantities.
  # Helper function is called when impurities array is defined to access
  # "true" impurity density and compute impurities average charge states.
  for ion in range(len(impurities)):
    _fill_impurity_quantities()


def _fill_currents(
    ids: ids_toplevel.IDSToplevel,
    i: int,
    post_processed_outputs_slice: post_processing.PostProcessedOutputs,
    cp_state: state.CoreProfiles,
    cs_state: source_profiles.SourceProfiles,
) -> None:
  q_cell = geometry_module.face_to_cell(cp_state.q_face)
  s_cell = geometry_module.face_to_cell(cp_state.s_face)
  ids.profiles_1d[i].q = np.concatenate(
      [[cp_state.q_face[0]], q_cell, [cp_state.q_face[-1]]]
  )
  ids.profiles_1d[i].magnetic_shear = np.concatenate(
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
  # TODO(b/335204606): Clean this up once we finalize our COCOS convention.
  # Currents sign flipped due to the difference between TORAX COCOS convention
  # and IMAS one.
  ids.profiles_1d[i].j_total = -1 * j_total
  # TODO: Add consistent boundary values. They are not available for the moment
  # for j_ohmic and jni so copied the neighbouring points values.
  j_ohmic = -1 * post_processed_outputs_slice.j_ohmic
  ids.profiles_1d[i].j_ohmic = np.concatenate(
      [[j_ohmic[0]], j_ohmic, [j_ohmic[-1]]]
  )
  j_non_inductive = (
      -1 * sum(cs_state.psi.values()) + cs_state.bootstrap_current.j_bootstrap
  )
  ids.profiles_1d[i].j_non_inductive = -(
      np.concatenate(
          [[j_non_inductive[0]], j_non_inductive, [j_non_inductive[1]]]
      )
  )
  ids.profiles_1d[i].j_bootstrap = -1 * j_bootstrap
  sigma = np.concatenate(
      [[cp_state.sigma_face[0]], cp_state.sigma, [cp_state.sigma_face[-1]]]
  )
  ids.profiles_1d[i].conductivity_parallel = sigma


def _fill_grid(
    ids: ids_toplevel.IDSToplevel,
    i: int,
    geometry_slice: geometry.Geometry,
    cp_state: state.CoreProfiles,
) -> None:
  ids.profiles_1d[i].grid.rho_tor_norm = np.concatenate(
      [[0.0], geometry_slice.rho_norm, [1.0]]
  )
  ids.profiles_1d[i].grid.rho_tor = np.concatenate(
      [[0.0], geometry_slice.rho, [geometry_slice.rho_b]]
  )
  ids.profiles_1d[i].grid.psi = cp_state.psi.cell_plus_boundaries()
  ids.profiles_1d[i].grid.psi_magnetic_axis = cp_state.psi._left_face_value()[0]
  ids.profiles_1d[i].grid.psi_boundary = cp_state.psi._right_face_value()[0]
  ids.profiles_1d[i].grid.rho_pol_norm = np.sqrt(
      (cp_state.psi.cell_plus_boundaries() - cp_state.psi._left_face_value()[0])
      / (
          cp_state.psi._right_face_value()[0]
          - cp_state.psi._left_face_value()[0]
      )
  )
  volume = np.concatenate([
      [geometry_slice.volume_face[0]],
      geometry_slice.volume,
      [geometry_slice.volume_face[-1]],
  ])
  area = np.concatenate([
      [geometry_slice.area_face[0]],
      geometry_slice.area,
      [geometry_slice.area_face[-1]],
  ])
  ids.profiles_1d[i].grid.volume = volume
  ids.profiles_1d[i].grid.area = area

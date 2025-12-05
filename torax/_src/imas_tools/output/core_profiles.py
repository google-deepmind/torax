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
from collections.abc import Sequence
import datetime

import imas
from imas import ids_toplevel
import jax
import jaxtyping as jt
import numpy as np
from torax._src import array_typing
from torax._src import constants
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.config import runtime_params
from torax._src.geometry import geometry
from torax._src.geometry import geometry as geometry_module
from torax._src.output_tools import output
from torax._src.output_tools import post_processing
from torax._src.physics import charge_states
from torax._src.sources import source_profiles
from torax._src.torax_pydantic import model_config


def core_profiles_to_IMAS(
    runtime_params_provider: build_runtime_params.RuntimeParamsProvider,
    torax_config: model_config.ToraxConfig,
    post_processed_outputs: Sequence[post_processing.PostProcessedOutputs],
    core_profiles: Sequence[state.CoreProfiles],
    core_sources: Sequence[source_profiles.SourceProfiles],
    geometry: Sequence[geometry.Geometry],
    times: array_typing.FloatVector,
    ids: ids_toplevel.IDSToplevel = None,
) -> ids_toplevel.IDSToplevel:
  """Save TORAX profiles into an IMAS core_profiles IDS.

  The output grid for all 1D quantities is the cell_plus_boundaries one. The
  function can be used to save an entire StateHistory or a single time slice.
  If you want to use this function programatically and save a single time
  slice, please make sure the inputs are Sequences.

  Args:
    runtime_params_provider: TORAX RuntimeParamsProvider to get the names of
      the ions and main_ion fractions.
    torax_config: ToraxConfig object to get number of main ions.
    post_processed_outputs: Sequence of TORAX PostProcessedOutputs objects.
    core_profiles: Sequence of TORAX CoreProfiles objects.
    core_sources: Sequence of TORAX SourceProfiles objects.
    geometry: Sequence of TORAX Geometry objects.
    times: Time array of the slices to save.
    ids: Optional IDS object to be filled. Can be either core_profiles or
      plasma_profiles. Default is an empty core_profiles IDS. Note that both
      exists currently from Data Dictionary version 4, with plasma_profiles
      being the merge of core_profiles and edge_profiles.

  Returns:
    Filled core_profiles or plasma_profiles IDS object.
    Signs used for currents and magnetic quantities might not be consistent
    with IMAS COCOS convention. This should be checked once TORAX COCOS will be
    defined.
  """
  if ids is None:
    ids = imas.IDSFactory().core_profiles()

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
    torax_config: model_config.ToraxConfig,
    post_processed_outputs: list[post_processing.PostProcessedOutputs],
    core_profiles: list[state.CoreProfiles],
    geometry: list[geometry.Geometry],
    times: array_typing.FloatVector,
) -> None:
  ids.time = times
  ids.vacuum_toroidal_field.r0 = geometry[0].R_major
  # TODO: Check signs for B_0 and currents once TORAX COCOS will be defined.
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
  # TODO: Add main_ion information to outputs
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
    runtime_params_provider: build_runtime_params.RuntimeParamsProvider,
    post_processed_outputs: list[post_processing.PostProcessedOutputs],
    core_profiles: list[state.CoreProfiles],
    core_sources: list[source_profiles.SourceProfiles],
    geometry: list[geometry.Geometry],
    times: array_typing.FloatVector,
) -> None:
  # Resize profiles_1d to create its substructure for the different slices.
  ids.profiles_1d.resize(len(times))
  for i in range(len(times)):
    t = times[i]
    geometry_slice = geometry[i]
    cp_state = core_profiles[i]
    cs_state = core_sources[i]
    post_processed_outputs_slice = post_processed_outputs[i]
    # TODO: Add main_ion information to outputs to avoid use of runtime_params
    runtime_params = runtime_params_provider(t)

    ids.profiles_1d[i].time = t
    _fill_grid(ids, i, geometry_slice, cp_state)
    _fill_currents(ids, i, cp_state, cs_state)
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
    ids.profiles_1d[i].n_i_total_over_n_e = (n_i + n_impurity) / n_e
    ids.profiles_1d[i].electrons.pressure_thermal = (
        cp_state.pressure_thermal_e.cell_plus_boundaries()
    )
    ids.profiles_1d[i].pressure_ion_total = pressure_thermal_i
    ids.profiles_1d[i].pressure_thermal = (
        cp_state.pressure_thermal_total.cell_plus_boundaries()
    )
    _fill_ions(
        ids,
        i,
        runtime_params,
        post_processed_outputs_slice,
        cp_state,
        T_i,
        n_i,
        n_impurity,
        pressure_thermal_i,
    )
    Z_eff = output.extend_cell_grid_to_boundaries(
        [cp_state.Z_eff], np.array([cp_state.Z_eff_face])
    )[0]
    ids.profiles_1d[i].zeff = Z_eff


def _fill_ions(
    ids: ids_toplevel.IDSToplevel,
    i: int,
    runtime_params: runtime_params.RuntimeParams,
    post_processed_outputs_slice: post_processing.PostProcessedOutputs,
    cp_state: state.CoreProfiles,
    T_i: jt.Float[jax.Array, 't* cell+2'],
    n_i: jt.Float[jax.Array, 't* cell+2'],
    n_impurity: jt.Float[jax.Array, 't* cell+2'],
    pressure_thermal_i: jt.Float[jax.Array, 't* cell+2'],
) -> None:
  def _fill_main_ion_quantities(
      i,
      ion,
      symbol,
      frac,
      T_i,
      n_i,
  ) -> None:
    ion_properties = constants.ION_PROPERTIES_DICT[symbol]
    # TODO(b/459479939): i/539) - Indicate supported dd_versions and switch on
    # that instead of using a try-except.
    try:
      ids.profiles_1d[i].ion[ion].name = symbol
    except AttributeError:
      # Case ids is plasma_profiles in early DDv4 releases.
      ids.profiles_1d[i].ion[ion].label = symbol
    ids.profiles_1d[i].ion[ion].temperature = T_i
    ids.profiles_1d[i].ion[ion].density = n_i * frac
    ids.profiles_1d[i].ion[ion].density_thermal = n_i * frac
    ids.profiles_1d[i].ion[ion].density_fast = np.zeros(
        len(ids.profiles_1d[i].grid.rho_tor_norm)
    )
    # TODO: Map pressure to IDS.
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

  def _fill_impurity_quantities(
      i,
      ion,
      T_i,
  ) -> None:
    symbol, _ = impurities[ion]
    index = num_of_main_ions + ion
    ion_properties = constants.ION_PROPERTIES_DICT[symbol]
    # TODO: Map ion.z_ion_1d
    # TODO(b/459479939): i/539) - Indicate supported dd_versions and switch on
    # that instead of using a try-except.
    try:
      ids.profiles_1d[i].ion[index].name = symbol
    except AttributeError:
      # Case ids is plasma_profiles in early DDv4 releases.
      ids.profiles_1d[i].ion[index].label = symbol
    ids.profiles_1d[i].ion[index].temperature = T_i
    # TODO: Map density from computed frac
    ids.profiles_1d[i].ion[index].density_fast = np.zeros(
        len(ids.profiles_1d[i].grid.rho_tor_norm)
    )
    # Proportion of this ion for pressure ratio computation.
    # TODO: Map pressure to output
    ids.profiles_1d[i].ion[index].element.resize(1)
    ids.profiles_1d[i].ion[index].element[0].a = ion_properties.A
    ids.profiles_1d[i].ion[index].element[0].z_n = ion_properties.Z

    ids.global_quantities.ion[index].t_i_volume_average[i] = (
        post_processed_outputs_slice.T_i_volume_avg * 1e3
    )
    # TODO: Compute n_i_volume_average by hand for impurities

  main_ion = runtime_params.plasma_composition.main_ion.fractions
  impurity_symbols = runtime_params.plasma_composition.impurity_names
  impurity_fractions_arr = np.stack(
      [cp_state.impurity_fractions[symbol] for symbol in impurity_symbols]
  )
  impurities = list(zip(impurity_symbols, impurity_fractions_arr))

  num_of_main_ions = len(main_ion)
  num_ions = num_of_main_ions + len(impurities)
  ids.profiles_1d[i].ion.resize(num_ions)
  # Fill main ions quantities
  for ion, (symbol, frac) in enumerate(main_ion.items()):
    _fill_main_ion_quantities(i, ion, symbol, frac, T_i, n_i)

  # Fill impurity quantities.
  # Helper function is called when impurities array is defined to access
  # "true" impurity density and compute impurities average charge states.
  for ion in range(len(impurities)):
    _fill_impurity_quantities(i, ion, T_i)


def _fill_currents(
    ids: ids_toplevel.IDSToplevel,
    i: int,
    cp_state: state.CoreProfiles,
    cs_state: source_profiles.SourceProfiles,
) -> None:
  q_cell = geometry_module.face_to_cell(cp_state.q_face)
  s_cell = geometry_module.face_to_cell(cp_state.s_face)
  ids.profiles_1d[i].q = output.extend_cell_grid_to_boundaries(
      [q_cell], np.array([cp_state.q_face])
  )[0]
  ids.profiles_1d[i].magnetic_shear = output.extend_cell_grid_to_boundaries(
      [s_cell], np.array([cp_state.s_face])
  )[0]
  j_total = output.extend_cell_grid_to_boundaries(
      [cp_state.j_total], np.array([cp_state.j_total_face])
  )[0]
  j_bootstrap = output.extend_cell_grid_to_boundaries(
      [cs_state.bootstrap_current.j_bootstrap],
      np.array([cs_state.bootstrap_current.j_bootstrap_face]),
  )[0]
  # TODO(b/335204606): Clean this up once we finalize our COCOS convention.
  # Currents sign flipped due to the difference between TORAX COCOS convention
  # and IMAS one.
  ids.profiles_1d[i].j_total = -1 * j_total
  ids.profiles_1d[i].j_bootstrap = -1 * j_bootstrap
  # TODO: Add j_ni and j_ohmic to output. Requires discussion on extending
  # values to cell_plus_boundaries grid.
  ids.profiles_1d[i].conductivity_parallel = (
      output.extend_cell_grid_to_boundaries(
          [cp_state.sigma], np.array([cp_state.sigma_face])
      )[0]
  )


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
  ids.profiles_1d[i].grid.psi_magnetic_axis = cp_state.psi.left_face_value()[0]
  ids.profiles_1d[i].grid.psi_boundary = cp_state.psi.right_face_value()[0]
  ids.profiles_1d[i].grid.rho_pol_norm = np.sqrt(
      (cp_state.psi.cell_plus_boundaries() - cp_state.psi.left_face_value()[0])
      / (cp_state.psi.right_face_value()[0] - cp_state.psi.left_face_value()[0])
  )
  ids.profiles_1d[i].grid.volume = output.extend_cell_grid_to_boundaries(
      [geometry_slice.volume], np.array([geometry_slice.volume_face])
  )[0]
  ids.profiles_1d[i].grid.area = output.extend_cell_grid_to_boundaries(
      [geometry_slice.area], np.array([geometry_slice.area_face])
  )[0]

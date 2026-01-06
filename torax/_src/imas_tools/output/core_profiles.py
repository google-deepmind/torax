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
from torax._src import math_utils
from torax._src import array_typing
from torax._src import constants
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry as geometry_lib
from torax._src.output_tools import output
from torax._src.output_tools import post_processing
from torax._src.sources import source_profiles
from torax._src.torax_pydantic import model_config


# pylint: disable=invalid-name
def core_profiles_to_IMAS(
    runtime_params_provider: build_runtime_params.RuntimeParamsProvider,
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
  The function can be used to save an entire trajectory or a single time slice.
  If you want to use this function programatically and save a single time
  slice, please make sure the inputs are `Sequence`s of length 1.

  Args:
    runtime_params_provider: TORAX RuntimeParamsProvider to get the names of the
      ions and main_ion fractions. This is currently needed as we don't have
      the main_ion information in the outputs.
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
    runtime_params_provider: build_runtime_params.RuntimeParamsProvider,
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
    # TODO(b/459479939): i/1660) - Add main_ion information to outputs
    # to avoid use of runtime_params
    runtime_params = runtime_params_provider(t)

    ids.profiles_1d[i].time = t
    _fill_profiles_1d_grid(ids, i, geometry_slice, cp_state)
    _fill_profiles_1d_currents(ids, i, cp_state, cs_state)
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
    # TODO(b/459479939): i/1660) - We should be using the sum of the real
    # impurity densities here (`n_impurity_species`) and not the "bulk" impurity
    # species `n_impurity`.
    ids.profiles_1d[i].n_i_total_over_n_e = (n_i + n_impurity) / n_e
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
        runtime_params,
        post_processed_outputs_slice,
        cp_state,
        geometry_slice,
        T_i,
        n_i,
    )
    Z_eff = output.extend_cell_grid_to_boundaries(
        [cp_state.Z_eff], np.array([cp_state.Z_eff_face])
    )[0]
    ids.profiles_1d[i].zeff = Z_eff


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


def _fill_profiles_1d_currents(
    ids: ids_toplevel.IDSToplevel,
    i: int,
    cp_state: state.CoreProfiles,
    cs_state: source_profiles.SourceProfiles,
) -> None:
  """Fills currents quantities of `profiles_1d` in-place for the IDS."""
  q_cell = geometry_lib.face_to_cell(cp_state.q_face)
  s_cell = geometry_lib.face_to_cell(cp_state.s_face)
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
  # TODO(b/459479939): i/1660) - Add j_ni and j_ohmic to output. Requires
  # discussion on extending values to cell_plus_boundaries grid.
  ids.profiles_1d[i].conductivity_parallel = (
      output.extend_cell_grid_to_boundaries(
          [cp_state.sigma], np.array([cp_state.sigma_face])
      )[0]
  )


def _fill_profiles_1d_ions(
    ids: ids_toplevel.IDSToplevel,
    i: int,
    runtime_params: runtime_params_lib.RuntimeParams,
    post_processed_outputs_slice: post_processing.PostProcessedOutputs,
    cp_state: state.CoreProfiles,
    geometry_slice: geometry_lib.Geometry,
    T_i: jt.Float[jax.Array, 't* cell+2'],
    n_i: jt.Float[jax.Array, 't* cell+2'],
) -> None:
  """Fills `ion` section of `profiles_1d` in-place for the IDS."""
  main_ion = runtime_params.plasma_composition.main_ion.fractions
  impurity_symbols = runtime_params.plasma_composition.impurity_names

  num_of_main_ions = len(main_ion)
  num_of_impurities = len(impurity_symbols)
  num_ions = num_of_main_ions + num_of_impurities
  ids.profiles_1d[i].ion.resize(num_ions)
  for ion, symbol in enumerate(main_ion.keys()):
    _fill_main_ions(
        ids, i, ion, symbol, T_i, n_i,cp_state, geometry_slice, post_processed_outputs_slice
    )

  # Helper function is called when impurities array is defined to access
  # "true" impurity density and compute impurities average charge states.
  for ion in range(num_of_impurities):
    _fill_impurities(
        ids,
        i,
        ion,
        T_i,
        num_of_main_ions,
        post_processed_outputs_slice,
        impurity_symbols,
        cp_state,
    )


def _fill_main_ions(
    ids: ids_toplevel.IDSToplevel,
    i: int,
    ion: int,
    symbol: str,
    T_i: jt.Float[jax.Array, 't* cell+2'],
    n_i: jt.Float[jax.Array, 't* cell+2'],
    cp_state: state.CoreProfiles,
    geometry_slice: geometry_lib.Geometry,
    post_processed_outputs_slice: post_processing.PostProcessedOutputs,
) -> None:
  """Fills main ion quantities for the IDS."""
  ion_properties = constants.ION_PROPERTIES_DICT[symbol]
  main_ion_frac = cp_state.main_ion_fractions[symbol]
  if main_ion_frac.ndim > 0:
    #main_ion_frac_extended = output.extend_cell_grid_to_boundaries(
       main_ion_frac_extended = np.concatenate([
      [main_ion_frac[0]],  
      main_ion_frac,       
      [main_ion_frac[-1]]  
    ])
  else:
  # Scalar case
    main_ion_frac_extended = main_ion_frac
  # TODO(b/459479939): i/539) - Indicate supported dd_versions and switch on
  # that instead of using a try-except.
  try:
    ids.profiles_1d[i].ion[ion].name = symbol
  except AttributeError:
    # Case ids is plasma_profiles in early DDv4 releases.
    ids.profiles_1d[i].ion[ion].label = symbol
  ids.profiles_1d[i].ion[ion].temperature = T_i
  ids.profiles_1d[i].ion[ion].density = n_i * main_ion_frac_extended
  ids.profiles_1d[i].ion[ion].density_thermal = n_i * main_ion_frac_extended
  ids.profiles_1d[i].ion[ion].density_fast = np.zeros(
      len(ids.profiles_1d[i].grid.rho_tor_norm)
  )
  # TODO(b/459479939): i/1814) - Map pressure to IDS.
  ids.profiles_1d[i].ion[ion].element.resize(1)
  ids.profiles_1d[i].ion[ion].element[0].a = ion_properties.A
  ids.profiles_1d[i].ion[ion].element[0].z_n = ion_properties.Z
  # Temperatures are converted from keV to eV(IMAS standard unit).
  ids.global_quantities.ion[ion].t_i_volume_average[i] = (
      post_processed_outputs_slice.T_i_volume_avg * 1e3
  )
  # Handle both scalar and profile main_ion_fractions for volume average
  if  main_ion_frac.ndim > 0:
    n_i_this_ion_cell = n_i[1:-1] * main_ion_frac
    n_i_this_ion_volume_avg = math_utils.volume_average(
        n_i_this_ion_cell, geometry_slice
    )
    ids.global_quantities.ion[ion].n_i_volume_average[i] = n_i_this_ion_volume_avg
  else:
  # frac is scalar for main ions so can just take fraction of volume_avg.
    ids.global_quantities.ion[ion].n_i_volume_average[i] = (
        post_processed_outputs_slice.n_i_volume_avg * main_ion_frac
    )


def _fill_impurities(
    ids: ids_toplevel.IDSToplevel,
    i: int,
    ion: int,
    T_i: jt.Float[jax.Array, 't* cell+2'],
    num_of_main_ions: int,
    post_processed_outputs_slice: post_processing.PostProcessedOutputs,
    impurity_symbols: Sequence[str],
    cp_state: state.CoreProfiles,
) -> None:
  """Fills impurity quantities for the IDS."""
  # TODO(b/459479939): i/1660) - We can directly use the impurity densities
  # from profiles.n_impurity_species here.
  impurity_fractions_arr = np.stack(
      [cp_state.impurity_fractions[symbol] for symbol in impurity_symbols]
  )
  impurities = list(zip(impurity_symbols, impurity_fractions_arr, strict=True))
  symbol, _ = impurities[ion]
  index = num_of_main_ions + ion
  ion_properties = constants.ION_PROPERTIES_DICT[symbol]
  # TODO(b/459479939): i/1814) - Map ion.z_ion_1d
  # TODO(b/459479939): i/539) - Indicate supported dd_versions and switch on
  # that instead of using a try-except.
  try:
    ids.profiles_1d[i].ion[index].name = symbol
  except AttributeError:
    # Case ids is plasma_profiles in early DDv4 releases.
    ids.profiles_1d[i].ion[index].label = symbol
  ids.profiles_1d[i].ion[index].temperature = T_i
  # TODO(b/459479939): i/1814) - Map density from computed frac
  ids.profiles_1d[i].ion[index].density_fast = np.zeros(
      len(ids.profiles_1d[i].grid.rho_tor_norm)
  )
  # Proportion of this ion for pressure ratio computation.
  # TODO(b/459479939): i/1814) - Map pressure to output
  ids.profiles_1d[i].ion[index].element.resize(1)
  ids.profiles_1d[i].ion[index].element[0].a = ion_properties.A
  ids.profiles_1d[i].ion[index].element[0].z_n = ion_properties.Z

  ids.global_quantities.ion[index].t_i_volume_average[i] = (
      post_processed_outputs_slice.T_i_volume_avg * 1e3
  )
  # TODO(b/459479939): i/1814) - Compute n_i_volume_average for impurities

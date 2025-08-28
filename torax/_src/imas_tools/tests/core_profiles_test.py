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

"""Unit tests for torax.torax_imastools.core_profiles.py"""
import os
from typing import Any, Optional

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

try:
  import imas
except ImportError:
  IDSToplevel = Any
import torax
from torax._src.geometry.imas import _load_imas_data
from torax._src.orchestration import run_loop
from torax._src.orchestration.run_simulation import prepare_simulation
from torax._src.test_utils import sim_test_case
from torax._src.torax_pydantic import model_config
from torax._src.imas_tools.core_profiles import core_profiles_from_IMAS
from torax._src.imas_tools.core_profiles import core_profiles_to_IMAS
from torax._src.config import build_runtime_params
from torax._src import state
from torax.tests.sim_test import _ALL_PROFILES


class CoreProfilesTest(sim_test_case.SimTestCase):
  """Integration Run with core_profiles from a reference run. To be
  integrated in sim_test_case probably."""

  def test_run_with_core_profiles_to_IMAS(
      self,
  ):
    """Test that TORAX simulation example can run with input core_profiles
    ids profiles, without raising error.
    The IMAS netCDF file was generated manually filling the core_profiles
    IDS based on the profile_conditions contained in the
    iterhybrid_rampup.py example config (after unit conversions).
    """

    # Input core_profiles reading and config loading
    config = self._get_config_dict('test_iterhybrid_rampup_short.py')

    path = 'core_profiles_ddv4_iterhybrid_rampup_conditions.nc'
    dir = os.path.join(torax.__path__[0], 'data/third_party/imas_data')
    core_profiles_in = _load_imas_data(
        path, 'core_profiles', geometry_directory=dir
    )

    # Modifying the input config profiles_conditions class
    core_profiles_conditions = core_profiles_from_IMAS(
        core_profiles_in, read_psi_from_geo=True
    )
    torax_config = model_config.ToraxConfig.from_dict(config)
    torax_config.update_fields(core_profiles_conditions)

    # Run Sim
    _, results = torax.run_simulation(torax_config)
    # Check that the simulation completed successfully.
    if results.sim_error != torax.SimError.NO_ERROR:
      raise ValueError(
          f'TORAX failed to run the simulation with error: {results.sim_error}.'
      )

  @parameterized.parameters([
      dict(config_name='test_iterhybrid_rampup_short.py', rtol=1e-6, atol=1e-8),
  ])
  def test_init_profiles_from_IMAS(
      self,
      config_name,
      rtol: Optional[float] = None,
      atol: Optional[float] = None,
  ):
    """Test to compare initialized profiles in TORAX with the initial
    core_profiles used to check consistency.
    The IMAS netCDF file comes from a METIS ITER baseline scenario time
    slice taken on the flat top. It is useful to have such file with
    profiles having more radial resolution. The IDS grid is made of 21
    points."""

    if rtol is None:
      rtol = self.rtol
    if atol is None:
      atol = self.atol
    # Input core_profiles reading and config loading
    config = self._get_config_dict(config_name)
    path = (  # Using this as input instead of rampup_conditions because it has more radial resolution.
        'core_profiles_15MA_DT_50_50_flat_top_slice.nc'
    )
    dir = os.path.join(torax.__path__[0], 'data/third_party/imas_data')
    core_profiles_in = _load_imas_data(
        path, 'core_profiles', geometry_directory=dir
    )
    rhon_in = core_profiles_in.profiles_1d[0].grid.rho_tor_norm

    # Modifying the input config profiles_conditions class
    core_profiles_conditions = core_profiles_from_IMAS(
        core_profiles_in, read_psi_from_geo=False
    )
    config['geometry']['n_rho'] = 200
    torax_config = model_config.ToraxConfig.from_dict(config)
    torax_config.update_fields(core_profiles_conditions)

    # Init sim from config
    _, sim_state, _, _ = prepare_simulation(torax_config)

    # Read output values
    torax_mesh = torax_config.geometry.build_provider.torax_mesh
    cell_centers = torax_mesh.cell_centers
    # Compare the initial core_profiles with the ids profiles
    init_core_profiles = sim_state.core_profiles
    np.testing.assert_allclose(
        init_core_profiles.T_e.value * 1e3,
        np.interp(
            cell_centers,
            rhon_in,
            core_profiles_in.profiles_1d[0].electrons.temperature,
        ),
        rtol=rtol,
        atol=atol,
        err_msg='Te profile failed',
    )
    np.testing.assert_allclose(
        init_core_profiles.T_i.value * 1e3,
        np.interp(
            cell_centers, rhon_in, core_profiles_in.profiles_1d[0].t_i_average
        ),
        rtol=rtol,
        atol=atol,
        err_msg='Ti profile failed',
    )
    np.testing.assert_allclose(
        init_core_profiles.n_e.value,
        np.interp(
            cell_centers,
            rhon_in,
            core_profiles_in.profiles_1d[0].electrons.density,
        ),
        rtol=rtol,
        atol=atol,
        err_msg='ne profile failed',
    )
    np.testing.assert_allclose(
        init_core_profiles.psi.value,
        np.interp(
            cell_centers, rhon_in, core_profiles_in.profiles_1d[0].grid.psi
        ),
        rtol=rtol,
        atol=atol,
        err_msg='psi profile failed',
    )

  @parameterized.parameters([
      dict(ids_out=imas.IDSFactory().core_profiles()),
      dict(ids_out=imas.IDSFactory().plasma_profiles()),
  ])
  def test_save_profiles_to_IMAS(
      self,
      ids_out,
  ):
    """Test to check that data can be written in output to the IDS, either core_profiles or plasma_profiles."""
    # Input core_profiles reading and config loading
    config = self._get_config_dict('test_iterhybrid_rampup_short.py')
    path = 'core_profiles_ddv4_iterhybrid_rampup_conditions.nc'
    dir = os.path.join(torax.__path__[0], 'data/third_party/imas_data')
    core_profiles_in = _load_imas_data(
        path, 'core_profiles', geometry_directory=dir
    )

    # Modifying the input config profiles_conditions class
    core_profiles_conditions = core_profiles_from_IMAS(
        core_profiles_in, read_psi_from_geo=False
    )
    config['geometry']['n_rho'] = 20
    torax_config = model_config.ToraxConfig.from_dict(config)
    torax_config.update_fields(core_profiles_conditions)

    # Init sim from config
    (
        dynamic_runtime_params_slice_provider,
        initial_state,
        post_processed_outputs,
        step_fn,
    ) = prepare_simulation(torax_config)

    state_history, post_processed_outputs_history, sim_error = (
        run_loop.run_loop(
            dynamic_runtime_params_slice_provider=dynamic_runtime_params_slice_provider,
            initial_state=initial_state,
            initial_post_processed_outputs=post_processed_outputs,
            step_fn=step_fn,
            log_timestep_info=False,
            progress_bar=False,
        )
    )

    if sim_error != torax.SimError.NO_ERROR:
      raise ValueError(
          f'TORAX failed to run the simulation with error: {sim_error}.'
      )
    post_processed_outputs = post_processed_outputs_history[-1]
    final_sim_state = state_history[-1]
    t_final = final_sim_state.t
    filled_ids = core_profiles_to_IMAS(
        torax_config,
        dynamic_runtime_params_slice_provider(t_final),
        post_processed_outputs,
        final_sim_state,
        ids_out,
    )
    filled_ids.validate()

  def test_round_trip_with_IMAS_profiles(
    self,
  ):
    """Test that TORAX simulation can run from a standard config, write into
    IMAS and load the profiles back and run the simulation without changes.
    The test: 
        -Run once with test_iterhybrid_predictor_corrector config,
        -Output the initial time slice into a core_profiles IDS, 
        -Update the initial config loading the profiles contained into the IDS, 
        -Run the simulation again with these new profiles and compare the 2 runs.
    """
    rtol = 1e-10
    atol = 0
    # Input core_profiles reading and config loading
    config = self._get_config_dict('test_iterhybrid_predictor_corrector.py')
    torax_config = model_config.ToraxConfig.from_dict(config)
    # Run Sim
    (
      dynamic_runtime_params_slice_provider,
      initial_state,
      post_processed_outputs,
      step_fn,
    ) = prepare_simulation(torax_config)

    state_history, post_processed_outputs_history, _ = (
        run_loop.run_loop(
            dynamic_runtime_params_slice_provider=dynamic_runtime_params_slice_provider,
            initial_state=initial_state,
            initial_post_processed_outputs=post_processed_outputs,
            step_fn=step_fn,
            log_timestep_info=False,
            progress_bar=False,
        )
    )
    #Fill an IDS with the output of the simulation, from the initial time step.
    sim_state = state_history[0]
    post_processed_outputs = post_processed_outputs_history[0]
    filled_ids = core_profiles_to_IMAS(
        torax_config,
        dynamic_runtime_params_slice_provider(0),
        post_processed_outputs,
        sim_state,
    )
    # Modifying the input config profiles_conditions class with the IDS filled
    # from the previous simulation.
    core_profiles_conditions = core_profiles_from_IMAS(
        filled_ids, read_psi_from_geo=False
    )
    core_profiles_conditions['numerics.t_final'] = 5 #Cheating for now until we
    # decide if the function should map time or not.
    torax_config.update_fields(core_profiles_conditions)
    #Running simulation again with the new config
    imas_xr, imas_results = torax.run_simulation(torax_config, progress_bar=False)

    self.assertEqual(imas_results.sim_error, state.SimError.NO_ERROR)

    ref_profiles, ref_time = self._get_refs('test_iterhybrid_predictor_corrector.nc', _ALL_PROFILES)

    self._check_profiles_vs_expected(
        t=imas_xr.time.values,
        ref_time=ref_time,
        ref_profiles=ref_profiles,
        rtol=rtol,
        atol=atol,
        output_file=None,
        ds=imas_xr,
        write_output=False,
    )




if __name__ == '__main__':
  absltest.main()

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

import os
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized

try:
  import imas
except ImportError:
  IDSToplevel = Any
import torax
from torax._src import state
from torax._src.geometry.imas import _load_imas_data
from torax._src.imas_tools.input.core_profiles import core_profiles_from_IMAS
from torax._src.imas_tools.output.core_profiles import core_profiles_to_IMAS
from torax._src.orchestration import run_loop
from torax._src.orchestration.run_simulation import prepare_simulation
from torax._src.test_utils import sim_test_case
from torax._src.torax_pydantic import model_config
from torax.tests.sim_test import _ALL_PROFILES


class CoreProfilesTest(sim_test_case.SimTestCase):
  """Unit tests for torax.torax_imastools.output.core_profiles.py"""

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

    state_history, post_processed_outputs_history, _ = run_loop.run_loop(
        dynamic_runtime_params_slice_provider=dynamic_runtime_params_slice_provider,
        initial_state=initial_state,
        initial_post_processed_outputs=post_processed_outputs,
        step_fn=step_fn,
        log_timestep_info=False,
        progress_bar=False,
    )
    # Fill an IDS with the output of the simulation, from the initial time step.
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
    core_profiles_conditions['numerics.t_final'] = (
        5  # Cheating for now until we
    )
    # decide if the function should map time or not.
    torax_config.update_fields(core_profiles_conditions)
    # Running simulation again with the new config
    imas_xr, imas_results = torax.run_simulation(
        torax_config, progress_bar=False
    )

    self.assertEqual(imas_results.sim_error, state.SimError.NO_ERROR)

    ref_profiles, ref_time = self._get_refs(
        'test_iterhybrid_predictor_corrector.nc', _ALL_PROFILES
    )

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

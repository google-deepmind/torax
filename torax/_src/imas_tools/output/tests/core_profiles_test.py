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

from typing import Final, Sequence

from absl.testing import absltest
from absl.testing import parameterized
import imas
import torax
from torax._src import state
from torax._src.imas_tools.input import core_profiles as input_core_profiles
from torax._src.imas_tools.output import core_profiles as output_core_profiles
from torax._src.orchestration import run_loop
from torax._src.orchestration import run_simulation
from torax._src.output_tools import output
from torax._src.test_utils import sim_test_case
from torax._src.torax_pydantic import model_config


_ALL_PROFILES: Final[Sequence[str]] = (
    output.T_I,
    output.T_E,
    output.PSI,
    output.Q,
    output.N_E,
)


class CoreProfilesTest(sim_test_case.SimTestCase):

  @parameterized.parameters([
      dict(ids_out=imas.IDSFactory().core_profiles()),
      dict(ids_out=imas.IDSFactory().plasma_profiles()),
  ])
  def test_save_profiles_to_IMAS(
      self,
      ids_out,
  ):
    """Test to check that multiple time slice can be saved into an IDS."""
    # Run sim
    config = self._get_config_dict('test_iterhybrid_rampup_short.py')
    torax_config = model_config.ToraxConfig.from_dict(config)
    (
        initial_state,
        post_processed_outputs,
        step_fn,
    ) = run_simulation.prepare_simulation(torax_config)

    state_history, post_processed_outputs_history, sim_error = (
        run_loop.run_loop(
            initial_state=initial_state,
            initial_post_processed_outputs=post_processed_outputs,
            step_fn=step_fn,
            log_timestep_info=False,
            progress_bar=False,
        )
    )
    state_history = output.StateHistory(
        state_history=state_history,
        post_processed_outputs_history=post_processed_outputs_history,
        sim_error=sim_error,
        torax_config=torax_config,
    )
    # Save output profiles to IDS
    post_processed_outputs = state_history.post_processed_outputs
    core_profiles = state_history.core_profiles
    core_sources = state_history.source_profiles
    geometry = state_history.geometries
    times = state_history.times
    filled_ids = output_core_profiles.core_profiles_to_IMAS(
        step_fn.runtime_params_provider,
        torax_config,
        post_processed_outputs,
        core_profiles,
        core_sources,
        geometry,
        times,
        ids_out,
    )
    filled_ids.validate()

  def test_round_trip_with_IMAS_profiles(
      self,
  ):
    """Tests a round trip of TORAX simulation with IMAS IDS.

    The test:
        -Runs once with test_iterhybrid_predictor_corrector config,
        -Outputs the initial time slice into a core_profiles IDS,
        -Updates the initial config from the IDS profiles,
        -Runs the simulation again and compare the 2 runs.
    """
    rtol = 1e-10
    atol = 0
    # Input core_profiles reading and config loading
    config = self._get_config_dict('test_iterhybrid_predictor_corrector.py')
    torax_config = model_config.ToraxConfig.from_dict(config)
    # Run Sim
    (
        initial_state,
        post_processed_outputs,
        step_fn,
    ) = run_simulation.prepare_simulation(torax_config)

    state_history, post_processed_outputs_history, _ = run_loop.run_loop(
        initial_state=initial_state,
        initial_post_processed_outputs=post_processed_outputs,
        step_fn=step_fn,
        log_timestep_info=False,
        progress_bar=False,
    )
    # Fill an IDS with the output of the simulation, from the initial time step.
    sim_state = state_history[0]
    core_profiles = [sim_state.core_profiles]
    core_sources = [sim_state.core_sources]
    geometry = [sim_state.geometry]
    times = [sim_state.t]
    post_processed_outputs = [post_processed_outputs_history[0]]
    filled_ids = output_core_profiles.core_profiles_to_IMAS(
        step_fn.runtime_params_provider,
        torax_config,
        post_processed_outputs,
        core_profiles,
        core_sources,
        geometry,
        times,
    )
    # Modifying the input config profiles_conditions class with the IDS filled
    # from the previous simulation.
    core_profiles_conditions = input_core_profiles.profile_conditions_from_IMAS(
        filled_ids
    )
    config['profile_conditions'] = {
        **core_profiles_conditions,
    }
    new_config = model_config.ToraxConfig.from_dict(config)
    # Running simulation again with the new config
    imas_xr, imas_results = torax.run_simulation(new_config, progress_bar=False)

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

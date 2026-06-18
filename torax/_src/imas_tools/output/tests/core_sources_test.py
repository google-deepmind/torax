# Copyright 2026 DeepMind Technologies Limited
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

"""Tests for mapping TORAX sources to IMAS core_sources IDS."""

import pathlib

from absl.testing import absltest
import imas
import numpy as np
from torax._src.imas_tools.input import loader
from torax._src.imas_tools.input.core_sources import sources_from_IMAS
from torax._src.imas_tools.output.core_sources import _TORAX_SOURCE_NAME_TO_IMAS_SOURCE_ID
from torax._src.imas_tools.output.core_sources import core_sources_to_IMAS
from torax._src.orchestration import run_loop
from torax._src.orchestration import run_simulation
from torax._src.output_tools import output
from torax._src.test_utils import sim_test_case
from torax._src.torax_pydantic import model_config


class CoreSourcesTest(sim_test_case.SimTestCase):

  def test_save_sources_to_IMAS(self):
    """Checks that multiple time slices of sources can be saved into an IDS."""
    rtol = 1e-6
    atol = 1e-8

    config = self._get_config_dict("test_iterhybrid_rampup_short.py")
    config["numerics"]["t_final"] = 0.02
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

    # Save output sources to IDS
    filled_ids = core_sources_to_IMAS(
        torax_config,
        state_history.post_processed_outputs,
        state_history.core_profiles,
        state_history.source_profiles,
        state_history.geometries,
        state_history.times,
    )

    # Validate the output mapping
    filled_ids.validate()
    np.testing.assert_allclose(filled_ids.time, state_history.times)
    source_names = [str(src.identifier.name) for src in filled_ids.source]
    self.assertIn("fusion", source_names)
    self.assertIn("pellet", source_names)
    self.assertIn("gas_puff", source_names)
    self.assertIn("generic_current", source_names)
    self.assertIn("generic_heat", source_names)
    self.assertIn("generic_particle", source_names)
    # self.assertIn("collisional_equipartition", source_names)

    # Check fusion source IMAS output against TORAX for the first time slice.
    for src in filled_ids.source:
      if str(src.identifier.name) == "fusion":
        expected_T_e = state_history.source_profiles[0].T_e["fusion"]
        actual_T_e = src.profiles_1d[0].electrons.energy[1:-1]
        np.testing.assert_allclose(
            actual_T_e, expected_T_e, rtol=rtol, atol=atol
        )

        expected_T_i = state_history.source_profiles[0].T_i["fusion"]
        actual_T_i = src.profiles_1d[0].total_ion_energy[1:-1]
        np.testing.assert_allclose(
            actual_T_i, expected_T_i, rtol=rtol, atol=atol
        )
        break

  def test_round_trip_with_IMAS_sources(self):
    """Tests a round trip (IMAS -> TORAX -> IMAS) using the input core_sources test IDS."""
    rtol = 1e-6
    atol = 1e-8

    directory = pathlib.Path(__file__).parent.parent.parent / "input" / "tests"
    ids_in = loader.load_imas_data(
        "core_sources_ddv4.nc", "core_sources", directory=directory
    )

    # Update TORAX simulation config with IMAS sources
    imas_sources = sources_from_IMAS(ids_in, load_only_external_sources=True)

    config = self._get_config_dict("test_iterhybrid_rampup_short.py")
    config["sources"] |= imas_sources
    torax_config = model_config.ToraxConfig.from_dict(config)

    # Prepare simulation
    initial_state, post_processed_outputs, _ = (
        run_simulation.prepare_simulation(torax_config)
    )

    # Save initial state back to IMAS IDS
    core_profiles = [initial_state.core_profiles]
    core_sources = [initial_state.core_sources]
    geometry = [initial_state.geometry]
    times = np.array([initial_state.t])
    post_processed_outputs_seq = [post_processed_outputs]

    filled_ids = core_sources_to_IMAS(
        torax_config,
        post_processed_outputs_seq,
        core_profiles,
        core_sources,
        geometry,
        times,
    )

    source_names = [str(src.identifier.name) for src in filled_ids.source]
    self.assertIn("fusion", source_names)
    self.assertIn("pellet", source_names)
    self.assertIn("gas_puff", source_names)
    self.assertIn("collisional_equipartition", source_names)
    self.assertIn("custom_1", source_names)
    self.assertIn("custom_2", source_names)
    self.assertIn("custom_3", source_names)
    self.assertIn("ec", source_names)
    self.assertIn("ic", source_names)
    # Check that the exported IC and EC IDS profiles match the input IDS ones.
    for src_out in filled_ids.source:
      src_name = str(src_out.identifier.name)
      if src_name not in ["ec", "ic"]:
        continue

      grid_out = src_out.profiles_1d[0].grid.rho_tor_norm
      expected_T_e = np.zeros_like(grid_out)
      expected_T_i = np.zeros_like(grid_out)
      expected_psi = np.zeros_like(grid_out)

      # Combine source profiles in ids_in corresponding to the same source
      # as there are 2 EC sources (one for each launcher) for example.
      for source in ids_in.source:
        if str(source.identifier.name) == src_name:
          p1d = source.profiles_1d[0]
          grid_in = p1d.grid.rho_tor_norm

          expected_T_e += np.interp(grid_out, grid_in, p1d.electrons.energy)
          expected_T_i += np.interp(grid_out, grid_in, p1d.total_ion_energy)
          expected_psi += np.interp(grid_out, grid_in, p1d.j_parallel)

      # Check output profiles against input ones
      actual = src_out.profiles_1d[0].electrons.energy
      np.testing.assert_allclose(
          actual[1:-1], expected_T_e[1:-1], rtol=rtol, atol=atol
      )
      actual = src_out.profiles_1d[0].total_ion_energy
      np.testing.assert_allclose(
          actual[1:-1], expected_T_i[1:-1], rtol=rtol, atol=atol
      )
      actual = src_out.profiles_1d[0].j_parallel
      np.testing.assert_allclose(
          actual[1:-1], expected_psi[1:-1], rtol=rtol, atol=atol
      )


if __name__ == "__main__":
  absltest.main()

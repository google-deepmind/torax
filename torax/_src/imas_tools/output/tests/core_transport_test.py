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

"""Tests for the output core_transport module."""

from absl.testing import absltest
import imas
import numpy as np
from torax._src.geometry import geometry as geometry_lib
from torax._src.imas_tools.output import core_transport as output_core_transport
from torax._src.orchestration import run_loop
from torax._src.orchestration import run_simulation
from torax._src.output_tools import output
from torax._src.test_utils import sim_test_case
from torax._src.torax_pydantic import model_config


class CoreTransportTest(sim_test_case.SimTestCase):

  def test_save_core_transport_to_imas(self):
    """Test to check that multiple time slices can be saved into an IDS."""
    # Run sim
    config = self._get_config_dict("test_iterhybrid_rampup_short.py")
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

    post_processed_outputs = state_history.post_processed_outputs
    core_profiles = state_history.core_profiles
    core_transport = state_history.core_transport
    geometry = state_history.geometries
    times = state_history.times

    ids_out = imas.IDSFactory().core_transport()
    filled_ids = output_core_transport.core_transport_to_IMAS(
        torax_config,
        post_processed_outputs,
        core_profiles,
        core_transport,
        geometry,
        times,
        ids_out,
    )
    filled_ids.validate()

    # Compare values
    t_idx = 0
    model_combined = None
    for model in filled_ids.model:
      if model.identifier.name == "combined":
        model_combined = model
        break

    self.assertIsNotNone(model_combined)
    profiles_1d = model_combined.profiles_1d[t_idx]
    ct_state = core_transport[t_idx]

    expected_chi_e = output.extend_cell_grid_to_boundaries(
        [geometry_lib.face_to_cell(ct_state.chi_face_el_total)],
        np.array([ct_state.chi_face_el_total]),
    )[0]

    np.testing.assert_allclose(
        profiles_1d.electrons.energy.d, expected_chi_e, atol=1e-10, rtol=1e-5
    )


if __name__ == "__main__":
  absltest.main()

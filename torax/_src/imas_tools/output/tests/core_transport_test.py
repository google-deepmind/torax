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
import torax
from torax._src.imas_tools.output import core_transport as output_core_transport
from torax._src.test_utils import sim_test_case
from torax._src.torax_pydantic import config


class CoreTransportTest(sim_test_case.SimTestCase):

  def test_save_core_transport_to_imas(self):
    """Checks saving multiple time slices to core/plasma_transport IDSs."""
    config_dict = self._get_config_dict('test_iterhybrid_rampup_short.py')
    torax_config = config.ToraxConfig.from_dict(config_dict)
    state_history = torax.run_simulation(
        torax_config,
        log_timestep_info=False,
        progress_bar=False,
    )

    core_profiles = state_history.core_profiles
    core_transport = state_history.core_transport
    geometry = state_history.geometries
    times = state_history.times

    for ids_name in ('core_transport', 'plasma_transport'):
      ids_out = getattr(imas.IDSFactory(), ids_name)()
      filled_ids = output_core_transport.core_transport_to_IMAS(
          torax_config,
          core_profiles,
          core_transport,
          geometry,
          times,
          ids_out,
      )
      filled_ids.validate()

      models_by_name = {
          str(model.identifier.name): model for model in filled_ids.model
      }
      self.assertEqual(
          set(models_by_name.keys()),
          {'combined', 'transport_solver', 'anomalous', 'neoclassical'},
      )
      self.assertEqual(models_by_name['combined'].code.name, 'TORAX')
      self.assertEqual(models_by_name['transport_solver'].code.name, 'TORAX')
      self.assertEqual(models_by_name['anomalous'].code.name, 'qlknn')
      self.assertEqual(models_by_name['neoclassical'].code.name, 'TORAX sauter')

      expected = core_transport[-1].turbulent.core
      profiles_1d = models_by_name['anomalous'].profiles_1d[-1]
      np.testing.assert_allclose(
        profiles_1d.electrons.energy.d,
        expected.chi_face_el,
        atol=1e-10,
        rtol=1e-5,
      )
      np.testing.assert_allclose(
        profiles_1d.total_ion_energy.d,
        expected.chi_face_ion,
        atol=1e-10,
        rtol=1e-5,
      )
      np.testing.assert_allclose(
        profiles_1d.electrons.particles.d,
        expected.d_face_el,
        atol=1e-10,
        rtol=1e-5,
      )
      np.testing.assert_allclose(
        profiles_1d.electrons.particles.v,
        expected.v_face_el,
        atol=1e-10,
        rtol=1e-5,
      )


if __name__ == '__main__':
  absltest.main()

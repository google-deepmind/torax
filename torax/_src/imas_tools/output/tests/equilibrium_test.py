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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax._src.imas_tools.input import equilibrium as input_equilibrium
from torax._src.imas_tools.input import loader
from torax._src.imas_tools.output import equilibrium as output_equilibrium
from torax._src.orchestration import run_simulation
from torax._src.test_utils import sim_test_case
from torax._src.torax_pydantic import model_config


class EquilibriumTest(sim_test_case.SimTestCase):
  """Unit tests for the `toraximastools.equilibrium` module."""

  @parameterized.parameters([
      dict(
          config_name='test_iterhybrid_predictor_corrector_imas.py',
          rtol=2e-2,
          atol=1e-8,
      ),
  ])
  def test_IMAS_geometry_input_output_roundtrip(
      self,
      config_name,
      rtol: float | None = None,
      atol: float | None = None,
  ):
    """Test the default IMAS geometry can be built and converted back to IDS."""
    # Input equilibrium reading.
    config_dict = self._get_config_dict(config_name)
    equilibrium_in = loader.load_imas_data(
        config_dict['geometry']['imas_filepath'],
        loader.EQUILIBRIUM,
    )
    # Build TORAXSimState object and write output to equilibrium IDS.
    # Improve resolution to compare the input without losing too much
    # information.
    config_dict['geometry']['n_rho'] = (
        len(equilibrium_in.time_slice[0].profiles_1d.rho_tor_norm) - 1
    )  # rho_tor_norm is the face grid, subtract one for cell grid.

    config_dict['geometry']['equilibrium_object'] = equilibrium_in
    config_dict['geometry']['imas_filepath'] = None
    # Set a current as consistent as possible with the "unscaled" current
    # corresponding to the input psi.
    config_dict['profile_conditions']['Ip'] = 11.35e6
    torax_config = model_config.ToraxConfig.from_dict(config_dict)
    _, sim_state, post_processed_outputs, _ = run_simulation.prepare_simulation(
        torax_config
    )

    equilibrium_out = output_equilibrium.torax_state_to_imas_equilibrium(
        sim_state,
        post_processed_outputs,
        equilibrium_in=equilibrium_in,
    )

    rhon_out = equilibrium_out.time_slice[0].profiles_1d.rho_tor_norm
    rhon_in = equilibrium_in.time_slice[0].profiles_1d.rho_tor_norm
    for attr1, attr2 in [
        ('profiles_1d', 'phi'),
        ('profiles_1d', 'psi'),
        ('profiles_1d', 'q'),
        ('profiles_1d', 'gm1'),
        ('profiles_1d', 'gm2'),
        ('profiles_1d', 'gm3'),
        ('profiles_1d', 'gm7'),
        ('profiles_1d', 'volume'),
        ('profiles_1d', 'dvolume_dpsi'),
        ('profiles_1d', 'dpsi_drho_tor'),
        ('profiles_1d', 'f'),
        # goes through too many derivatives and calculations for accurate test
        # ('profiles_1d', 'j_phi'),
    ]:
      # Compare the output IDS with the input one.
      var_in = getattr(getattr(equilibrium_in.time_slice[0], attr1), attr2)
      var_out = getattr(getattr(equilibrium_out.time_slice[0], attr1), attr2)
      n = int(var_in.size / 10)
      print(f'{attr1} {attr2}')
      print(f'{var_in.shape}')
      print(f'{var_out.shape}')
      np.testing.assert_allclose(
          np.interp(rhon_in, rhon_out, var_out)[n:-n],
          var_in[n:-n],
          rtol=rtol,
          atol=atol,
          err_msg=f'{attr1} {attr2} failed',
      )

    np.testing.assert_allclose(
        equilibrium_out.time_slice[0].boundary.outline.r,
        equilibrium_in.time_slice[0].boundary.outline.r,
    )
    np.testing.assert_allclose(
        equilibrium_out.time_slice[0].boundary.outline.z,
        equilibrium_in.time_slice[0].boundary.outline.z,
    )


if __name__ == '__main__':
  absltest.main()

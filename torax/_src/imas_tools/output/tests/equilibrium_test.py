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
from torax._src.imas_tools.input import loader
from torax._src.imas_tools.output import core_profiles as output_core_profiles
from torax._src.imas_tools.output import equilibrium as output_equilibrium
from torax._src.orchestration import run_simulation
from torax._src.test_utils import sim_test_case
from torax._src.torax_pydantic import config


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
        'equilibrium',
    )
    # Build TORAXSimState object and write output to equilibrium IDS.
    # Improve resolution to compare the input without losing too much
    # information.
    config_dict['geometry']['n_rho'] = (
        len(equilibrium_in.time_slice[0].profiles_1d.rho_tor_norm) - 1
    )  # rho_tor_norm is the face grid, subtract one for cell grid.

    config_dict['geometry']['equilibrium_object'] = equilibrium_in
    config_dict['geometry']['imas_filepath'] = None
    # Set the current to be consistent with the input psi.
    config_dict['profile_conditions']['I_p'] = 11.75e6
    torax_config = config.ToraxConfig.from_dict(config_dict)
    sim_state, post_processed_outputs, _ = run_simulation.prepare_simulation(
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
      np.testing.assert_allclose(  # pyrefly: ignore[no-matching-overload]
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

  def test_vacuum_toroidal_field_keeps_input_r0(self):
    """Tests that the vacuum field is written back at the input r0."""
    config_dict = self._get_config_dict(
        'test_iterhybrid_predictor_corrector_imas.py'
    )
    equilibrium_in = loader.load_imas_data(
        config_dict['geometry']['imas_filepath'],
        'equilibrium',
    )
    # Move the reference radius away from R_major, keeping r0 * b0 fixed.
    vtf = equilibrium_in.vacuum_toroidal_field
    r0_in = float(vtf.r0) + 0.5
    b0_in = float(vtf.b0[0]) * float(vtf.r0) / r0_in
    vtf.r0 = r0_in
    vtf.b0 = [b0_in]
    config_dict['geometry']['equilibrium_object'] = equilibrium_in
    config_dict['geometry']['imas_filepath'] = None
    torax_config = config.ToraxConfig.from_dict(config_dict)
    sim_state, post_processed_outputs, _ = run_simulation.prepare_simulation(
        torax_config
    )
    self.assertGreater(abs(sim_state.geometry.R_major - r0_in), 0.1)

    equilibrium_out = output_equilibrium.torax_state_to_imas_equilibrium(
        sim_state, post_processed_outputs
    )
    core_profiles_out = output_core_profiles.core_profiles_to_IMAS(
        torax_config,
        [post_processed_outputs],
        [sim_state.core_profiles],
        [sim_state.core_sources],
        [sim_state.geometry],
        [sim_state.t],
    )
    for ids in (equilibrium_out, core_profiles_out):
      np.testing.assert_allclose(ids.vacuum_toroidal_field.r0, r0_in)
      np.testing.assert_allclose(
          np.abs(ids.vacuum_toroidal_field.b0), [abs(b0_in)], rtol=1e-6
      )


if __name__ == '__main__':
  absltest.main()

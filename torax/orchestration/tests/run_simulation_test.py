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
from collections.abc import Sequence
import os

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax import output
from torax.orchestration import run_simulation
from torax.tests.test_lib import sim_test_case
import xarray as xr


_ALL_PROFILES = ('temp_ion', 'temp_el', 'psi', 'q_face', 's_face', 'ne')


class RunSimulationTest(sim_test_case.SimTestCase):

  @parameterized.named_parameters(
      # Tests Newton-Raphson nonlinear solver for ITER-hybrid-like-config
      (
          'test_iterhybrid_newton',
          'test_iterhybrid_newton.py',
          _ALL_PROFILES,
          5e-7,
      ),
      # Tests current and density rampup for for ITER-hybrid-like-config
      # using Newton-Raphson. Only case which reverts to coarse_tol for several
      # timesteps (with negligible impact on results compared to full tol).
      (
          'test_iterhybrid_rampup',
          'test_iterhybrid_rampup.py',
          _ALL_PROFILES,
          0,
          1e-6,
      ),
      # Tests time-dependent circular geometry.
      (
          'test_time_dependent_circular_geo',
          'test_time_dependent_circular_geo.py',
          _ALL_PROFILES,
          0,
      ),
  )
  def test_run_simulation(
      self,
      config_name: str,
      profiles: Sequence[str],
      rtol: float | None = None,
      atol: float | None = None,
  ):
    self._test_run_simulation(
        config_name=config_name,
        profiles=profiles,
        rtol=rtol,
        atol=atol,
    )

  def test_change_config(self):
    torax_config = self._get_torax_config('test_iterhybrid_newton.py')
    history = run_simulation.run_simulation(torax_config)

    original_value = torax_config.runtime_params.profile_conditions.nbar
    new_value = original_value.value * 1.1

    torax_config.update_fields(
        {'runtime_params.profile_conditions.nbar': new_value}
    )
    new_history = run_simulation.run_simulation(torax_config)

    self.assertFalse(
        np.array_equal(
            history.core_profiles.ne.value[-1],
            new_history.core_profiles.ne.value[-1],
        )
    )

  def test_restart(self):
    test_config_state_file = 'test_iterhybrid_rampup.nc'
    restart_config = 'test_iterhybrid_rampup_restart.py'

    torax_config = self._get_torax_config(restart_config)
    history = run_simulation.run_simulation(torax_config)

    data_tree_restart = history.simulation_output_to_xr()

    # Load the reference dataset.
    datatree_ref = output.load_state_file(
        os.path.join(self.test_data_dir, test_config_state_file)
    )

    # Stitch the restart state file to the beginning of the reference dataset.
    datatree_new = output.stitch_state_files(
        torax_config.restart, data_tree_restart
    )

    # Check equality for all time-dependent variables.
    def check_equality(ds1: xr.Dataset, ds2: xr.Dataset):
      for var_name in ds1.data_vars:
        if 'time' in ds1[var_name].dims:
          with self.subTest(var_name=var_name):
            np.testing.assert_allclose(
                ds1[var_name].values,
                ds2[var_name].values,
                err_msg=f'Mismatch for {var_name} in restart test',
                rtol=1e-6,
            )

    xr.map_over_datasets(check_equality, datatree_ref, datatree_new)


if __name__ == '__main__':
  absltest.main()

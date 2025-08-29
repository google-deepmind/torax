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


import os
from typing import Optional

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torax
from torax._src.imas_tools.input import core_profiles
from torax._src.imas_tools.input import loader
from torax._src.orchestration import run_simulation
from torax._src.test_utils import sim_test_case
from torax._src.torax_pydantic import model_config


class CoreProfilesTest(sim_test_case.SimTestCase):
  """Unit tests for torax.torax_imastools.input.core_profiles.py"""

  def test_offset_time(
      self,
  ):
    """Unit tests to check the t_initial optional args offset correctly the
    time array."""
    offset = 100.0
    path = "core_profiles_ddv4_iterhybrid_rampup_conditions.nc"
    dir = os.path.join(torax.__path__[0], "data/third_party/imas_data")
    ids_in = loader.load_imas_data(path, loader.CORE_PROFILES, dir)
    core_profiles_conditions = core_profiles.profile_conditions_from_IMAS(
        ids_in,
        t_initial=offset,
    )
    t_in = np.array([
        float(ids_in.profiles_1d[i].time)
        for i in range(len(ids_in.profiles_1d))
    ])
    t_out = np.array(list(core_profiles_conditions["Ip"][0]))
    np.testing.assert_equal(t_out, t_in + 100.0)

  @parameterized.parameters([
      dict(config_name="test_iterhybrid_rampup_short.py", rtol=1e-6, atol=1e-8),
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
    points.
    """
    if rtol is None:
      rtol = self.rtol
    if atol is None:
      atol = self.atol
    # Input core_profiles reading and config loading
    config = self._get_config_dict(config_name)
    path = (  # Using this as input instead of rampup_conditions because it has more radial resolution.
        "core_profiles_15MA_DT_50_50_flat_top_slice.nc"
    )
    dir = os.path.join(torax.__path__[0], "data/third_party/imas_data")
    core_profiles_in = loader.load_imas_data(path, loader.CORE_PROFILES, dir)
    rhon_in = core_profiles_in.profiles_1d[0].grid.rho_tor_norm

    # Modifying the input config profiles_conditions class
    core_profiles_conditions = core_profiles.profile_conditions_from_IMAS(
        core_profiles_in,
        t_initial=0.0,
    )
    config["geometry"]["n_rho"] = 200
    config["profile_conditions"] = {
        **core_profiles_conditions,
    }
    torax_config = model_config.ToraxConfig.from_dict(config)

    # Init sim from config
    _, sim_state, _, _ = run_simulation.prepare_simulation(torax_config)

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
        err_msg="Te profile failed",
    )
    np.testing.assert_allclose(
        init_core_profiles.T_i.value * 1e3,
        np.interp(
            cell_centers, rhon_in, core_profiles_in.profiles_1d[0].t_i_average
        ),
        rtol=rtol,
        atol=atol,
        err_msg="Ti profile failed",
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
        err_msg="ne profile failed",
    )
    np.testing.assert_allclose(
        init_core_profiles.psi.value,
        np.interp(
            cell_centers, rhon_in, core_profiles_in.profiles_1d[0].grid.psi
        ),
        rtol=rtol,
        atol=atol,
        err_msg="psi profile failed",
    )


if __name__ == "__main__":
  absltest.main()
